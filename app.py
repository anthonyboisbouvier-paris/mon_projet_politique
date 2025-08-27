#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube -> Subtitles (VTT, FR) -> (opt) pyannote diarization -> JSON minimal

Sortie JSON:
{
  "text": "<transcript complet (nettoyé)>",
  "utterances": [
     {"speaker": "SPEAKER_00", "start": 12.34, "end": 18.90, "text": "..."},
     ...
  ]
}

Points clés:
- Pas de Whisper. On utilise YT-DLP pour récupérer les sous-titres VTT.
- Préférence pour les sous-titres manuels FR. Fallback auto uniquement si --allow-auto.
- Diarization pyannote (si HUGGINGFACE_TOKEN / HF_TOKEN dispo), sinon fallback "SPEAKER_00" unique.
- Nettoyage anti-doublons (sans IA) sur tout: transcript et utterances (DEDUP-SIMPLE).
- Option d’extrait: --start HH:MM:SS --duration N (seconds), pour accélérer.

Exemples:
  # Healthcheck
  python app.py --healthcheck

  # Full vidéo (FR manuels uniquement)
  python app.py --input "https://www.youtube.com/watch?v=XXXX" --output outputs/out.json

  # Autoriser auto-subs si pas de FR manuels
  python app.py --input "https://www.youtube.com/watch?v=XXXX" --allow-auto --output outputs/out.json

  # Traiter seulement un clip (5 minutes à partir de 00:05:00)
  python app.py --input "https://www.youtube.com/watch?v=XXXX" --start 00:05:00 --duration 300 --output outputs/clip.json

  # Utiliser un .vtt local à la place du download
  python app.py --input ".\mon_audio.mp4" --vtt ".\sous_titres.fr.vtt" --output outputs/out.json
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import unicodedata

# -------------------- Logs --------------------
def print_step(msg: str) -> None:
    print(f"[STEP] {msg}", flush=True)

def print_check(msg: str) -> None:
    print(f"[CHECK] {msg}", flush=True)

def fail(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)

# ----------------- Utilitaires ----------------
def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        fail("ffmpeg non trouvé dans le PATH.")

def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

def parse_hhmmss(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s = s.strip()
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + float(sec)
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + float(sec)
        return float(s)
    except Exception:
        fail(f"Format temps invalide: {s}")

def to_seconds(ts: str) -> float:
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

# -------------- Téléchargements ---------------
def download_youtube_audio(url: str, out_dir: Path) -> Path:
    from yt_dlp import YoutubeDL
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(title).200s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded = ydl.prepare_filename(info)
    return Path(downloaded)

def _find_best_vtt(out_dir: Path, langs_pref: List[str]) -> Optional[Path]:
    # cherche exact .<lang>.vtt puis *.vtt
    for lang in langs_pref:
        hits = sorted(out_dir.glob(f"*.{lang}.vtt"))
        if hits:
            return hits[0]
    vtts = sorted(out_dir.glob("*.vtt"))
    return vtts[0] if vtts else None

def download_youtube_subs(url: str, out_dir: Path, langs_pref: List[str], allow_auto: bool) -> Optional[Path]:
    from yt_dlp import YoutubeDL, utils
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Essai strict: manuels uniquement
    ydl_opts_manual = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": False,
        "subtitleslangs": langs_pref,
        "subtitlesformat": "vtt",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "extractor_retries": 3,
        "sleep_interval_requests": 1,
        "outtmpl": str(out_dir / "%(title).200s.%(ext)s"),
    }
    try:
        with YoutubeDL(ydl_opts_manual) as ydl:
            ydl.extract_info(url, download=True)
        vtt = _find_best_vtt(out_dir, langs_pref)
        if vtt:
            return vtt
    except utils.DownloadError:
        pass

    # 2) Fallback auto-subs (si autorisé)
    if allow_auto:
        ydl_opts_auto = dict(ydl_opts_manual)
        ydl_opts_auto["writeautomaticsub"] = True
        ydl_opts_auto["writesubtitles"] = True
        try:
            with YoutubeDL(ydl_opts_auto) as ydl:
                ydl.extract_info(url, download=True)
            vtt = _find_best_vtt(out_dir, langs_pref)
            if vtt:
                return vtt
        except utils.DownloadError:
            return None
    return None

# ------------------ Audio I/O -----------------
def convert_to_wav_mono16k(src: Path, dst: Path, start: Optional[float] = None, duration: Optional[int] = None) -> Path:
    cmd = ["ffmpeg", "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(src)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-ac", "1", "-ar", "16000", "-vn", str(dst)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        fail(f"Echec conversion audio (ffmpeg): {p.stderr[:400]}")
    return dst

# --------------- Parsing du VTT ---------------
def parse_vtt_captions(vtt_path: Path) -> List[Dict[str, Any]]:
    import webvtt
    caps: List[Dict[str, Any]] = []
    for cue in webvtt.read(str(vtt_path)):
        text = " ".join(cue.text.strip().split())
        if not text:
            continue
        caps.append({"start": to_seconds(cue.start), "end": to_seconds(cue.end), "text": text})
    return caps

def parse_vtt_to_words(vtt_path: Path) -> List[Dict[str, Any]]:
    import webvtt
    words: List[Dict[str, Any]] = []
    for cue in webvtt.read(str(vtt_path)):
        text = " ".join(cue.text.strip().split())
        if not text:
            continue
        start, end = to_seconds(cue.start), to_seconds(cue.end)
        toks = text.split()
        dur = max(1e-3, end - start)
        step = dur / max(1, len(toks))
        for i, tok in enumerate(toks):
            ws = start + i * step
            we = start + (i + 1) * step
            words.append({"start": ws, "end": we, "text": tok})
    return words

# --------- DEDUP POST-PROCESSING (no-IA) -----
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _norm(s: str) -> str:
    s = _strip_accents(s.lower())
    s = re.sub(r"[^\w\s']", " ", s)  # lettres/chiffres/_ + apostrophe
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _words_with_norms(text: str) -> Tuple[List[str], List[str]]:
    words = re.findall(r"\S+", text)
    norms = [_norm(w) for w in words]
    return words, norms

def _antiwheel(words: List[str], norms: List[str], lens=(12, 9, 6, 4, 3)) -> Tuple[List[str], List[str]]:
    out_w, out_n = [], []
    i = 0
    L = len(words)
    while i < L:
        skipped = False
        for n in lens:
            if len(out_n) >= n and i + n <= L and out_n[-n:] == norms[i:i+n]:
                i += n
                skipped = True
                break
        if skipped:
            continue
        out_w.append(words[i]); out_n.append(norms[i]); i += 1
    return out_w, out_n

def _dedupe_sentences(text: str) -> str:
    parts = re.split(r"([.!?…]+)", text)
    out, seen = [], set()
    for i in range(0, len(parts), 2):
        sent = parts[i].strip()
        if not sent:
            continue
        end = parts[i+1] if i+1 < len(parts) else ""
        key = _norm(sent)
        if key and key not in seen:
            out.append(sent + (end if end else ""))
            seen.add(key)
    return " ".join(out).strip()

def _squeeze_repeated_ngrams(text: str, max_n: int = 6, passes: int = 2) -> str:
    s = text
    for _ in range(passes):
        for n in range(max_n, 1, -1):
            pattern = r"\b(" + r"(?:\w+\s+)"+str(n-1) + r"\w+)\s+\1\b"
            s = re.sub(pattern, r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_text_intra(text: str, aggressive: bool = True) -> str:
    words, norms = _words_with_norms(text)
    words, norms = _antiwheel(words, norms)          # rouleaux adjacents
    s = " ".join(words)
    s = _dedupe_sentences(s)                         # doublons phrase-internes
    if aggressive:
        s = _squeeze_repeated_ngrams(s, max_n=6, passes=2)
    return s

def longest_overlap(prev_norm_tokens: List[str],
                    curr_norm_tokens: List[str],
                    max_overlap: int = 20,
                    min_overlap: int = 3) -> int:
    upto = min(max_overlap, len(prev_norm_tokens), len(curr_norm_tokens))
    for n in range(upto, min_overlap - 1, -1):
        if prev_norm_tokens[-n:] == curr_norm_tokens[:n]:
            return n
    return 0

def dedupe_utterances_simple(utterances: List[Dict[str, Any]],
                             aggressive: bool = True,
                             window_seconds: float = 10.0,
                             max_overlap: int = 20,
                             min_overlap: int = 3) -> List[Dict[str, Any]]:
    """
    Nettoie les redites intra-phrase, coupe les reprises entre locuteurs,
    supprime les doublons exacts proches dans une fenêtre temporelle.
    """
    cleaned: List[Dict[str, Any]] = []
    recent = deque()   # (key_norm, start_time)
    prev_norm_tokens: List[str] = []

    for u in utterances:
        text = u.get("text", "")
        if not text:
            continue

        # 1) Intra-utterance
        s = clean_text_intra(text, aggressive=aggressive)
        if not s:
            continue

        # 2) Inter-utterance (chevauchement suffixe/préfixe)
        _, curr_norms = _words_with_norms(s)
        n = longest_overlap(prev_norm_tokens, curr_norms,
                            max_overlap=max_overlap, min_overlap=min_overlap)
        if n > 0:
            words2, norms2 = _words_with_norms(s)
            s = " ".join(words2[n:])
            curr_norms = norms2[n:]
            if not s:
                continue

        # 3) Filtre fenêtre (exact match proche)
        key = " ".join(curr_norms)
        start = float(u.get("start", 0.0))
        keep = True
        for prev_key, prev_start in list(recent):
            if key == prev_key and abs(start - prev_start) <= window_seconds:
                keep = False
                break
        if not keep:
            continue

        cleaned.append({**u, "text": s})
        recent.append((key, start))
        # purge fenêtre glissante
        while recent and start - recent[0][1] > window_seconds:
            recent.popleft()
        prev_norm_tokens = curr_norms

    return cleaned

def dedupe_full_text_from_captions(captions: List[Dict[str, Any]]) -> str:
    raw = " ".join(c["text"] for c in captions)
    s = clean_text_intra(raw, aggressive=True)
    return s

# ------------------ Diarization ----------------
def diarize_pyannote(wav_path: Path, num_speakers: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        print("[WARN] Pas de token HF -> diarization désactivée.")
        return None
    try:
        import torch
        from pyannote.audio import Pipeline
    except Exception as e:
        print(f"[WARN] pyannote.audio indisponible ({e}) -> diarization désactivée.")
        return None

    print_step("Diarization (pyannote.audio)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        ).to(device)
        print_check(f"pyannote device: {device.type}")
    except Exception as e:
        print(f"[WARN] Echec chargement pipeline: {e} -> diarization désactivée.")
        return None

    infer_args = {"audio": str(wav_path)}
    diar = pipeline(infer_args, num_speakers=num_speakers) if num_speakers is not None else pipeline(infer_args)
    segments: List[Dict[str, Any]] = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
        for turn, _, speaker in diar.itertracks(yield_label=True)
    ]
    segments.sort(key=lambda x: x["start"])
    print_check(f"Diarization ok: {len(segments)} segments.")
    return segments

# ------------- Alignement mots ↔ spk -----------
def align_words_to_diarization(words: List[Dict[str, Any]], segs: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Si segs est None: retourne une seule utterance (SPEAKER_00) avec le texte complet.
    Sinon, assigne chaque mot au speaker du segment recouvrant son milieu; fusionne consécutifs.
    """
    if not words:
        return []

    if not segs:
        txt = " ".join(w["text"] for w in words).strip()
        return [{
            "speaker": "SPEAKER_00",
            "start": float(words[0]["start"]),
            "end": float(words[-1]["end"]),
            "text": txt
        }]

    segs = sorted(segs, key=lambda s: (s["start"], s["end"]))

    def speaker_at(t: float) -> str:
        for s in segs:
            if s["start"] <= t <= s["end"]:
                return s["speaker"]
        # plus proche
        nearest = min(segs, key=lambda s: min(abs(s["start"] - t), abs(s["end"] - t)))
        return nearest["speaker"]

    utt: List[Dict[str, Any]] = []
    cur_spk = None
    cur_start: Optional[float] = None
    buf: List[str] = []

    for w in words:
        mid = (w["start"] + w["end"]) / 2.0
        spk = speaker_at(mid)
        if cur_spk is None:
            cur_spk, cur_start, buf = spk, w["start"], [w["text"]]
        elif spk == cur_spk:
            buf.append(w["text"])
        else:
            text = re.sub(r"\s+", " ", " ".join(buf)).strip()
            if text:
                utt.append({"speaker": cur_spk, "start": float(cur_start), "end": float(w["start"]), "text": text})
            cur_spk, cur_start, buf = spk, w["start"], [w["text"]]

    if buf:
        text = re.sub(r"\s+", " ", " ".join(buf)).strip()
        utt.append({"speaker": cur_spk, "start": float(cur_start), "end": float(words[-1]["end"]), "text": text})

    # fusion consécutifs même speaker
    merged: List[Dict[str, Any]] = []
    for u in utt:
        if merged and merged[-1]["speaker"] == u["speaker"]:
            merged[-1]["text"] = (merged[-1]["text"] + " " + u["text"]).strip()
            merged[-1]["end"] = u["end"]
        else:
            merged.append(u)
    return merged

# ---------------------- Main -------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="YT-DLP (subs FR) -> (opt) diarization -> JSON minimal")
    parser.add_argument("--input", type=str, required=False, help="URL YouTube OU chemin fichier audio/vidéo")
    parser.add_argument("--vtt", type=str, default=None, help="Chemin d'un .vtt local (si fourni, on évite le download)")
    parser.add_argument("--output", type=str, default="result.json", help="Chemin du JSON de sortie")
    parser.add_argument("--subs-lang", type=str, default="fr,fr-FR,fr-CA", help="Langues préférées, séparées par virgule")
    parser.add_argument("--allow-auto", action="store_true", help="Autoriser auto-subs si manuels absents")
    parser.add_argument("--no-diarization", action="store_true", help="Désactiver pyannote même si token présent")
    parser.add_argument("--num-speakers", type=int, default=None, help="Fixer le nombre de locuteurs (optionnel)")
    parser.add_argument("--start", type=str, default=None, help="Début HH:MM:SS ou MM:SS (extrait à traiter)")
    parser.add_argument("--duration", type=int, default=None, help="Durée (s) de l'extrait")
    parser.add_argument("--healthcheck", action="store_true", help="Vérifie l'environnement")
    args = parser.parse_args()

    # Healthcheck
    if args.healthcheck:
        print_step("Healthcheck...")
        if shutil.which("ffmpeg"): print_check("ffmpeg OK")
        else: fail("ffmpeg manquant")
        try: import yt_dlp; print_check("yt-dlp OK")
        except Exception: fail("yt-dlp manquant")
        try: import webvtt; print_check("webvtt-py OK")
        except Exception: fail("webvtt-py manquant")
        if not args.no_diarization:
            try: import pyannote.audio; print_check("pyannote.audio OK")
            except Exception: print("[WARN] pyannote.audio non importable (ok si --no-diarization)")
            if os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN"):
                print_check("HF token trouvé")
            else:
                print("[WARN] HF token absent (ok si --no-diarization)")
        print_check("Healthcheck terminé ✅")
        sys.exit(0)

    # Entrée requise si pas healthcheck
    if not args.input and not args.vtt:
        fail("--input (URL/fichier) requis ou bien --vtt (chemin VTT).")

    ensure_ffmpeg()
    langs_pref = [s.strip() for s in args.subs_lang.split(",") if s.strip()]
    start_sec = parse_hhmmss(args.start) if args.start else None

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # 1) Préparer sources média + VTT
        media_path: Optional[Path] = None
        vtt_path: Optional[Path] = None

        # a) .vtt local forcé
        if args.vtt:
            vtt_path = Path(args.vtt)
            if not vtt_path.exists():
                fail(f"VTT introuvable: {vtt_path}")
            print_check(f"VTT local: {vtt_path.name}")

        # b) YT URL ou fichier média local
        if args.input:
            if is_url(args.input):
                # Download audio
                print_step("Téléchargement audio YouTube...")
                media_path = download_youtube_audio(args.input, tmp)
                print_check(f"Fichier téléchargé: {media_path.name}")

                if not vtt_path:
                    print_step("Récupération des sous-titres .vtt (FR préférés)...")
                    vtt_path = download_youtube_subs(args.input, tmp, langs_pref, allow_auto=args.allow_auto)
                    if vtt_path is None:
                        fail("Sous-titres FR introuvables (et auto désactivé). Relance avec --allow-auto si besoin.")
                    print_check(f"Sous-titres trouvés: {vtt_path.name}")
            else:
                # Fichier local (audio/vidéo)
                media_path = Path(args.input)
                if not media_path.exists():
                    fail(f"Fichier introuvable: {media_path}")
                print_check(f"Fichier média local: {media_path.name}")
                if not vtt_path:
                    fail("Avec un fichier local, fournis --vtt pour le transcript.")

        # 2) Parse VTT -> captions + words
        captions = parse_vtt_captions(vtt_path)
        words_all = parse_vtt_to_words(vtt_path)
        full_text_clean = dedupe_full_text_from_captions(captions)

        # 3) Audio -> wav mono 16k (éventuellement extrait)
        utterances: List[Dict[str, Any]]
        if media_path:
            wav_path = tmp / "audio_16k_mono.wav"
            print_step("Conversion en WAV mono 16k...")
            convert_to_wav_mono16k(media_path, wav_path, start=start_sec, duration=args.duration)
            print_check(f"Audio prêt: {wav_path}")

            # Restreindre les mots si extrait
            if start_sec is not None and args.duration is not None:
                end_sec = start_sec + args.duration
                words_clip = [w for w in words_all if start_sec <= (w["start"] + w["end"]) / 2.0 <= end_sec]
            else:
                words_clip = words_all

            # 4) Diarization (si activée)
            spk_segments = None
            if not args.no_diarization:
                spk_segments = diarize_pyannote(wav_path, num_speakers=args.num_speakers)

            print_step("Alignement mots ↔ locuteurs...")
            utterances = align_words_to_diarization(words_clip, spk_segments)
        else:
            # Pas de média: fallback utterance unique depuis les captions
            print_step("Pas d'audio: génération d'une utterance unique depuis VTT.")
            if not captions:
                utterances = []
            else:
                utterances = [{
                    "speaker": "SPEAKER_00",
                    "start": float(captions[0]["start"]),
                    "end": float(captions[-1]["end"]),
                    "text": " ".join(c["text"] for c in captions)
                }]

        # 5) Nettoyage anti-doublons (sans IA)
        utterances = dedupe_utterances_simple(utterances, aggressive=True)

        # 6) Sauvegarde JSON minimal
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "text": full_text_clean,
            "utterances": utterances
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print_check(f"JSON écrit: {out.resolve()}")
        print("Terminé ✅")


if __name__ == "__main__":
    main()
