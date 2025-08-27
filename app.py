#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YouTube/Local → VTT → Diarization → JSON (clean)
- Préfère sous-titres manuels; fallback auto si --allow-auto
- Dédoublonnage agressif (intra-phrase et entre locuteurs)
- Exporte: meta + text + utterances
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------- Logging -----------------
def step(msg: str):   print(f"[STEP] {msg}", flush=True)
def ok(msg: str):     print(f"[CHECK] {msg}", flush=True)
def info(msg: str):   print(f"[INFO] {msg}", flush=True)
def err(msg: str):    print(f"[ERROR] {msg}", flush=True)

def fail(msg: str, code: int = 1):
    err(msg)
    sys.exit(code)

# ----------------- Utils -------------------
def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        fail("ffmpeg introuvable (installe-le puis relance).")

def to_seconds(ts: str) -> float:
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# --------------- yt-dlp I/O ---------------
def download_ytdlp_media(url: str, out_dir: Path) -> Tuple[Path, Dict[str, Any]]:
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
        media = Path(ydl.prepare_filename(info))
    return media, info

def download_ytdlp_subs(
    url: str,
    out_dir: Path,
    preferred_langs: List[str],
    allow_auto: bool,
) -> Optional[Path]:
    """
    Essaie de récupérer des VTT FR (manuels en priorité).
    Si allow_auto=False et seuls auto existent → None.
    """
    from yt_dlp import YoutubeDL, utils
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": allow_auto,
        "subtitlesformat": "vtt",
        "subtitleslangs": preferred_langs,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "outtmpl": str(out_dir / "%(title).200s.%(ext)s"),
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            _ = info  # silence lvar
    except utils.DownloadError:
        return None

    vtts = sorted(out_dir.glob("*.vtt"))
    if not vtts:
        return None

    # Heuristique: privilégier les manuels (souvent sans "auto" dans le nom)
    manual = [p for p in vtts if "auto" not in p.name.lower()]
    if manual:
        return manual[0]
    return vtts[0] if allow_auto else None

# --------------- Audio (ffmpeg) ---------------
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

# -------------- VTT parsing ----------------
def parse_vtt_words(vtt_path: Path) -> List[Dict[str, Any]]:
    import webvtt
    words: List[Dict[str, Any]] = []
    for cue in webvtt.read(str(vtt_path)):
        raw = " ".join(cue.text.strip().split())
        if not raw:
            continue
        start, end = to_seconds(cue.start), to_seconds(cue.end)
        toks = raw.split()
        dur = max(1e-3, end - start)
        step = dur / max(1, len(toks))
        for i, tok in enumerate(toks):
            ws = start + i * step
            we = start + (i + 1) * step
            words.append({"start": ws, "end": we, "text": tok})
    return words

def full_text_from_words(words: List[Dict[str, Any]]) -> str:
    return " ".join(w["text"] for w in words)

# --------------- Diarization ----------------
def diarize_pyannote(wav_path: Path, num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        fail("HUGGINGFACE_TOKEN/HF_TOKEN non défini (requis pour pyannote).")

    try:
        import torch
        from pyannote.audio import Pipeline
    except Exception as e:
        fail("pyannote.audio manquant ou incompatible. pip install 'pyannote.audio>=3.1.1'")

    step("Diarization (pyannote.audio)…")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token).to(device)
        ok(f"pyannote device: {device.type}")
    except Exception as e:
        import traceback
        print("TRACEBACK:\n", traceback.format_exc())
        fail(f"Echec chargement pipeline: {e}")

    infer_args = {"audio": str(wav_path)}
    diar = pipeline(infer_args, num_speakers=num_speakers) if num_speakers is not None else pipeline(infer_args)

    segments: List[Dict[str, Any]] = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
        for turn, _, speaker in diar.itertracks(yield_label=True)
    ]
    segments.sort(key=lambda s: (s["start"], s["end"]))
    ok(f"Diarization ok. {len(segments)} segments.")
    return segments

# -------- Alignment words ↔ speakers --------
def align_words_to_segments(words: List[Dict[str, Any]], segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not words:
        return []
    if not segs:
        txt = " ".join(w["text"] for w in words)
        return [{"speaker": "SPEAKER_00", "start": words[0]["start"], "end": words[-1]["end"], "text": txt}]

    segs = sorted(segs, key=lambda s: (s["start"], s["end"]))

    def speaker_at(t: float) -> str:
        for s in segs:
            if s["start"] <= t <= s["end"]:
                return s["speaker"]
        # nearest
        nearest = min(segs, key=lambda s: min(abs(s["start"] - t), abs(s["end"] - t)))
        return nearest["speaker"]

    utt: List[Dict[str, Any]] = []
    cur_spk = None
    cur_start = None
    buf: List[str] = []

    for w in words:
        mid = (w["start"] + w["end"]) / 2.0
        spk = speaker_at(mid)
        if cur_spk is None:
            cur_spk, cur_start, buf = spk, w["start"], [w["text"]]
        elif spk == cur_spk:
            buf.append(w["text"])
        else:
            text = " ".join(buf).strip()
            if text:
                utt.append({"speaker": cur_spk, "start": float(cur_start), "end": float(w["start"]), "text": text})
            cur_spk, cur_start, buf = spk, w["start"], [w["text"]]

    if buf:
        utt.append({"speaker": cur_spk, "start": float(cur_start), "end": float(words[-1]["end"]), "text": " ".join(buf).strip()})

    # Merge same-speaker consecutive
    merged: List[Dict[str, Any]] = []
    for u in utt:
        if merged and merged[-1]["speaker"] == u["speaker"]:
            merged[-1]["text"] = (merged[-1]["text"] + " " + u["text"]).strip()
            merged[-1]["end"] = u["end"]
        else:
            merged.append(u)
    return merged

# --------------- DEDUPE CORE ----------------
_WORD = re.compile(r"[^\wÀ-ÖØ-öø-ÿ'-]+", flags=re.UNICODE)

def _normalize_for_match(s: str) -> List[str]:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    # on ne retire pas les apostrophes; on nettoie ponctuation forte
    tokens = [t for t in _WORD.sub(" ", s).split() if t]
    return tokens

def _collapse_token_stutter(tokens: List[str]) -> List[str]:
    """Supprime répétitions immédiates (de, de, de → de ; vous vous → vous)."""
    out: List[str] = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out

def _trim_overlap(prev: str, cur: str) -> str:
    """
    Si le début de cur répète la fin de prev, coupe ce préfixe.
    Cherche le plus grand k (3..30 tokens) tel que suffix(prev,k)==prefix(cur,k).
    """
    a = _normalize_for_match(prev)
    b = _normalize_for_match(cur)
    max_k = min(30, len(a), len(b))
    cut = 0
    for k in range(max_k, 2, -1):
        if a[-k:] == b[:k]:
            cut = len(" ".join(cur.split()[:0]))  # placeholder, we'll reconstruct by tokens
            cur_tokens = cur.split()
            cur = " ".join(cur_tokens[k:])
            break
    return cur.strip()

def _dedupe_intra_sentence(text: str) -> str:
    """
    Nettoyage agressif dans une même phrase:
      - stutter tokens
      - n-grammes adjacents répétés (n=6→3)
      - fusion espaces/ponctuation
    """
    raw_tokens = text.split()
    toks = _collapse_token_stutter(raw_tokens)

    # Supprime n-grammes adjacents dupliqués
    for n in (6, 5, 4, 3):
        i = 0
        out = []
        while i < len(toks):
            out.append(toks[i])
            # compare bloc courant vs bloc suivant
            if i + 2*n <= len(toks) and toks[i:i+n] == toks[i+n:i+2*n]:
                i += n  # skip un bloc (garde un seul)
            i += 1
        toks = out

    s = " ".join(toks)
    # Compacte ponctuation et espaces
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([(\[{])\s+", r"\1", s)
    s = re.sub(r"\s+([)\]}])", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def _split_sentences(text: str) -> List[str]:
    # coupe sur ponctuation forte tout en conservant le signe
    parts = re.split(r"([.!?…])", text)
    out = []
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip()
        punct = parts[i+1] if i+1 < len(parts) else ""
        if chunk:
            out.append((chunk + punct).strip())
    if not out:
        return [text.strip()]
    return out

def clean_utterances(utts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Dédoublonne:
      1) intra-phrase (stutter, n-grammes)
      2) coupe les chevauchements entre locuteurs successifs
      3) supprime/compacte phrases quasi-identiques dans une fenêtre glissante
    Retourne (utterances_nettoyées, full_text).
    """
    cleaned: List[Dict[str, Any]] = []
    memory = deque(maxlen=80)  # mémoire de normalisés pour la dédup inter-locuteurs

    prev_text = ""
    for u in utts:
        t = _dedupe_intra_sentence(u["text"])
        if cleaned:
            t = _trim_overlap(cleaned[-1]["text"], t)

        # coupe en phrases, enlève quasi-doublons récents
        kept_sentences: List[str] = []
        for sent in _split_sentences(t):
            norm = tuple(_normalize_for_match(sent))
            if not norm:
                continue
            too_close = False
            # test exact ou quasi-identique avec mémoire
            if norm in memory:
                too_close = True
            else:
                # Similarité approximative par longueur d'intersection
                for mem in memory:
                    inter = len(set(norm) & set(mem))
                    if inter >= 0.9 * min(len(norm), len(mem)) and min(len(norm), len(mem)) >= 4:
                        too_close = True
                        break
            if not too_close:
                kept_sentences.append(sent)
                memory.append(norm)

        new_text = " ".join(kept_sentences).strip()
        if not new_text:
            # si tout a sauté, étend le segment précédent si même speaker
            if cleaned and cleaned[-1]["speaker"] == u["speaker"]:
                cleaned[-1]["end"] = max(cleaned[-1]["end"], u["end"])
            continue

        # merge si même speaker consécutif pour compacter
        if cleaned and cleaned[-1]["speaker"] == u["speaker"]:
            # si chevauchement, coupe début redondant
            new_text = _trim_overlap(cleaned[-1]["text"], new_text)
            if new_text:
                cleaned[-1]["text"] = (cleaned[-1]["text"] + " " + new_text).strip()
                cleaned[-1]["end"] = max(cleaned[-1]["end"], u["end"])
        else:
            cleaned.append({
                "speaker": u["speaker"],
                "start": u["start"],
                "end": u["end"],
                "text": new_text
            })

        prev_text = new_text

    full_text = " ".join(u["text"] for u in cleaned if u["text"]).strip()
    return cleaned, full_text

# ------------------- Main -------------------
def main():
    parser = argparse.ArgumentParser(description="YouTube/Local → VTT → Diarization → JSON (clean)")
    parser.add_argument("--input", required=True, help="URL YouTube ou chemin fichier audio local")
    parser.add_argument("--vtt", default=None, help="Chemin VTT local (optionnel; sinon yt-dlp)")
    parser.add_argument("--output", default="outputs/out.json", help="Chemin du JSON de sortie")
    parser.add_argument("--subs-lang", default="fr,fr-FR,fr-CA", help="Langues préférées VTT (ex: 'fr,fr-FR')")
    parser.add_argument("--allow-auto", action="store_true", help="Autoriser sous-titres auto si manuels absents")
    parser.add_argument("--num-speakers", type=int, default=None, help="Nombre de locuteurs (optionnel)")
    parser.add_argument("--start", default=None, help="Début (HH:MM:SS ou SS) pour découper l'audio (optionnel)")
    parser.add_argument("--duration", type=int, default=None, help="Durée en secondes (optionnel)")
    parser.add_argument("--healthcheck", action="store_true", help="Vérifie dépendances puis quitte")
    args = parser.parse_args()

    if args.healthcheck:
        step("Healthcheck…")
        ensure_ffmpeg(); ok("ffmpeg OK")
        try: import yt_dlp; ok("yt-dlp OK")
        except Exception: fail("yt-dlp manquant")
        try: import webvtt; ok("webvtt-py OK")
        except Exception: fail("webvtt-py manquant")
        try: import pyannote.audio; ok("pyannote.audio OK")
        except Exception: fail("pyannote.audio manquant")
        if os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN"): ok("HF token trouvé")
        else: info("HF token manquant (nécessaire pour la diarisation)")
        ok("Healthcheck terminé ✅")
        return

    ensure_ffmpeg()
    preferred_langs = [s.strip() for s in args.subs_lang.split(",") if s.strip()]

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        meta: Dict[str, Any] = {
            "created_at": now_iso(),
            "source": args.input,
            "params": {
                "subs_lang": preferred_langs,
                "allow_auto": bool(args.allow_auto),
                "num_speakers": args.num_speakers,
                "start": args.start,
                "duration": args.duration,
            },
        }

        # --- Input handling
        if is_url(args.input):
            step("Téléchargement audio YouTube…")
            media_path, yinfo = download_ytdlp_media(args.input, tmp)
            ok(f"Fichier téléchargé: {media_path.name}")
            meta["title"] = yinfo.get("title")
            meta["uploader"] = yinfo.get("uploader")
            meta["duration"] = yinfo.get("duration")
            meta["webpage_url"] = yinfo.get("webpage_url")
            # VTT
            vtt_path = Path(args.vtt) if args.vtt else download_ytdlp_subs(
                args.input, tmp, preferred_langs, args.allow_auto
            )
            step("Récupération des sous-titres .vtt (FR préférés)…")
            if not vtt_path or not vtt_path.exists():
                if args.allow_auto:
                    fail("Sous-titres FR introuvables même en auto.")
                else:
                    fail("Sous-titres FR introuvables (et auto désactivé). Relance avec --allow-auto si besoin.")
            ok(f"VTT: {vtt_path.name}")
        else:
            media_path = Path(args.input).expanduser().resolve()
            if not media_path.exists():
                fail(f"Fichier introuvable: {media_path}")
            meta["title"] = media_path.name
            vtt_path = Path(args.vtt) if args.vtt else None
            if not vtt_path or not vtt_path.exists():
                fail("Pour un fichier local, fournis --vtt <chemin.vtt> (transcript indispensable).")

        # --- Convert audio (extrait si demandé)
        start_sec = None
        if args.start:
            try:
                parts = [float(x) for x in args.start.split(":")]
                if len(parts) == 3:
                    start_sec = parts[0]*3600 + parts[1]*60 + parts[2]
                elif len(parts) == 2:
                    start_sec = parts[0]*60 + parts[1]
                else:
                    start_sec = float(args.start)
            except Exception:
                fail("Format --start invalide (utilise HH:MM:SS, MM:SS ou SS).")
        wav_path = tmp / "audio_16k_mono.wav"
        step("Conversion en WAV mono 16k…")
        convert_to_wav_mono16k(media_path, wav_path, start=start_sec, duration=args.duration)
        ok(f"Audio prêt: {wav_path}")

        # --- Parse VTT → words
        step("Parsing VTT → mots horodatés…")
        words = parse_vtt_words(vtt_path)
        if start_sec is not None and args.duration:
            end_sec = start_sec + args.duration
            words = [w for w in words if start_sec <= (w["start"] + w["end"])/2.0 <= end_sec]
        ok(f"{len(words)} tokens.")

        if not words:
            fail("Aucun mot issu du VTT dans la fenêtre demandée.")

        # --- Diarization
        spk_segments = diarize_pyannote(wav_path, num_speakers=args.num_speakers)

        # --- Align & Clean
        step("Alignement mots ↔ locuteurs…")
        utterances_raw = align_words_to_segments(words, spk_segments)
        ok(f"{len(utterances_raw)} blocs bruts.")
        step("Nettoyage / dédoublonnage…")
        utterances_clean, full_text = clean_utterances(utterances_raw)
        ok(f"{len(utterances_clean)} blocs après nettoyage.")

        # --- Export
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": meta,
            "text": full_text,
            "utterances": utterances_clean,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        ok(f"JSON écrit: {out.resolve()}")
        print("Terminé ✅")

if __name__ == "__main__":
    main()
