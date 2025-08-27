#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
app.py — YouTube → VTT FR (YTDLP-only) → Diarization (pyannote) → JSON (+ SRT / CSV / MD)

Option B (sans Whisper) pour supprimer les "méga redites" :
1) Anti-reprises VTT (containment + similarité), garde la version la plus complète.
2) "Delta-cut" logique post-diarization (via préfixes communs) incluse dans la dédup.
3) Mémoire n-gram courte : rejette les blocs quasi-identiques dans une fenêtre de temps.

Dépendances: yt-dlp, webvtt-py, pyannote.audio, torch, ffmpeg
Optionnel (ponctuation): deepmultilingualpunctuation
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

# -------------------- Logs --------------------

def print_step(msg: str) -> None:
    print(f"[STEP] {msg}", flush=True)

def print_check(msg: str) -> None:
    print(f"[CHECK] {msg}", flush=True)

def fail(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)

# --------------- Sanity/Helpers ---------------

def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        fail("ffmpeg non trouvé dans le PATH.")

def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

def to_seconds(ts: str) -> float:
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def hhmmss(ts: float) -> str:
    ms = int(round((ts - int(ts)) * 1000))
    s = int(ts) % 60
    m = (int(ts) // 60) % 60
    h = int(ts) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def parse_hhmmss(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s = s.strip()
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + float(sec)
    if len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + float(sec)
    return float(s)

# --------------- Texte utils ------------------

FILLERS_FR = {
    "euh","bah","ben","hein","du coup","voilà","en fait","je veux dire",
    "comment dire","j'ai envie de dire","entre guillemets"
}

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _norm_text(s: str) -> str:
    s = s.lower()
    s = _strip_accents(s)
    for f in sorted(FILLERS_FR, key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(f)}\b", " ", s)
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokenize(s: str) -> List[str]:
    return _norm_text(s).split()

def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm_text(a), _norm_text(b)).ratio()

def _containment_ratio(a: str, b: str) -> float:
    """
    Containment 'a dans b' et 'b dans a' via ensembles de tokens.
    Retourne max(contain(a|b), contain(b|a)).
    """
    A, B = set(_tokenize(a)), set(_tokenize(b))
    if not A or not B:
        return 0.0
    ca = len([w for w in A if w in B]) / max(1, len(A))
    cb = len([w for w in B if w in A]) / max(1, len(B))
    return max(ca, cb)

def compress_repeats_inside_text(text: str, max_ngram: int = 6) -> str:
    """Supprime répétitions adjacentes (A B C A B C …) & compacte espaces."""
    t = text
    for n in range(max_ngram, 1, -1):
        pat = re.compile(rf"(\b(?:\w+\s+){{{n-1}}}\w+\b)(?:\s+\1)+", re.IGNORECASE | re.UNICODE)
        while True:
            new = pat.sub(r"\1", t)
            if new == t:
                break
            t = new
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _longest_common_prefix_tokens(a: str, b: str) -> int:
    A, B = _tokenize(a), _tokenize(b)
    i, n = 0, min(len(A), len(B))
    while i < n and A[i] == B[i]:
        i += 1
    return i

# ---------------- yt-dlp I/O ------------------

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

def download_youtube_subs(url: str, out_dir: Path) -> Optional[Path]:
    from yt_dlp import YoutubeDL, utils
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["fr", "fr-FR", "fr-CA"],  # FR only → limite 429
        "subtitlesformat": "vtt",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "extractor_retries": 3,
        "sleep_interval_requests": 1,
        "outtmpl": str(out_dir / "%(title).200s.%(ext)s"),
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
    except utils.DownloadError:
        return None
    vtts = sorted(out_dir.glob("*.vtt"))
    return vtts[0] if vtts else None

# -------------- Audio (ffmpeg) ----------------

def convert_to_wav_mono16k(src: Path, dst: Path,
                           start: Optional[float] = None,
                           duration: Optional[int] = None) -> Path:
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

# -------------- VTT parsing -------------------

def parse_vtt_captions(vtt_path: Path,
                       dedup_vtt: bool = True,
                       near_window: float = 5.0,
                       near_sim: float = 0.92,
                       contain_thr: float = 0.75) -> List[Dict[str, Any]]:
    """
    Anti-reprises VTT ('rolling captions'):
    - dans une fenêtre near_window (s), si B ~ A (similarité) OU B contient fortement A,
      on ne garde qu'une version (la plus complète).
    """
    import webvtt
    if not dedup_vtt:
        # mode simple : sans dédup
        caps = []
        for cue in webvtt.read(str(vtt_path)):
            text = " ".join(cue.text.strip().split())
            if not text:
                continue
            start, end = to_seconds(cue.start), to_seconds(cue.end)
            caps.append({"start": start, "end": end, "text": compress_repeats_inside_text(text)})
        return caps

    buffer = deque()     # captions récentes dans la fenêtre
    kept: List[Dict[str, Any]] = []

    for cue in webvtt.read(str(vtt_path)):
        raw = " ".join(cue.text.strip().split())
        if not raw:
            continue
        start, end = to_seconds(cue.start), to_seconds(cue.end)
        text = compress_repeats_inside_text(raw)

        # purge du buffer (hors fenêtre)
        while buffer and start - buffer[0]["end"] > near_window:
            kept.append(buffer.popleft())

        cur = {"start": start, "end": end, "text": text}
        drop_cur = False

        # compare avec les captions dans la fenêtre
        for i in range(len(buffer)-1, -1, -1):
            prev = buffer[i]
            sim = _similar(prev["text"], cur["text"])
            contain = _containment_ratio(prev["text"], cur["text"])

            if sim >= near_sim or contain >= contain_thr:
                # garder la plus "complète" (plus de tokens)
                if len(_tokenize(cur["text"])) >= len(_tokenize(prev["text"])):
                    buffer[i] = cur
                    drop_cur = True  # IMPORTANT: évite d'ajouter cur une 2e fois
                else:
                    drop_cur = True
                break

        if not drop_cur:
            buffer.append(cur)

    kept.extend(list(buffer))
    kept.sort(key=lambda x: x["start"])
    return kept

def parse_vtt_to_words(vtt_path: Path) -> List[Dict[str, Any]]:
    """Découpe chaque caption en tokens « approximativement horodatés »."""
    import webvtt
    words: List[Dict[str, Any]] = []
    for cue in webvtt.read(str(vtt_path)):
        text = " ".join(cue.text.strip().split())
        if not text:
            continue
        start, end = to_seconds(cue.start), to_seconds(cue.end)
        toks = text.split()
        dur = max(1e-3, end - start)
        step = dur / len(toks)
        for i, tok in enumerate(toks):
            ws = start + i * step
            we = start + (i + 1) * step
            words.append({"start": ws, "end": we, "text": tok})
    return words

# --------------- Diarization -----------------

def diarize_pyannote(wav_path: Path, num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        fail("HUGGINGFACE_TOKEN/HF_TOKEN manquant (crée un token Hugging Face et exporte la variable).")
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        fail("pyannote.audio manquant. Installe-le (pip install 'pyannote.audio>=3.1.1').")

    print_step("Diarization (pyannote.audio)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        ).to(device)
        print_check(f"pyannote device: {device.type}")
    except Exception as e:
        import traceback
        print("TRACEBACK:\n", traceback.format_exc())
        fail(f"Echec chargement pipeline: {e}")

    infer = {"audio": str(wav_path)}
    diar = pipeline(infer, num_speakers=num_speakers) if num_speakers is not None else pipeline(infer)

    segs: List[Dict[str, Any]] = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
        for turn, _, speaker in diar.itertracks(yield_label=True)
    ]
    segs.sort(key=lambda s: (s["start"], s["end"]))
    print_check(f"Diarization ok. {len(segs)} segments.")
    return segs

# ------ Alignement mots → locuteur -----------

def align_words_to_diarization(words: List[Dict[str, Any]],
                               segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not words:
        return []
    if not segs:
        txt = " ".join(w["text"] for w in words).strip()
        return [{"speaker": "SPEAKER_00", "start": words[0]["start"], "end": words[-1]["end"], "text": txt}]

    segs = sorted(segs, key=lambda s: (s["start"], s["end"]))

    def speaker_at(t: float) -> str:
        for s in segs:
            if s["start"] <= t <= s["end"]:
                return s["speaker"]
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

    # fusion consécutifs (même locuteur)
    merged: List[Dict[str, Any]] = []
    for u in utt:
        if merged and merged[-1]["speaker"] == u["speaker"]:
            merged[-1]["text"] = (merged[-1]["text"] + " " + u["text"]).strip()
            merged[-1]["end"] = u["end"]
        else:
            merged.append(u)
    return merged

# ------------- Post-traitements ----------------

def apply_punctuation_if_needed(utterances: List[Dict[str, Any]], enabled: bool) -> List[Dict[str, Any]]:
    if not enabled or not utterances:
        return utterances
    try:
        from deepmultilingualpunctuation import PunctuationModel
        model = PunctuationModel()
    except Exception as e:
        print_check(f"Ponctuation ignorée (modèle non dispo) : {e}")
        return utterances
    for u in utterances:
        u["text"] = model.restore_punctuation(u["text"])
    return utterances

def merge_micro_segments(utterances: List[Dict[str, Any]],
                         min_chars: int = 60,
                         merge_gap_sec: float = 0.8) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for u in utterances:
        u["text"] = compress_repeats_inside_text(u["text"])
        if out:
            prev = out[-1]
            if (u["speaker"] == prev["speaker"] and
                (u["start"] - prev["end"]) <= merge_gap_sec and
                (len(prev["text"]) < min_chars or len(u["text"]) < min_chars)):
                prev["text"] = compress_repeats_inside_text((prev["text"] + " " + u["text"]).strip())
                prev["end"] = u["end"]
                continue
        out.append(u)
    return out

def dedup_utterances(utts: List[Dict[str, Any]],
                     window_sec: float = 12.0,
                     sim: float = 0.88,
                     prefer_longer: bool = True,
                     ngram: int = 5,
                     novelty_ratio: float = 0.20) -> List[Dict[str, Any]]:
    """
    Agressif cross-speaker:
      - similarité + containment contre les blocs récents (fenêtre 'window_sec')
      - delta-cut (préfixe commun retiré) si répétition
      - mémoire n-gram : rejette si quasi rien de nouveau
    """
    out: List[Dict[str, Any]] = []
    seen = {}  # ngram -> last_end_time

    def ngrams(tokens, n):
        return set(tuple(tokens[i:i+n]) for i in range(0, max(0, len(tokens)-n+1)))

    for u in utts:
        utext = compress_repeats_inside_text(u["text"])
        u["text"] = utext
        keep = True

        # 1) dédup vs backlog : similarité ou containment
        for v in reversed(out):
            if u["start"] - v["end"] > window_sec:
                break
            s = _similar(utext, v["text"])
            contain = _containment_ratio(utext, v["text"])
            if s >= sim or contain >= 0.75:
                # delta-cut : enlève le plus long préfixe commun (en tokens)
                lcp = _longest_common_prefix_tokens(v["text"], utext)
                utoks = _tokenize(utext)
                new_tail = " ".join(utoks[lcp:])
                new_tail = compress_repeats_inside_text(new_tail)
                if len(new_tail) >= 8:  # assez de "nouveau"
                    u["text"] = new_tail
                else:
                    keep = False
                break

        if not keep:
            continue

        # 2) filtre n-gram de nouveauté
        toks = _tokenize(u["text"])
        N = ngrams(toks, ngram) if len(toks) >= ngram else set()
        if N:
            known = sum(1 for g in N if g in seen and (u["start"] - seen[g]) <= window_sec)
            if known / max(1, len(N)) > (1.0 - novelty_ratio):
                keep = False
            else:
                for g in N:
                    seen[g] = u["end"]

        if keep:
            out.append(u)

    # fusion finale même speaker + petite pause
    merged: List[Dict[str, Any]] = []
    for u in out:
        if merged and merged[-1]["speaker"] == u["speaker"] and (u["start"] - merged[-1]["end"]) <= 0.8:
            merged[-1]["text"] = compress_repeats_inside_text((merged[-1]["text"] + " " + u["text"]).strip())
            merged[-1]["end"] = u["end"]
        else:
            merged.append(u)
    return merged

# ------------------ Exports -------------------

def export_json(path: Path,
                src_url: str,
                params: Dict[str, Any],
                captions: List[Dict[str, Any]],
                full_text: str,
                utterances: List[Dict[str, Any]],
                diag_segments_count: int) -> None:
    result = {
        "source": src_url,
        "params": params,
        "transcript": {
            "captions": captions,
            "text": full_text
        },
        "utterances": utterances,
        "summary": {
            "num_captions": len(captions),
            "num_words_clip": sum(len(u["text"].split()) for u in utterances),
            "num_speaker_segments": diag_segments_count,
            "num_utterances": len(utterances),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def export_srt(path: Path, utterances: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, u in enumerate(utterances, 1):
            f.write(f"{i}\n{hhmmss(u['start'])} --> {hhmmss(u['end'])}\n")
            f.write(f"[{u['speaker']}] {u['text']}\n\n")

def export_csv(path: Path, utterances: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_sec", "end_sec", "speaker", "text"])
        for u in utterances:
            w.writerow([f"{u['start']:.3f}", f"{u['end']:.3f}", u["speaker"], u["text"]])

def export_md(path: Path, utterances: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        cur = None
        buf: List[str] = []
        def flush():
            nonlocal buf
            if buf:
                f.write(" ".join(buf).strip() + "\n\n")
                buf = []
        for u in utterances:
            if u["speaker"] != cur:
                flush()
                cur = u["speaker"]
                f.write(f"### {cur}\n\n")
            buf.append(u["text"])
        flush()

# --------------------- Main -------------------

def main() -> None:
    p = argparse.ArgumentParser(description="YouTube → VTT FR (YTDLP-only) + Diarization (pyannote) → JSON")
    p.add_argument("--input", type=str, required=False, help="URL YouTube")
    p.add_argument("--output", type=str, default="result.json", help="Chemin du JSON de sortie")
    p.add_argument("--healthcheck", action="store_true", help="Vérifie l'environnement et quitte")

    # Diarization
    p.add_argument("--start", type=str, default=None, help="Début HH:MM:SS ou MM:SS de l'extrait à diariser")
    p.add_argument("--duration", type=int, default=None, help="Durée en secondes de l'extrait à diariser")
    p.add_argument("--num-speakers", type=int, default=None, help="Forcer N locuteurs (optionnel)")

    # Lisibilité / fusion
    p.add_argument("--punctuate", action="store_true", help="Ponctuation & majuscules (modèle multilingue)")
    p.add_argument("--min-utt-chars", type=int, default=60, help="Fusion si segment trop court")
    p.add_argument("--merge-gap-sec", type=float, default=0.8, help="Fusion si pause ≤ N secondes")

    # Dédup VTT (pré)
    p.add_argument("--no-dedup-vtt", action="store_true", help="Désactive l'anti-doublons VTT")
    p.add_argument("--vtt-window", type=float, default=5.0, help="Fenêtre VTT (sec) anti-reprises")
    p.add_argument("--vtt-sim", type=float, default=0.92, help="Seuil similarité VTT")
    p.add_argument("--vtt-contain", type=float, default=0.75, help="Seuil containment VTT")

    # Dédup utterances (post)
    p.add_argument("--no-dedup", action="store_true", help="Désactive la dédup post-diarization")
    p.add_argument("--dedup-window", type=float, default=12.0, help="Fenêtre (sec) post-diarization")
    p.add_argument("--dedup-sim", type=float, default=0.88, help="Seuil similarité post-diarization")
    p.add_argument("--dedup-ngram", type=int, default=5, help="Taille n-gram pour nouveauté")
    p.add_argument("--dedup-novelty", type=float, default=0.20, help="Ratio de nouveauté minimal (0-1)")

    # Exports
    p.add_argument("--export-srt", action="store_true")
    p.add_argument("--export-csv", action="store_true")
    p.add_argument("--export-md", action="store_true")

    args = p.parse_args()

    # -------- Healthcheck --------
    if args.healthcheck:
        print_step("Healthcheck...")
        if shutil.which("ffmpeg"): print_check("ffmpeg OK")
        else: fail("ffmpeg manquant")
        try: import yt_dlp; print_check("yt-dlp OK")
        except Exception: fail("yt-dlp manquant")
        try: import webvtt; print_check("webvtt-py OK")
        except Exception: fail("webvtt-py manquant")
        try: import pyannote.audio; print_check("pyannote.audio OK")
        except Exception: fail("pyannote.audio manquant")
        if os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN"): print_check("HF token trouvé")
        else: fail("HF token manquant")
        print_check("Healthcheck terminé ✅")
        sys.exit(0)

    if not args.input:
        fail("--input requis (sauf --healthcheck).")
    if not is_url(args.input):
        fail("--input doit être une URL YouTube (YTDLP-only).")

    ensure_ffmpeg()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # 1) Audio
        print_step("Téléchargement audio YouTube...")
        media_path = download_youtube_audio(args.input, tmp)
        print_check(f"Fichier téléchargé: {media_path.name}")

        # 2) VTT FR
        print_step("Récupération des sous-titres .vtt (FR)...")
        vtt = download_youtube_subs(args.input, tmp)
        if vtt is None:
            fail("Pas de sous-titres FR disponibles (YTDLP-only).")
        print_check(f"Sous-titres trouvés: {vtt.name}")

        captions = parse_vtt_captions(
            vtt,
            dedup_vtt=not args.no_dedup_vtt,
            near_window=args.vtt_window,
            near_sim=args.vtt_sim,
            contain_thr=args.vtt_contain,
        )
        words_all = parse_vtt_to_words(vtt)
        full_text = " ".join(c["text"] for c in captions)

        # 3) Audio 16k mono (extrait si demandé)
        start_sec = parse_hhmmss(args.start) if args.start else None
        wav_path = tmp / "audio_16k_mono.wav"
        print_step("Conversion en WAV mono 16k...")
        convert_to_wav_mono16k(media_path, wav_path, start=start_sec, duration=args.duration)
        print_check(f"Audio prêt: {wav_path}")

        # 4) Restreindre les mots à l'extrait (si fourni)
        if start_sec is not None and args.duration is not None:
            end_sec = start_sec + args.duration
            words_clip = [w for w in words_all if start_sec <= (w["start"] + w["end"]) / 2.0 <= end_sec]
        else:
            words_clip = words_all
        print_check(f"{len(words_clip)} 'mots' à aligner.")

        # 5) Diarization
        spk_segments = diarize_pyannote(wav_path, num_speakers=args.num_speakers)

        # 6) Alignement + post-traitements
        print_step("Alignement mots ↔ locuteurs...")
        utterances = align_words_to_diarization(words_clip, spk_segments)
        utterances = apply_punctuation_if_needed(utterances, args.punctuate)
        utterances = merge_micro_segments(utterances, min_chars=args.min_utt_chars, merge_gap_sec=args.merge_gap_sec)
        if not args.no_dedup:
            utterances = dedup_utterances(
                utterances,
                window_sec=args.dedup_window,
                sim=args.dedup_sim,
                prefer_longer=True,
                ngram=args.dedup_ngram,
                novelty_ratio=args.dedup_novelty,
            )
        print_check(f"{len(utterances)} blocs speaker+texte.")

        # 7) Exports
        out_json = Path(args.output)
        export_json(
            out_json,
            src_url=args.input,
            params={
                "start": args.start,
                "duration": args.duration,
                "num_speakers": args.num_speakers,
                "punctuate": args.punctuate,
                "min_utt_chars": args.min_utt_chars,
                "merge_gap_sec": args.merge_gap_sec,
                "dedup_vtt": not args.no_dedup_vtt,
                "vtt_window": args.vtt_window,
                "vtt_sim": args.vtt_sim,
                "vtt_contain": args.vtt_contain,
                "dedup": not args.no_dedup,
                "dedup_window": args.dedup_window,
                "dedup_sim": args.dedup_sim,
                "dedup_ngram": args.dedup_ngram,
                "dedup_novelty": args.dedup_novelty,
            },
            captions=captions,
            full_text=full_text,
            utterances=utterances,
            diag_segments_count=len(spk_segments),
        )
        print_check(f"JSON écrit: {out_json.resolve()}")

        base = out_json.with_suffix("")
        if args.export_srt:
            export_srt(base.with_suffix(".srt"), utterances)
            print_check(f"SRT écrit: {base.with_suffix('.srt')}")
        if args.export_csv:
            export_csv(base.with_suffix(".csv"), utterances)
            print_check(f"CSV écrit: {base.with_suffix('.csv')}")
        if args.export_md:
            export_md(base.with_suffix(".md"), utterances)
            print_check(f"MD écrit: {base.with_suffix('.md')}")

        print("Terminé ✅")

if __name__ == "__main__":
    main()
