# --- Setup YTDLP-only for mon_projet_politique ---
$ErrorActionPreference = "Stop"

function Backup-IfExists($path) {
  if (Test-Path $path) {
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $bak = "$path.bak_$ts"
    Copy-Item -Force $path $bak
    Write-Host "↳ Backup: $path -> $bak"
  }
}

# 0) Sanity: run from project root
if (-not (Test-Path ".\requirements.txt") -and -not (Test-Path ".\app.py")) {
  Write-Error "Lance ce script depuis la racine du projet (là où se trouve app.py / requirements.txt)."
}

# 1) Backup existing files
Backup-IfExists ".\app.py"
Backup-IfExists ".\requirements.txt"
Backup-IfExists ".\README.md"

# 2) Write new requirements (YTDLP + VTT + Pyannote)
$requirements = @'
yt-dlp>=2024.5.27
webvtt-py>=0.5.1
pyannote.audio>=3.1.1
torch
numpy
soundfile
'@
Set-Content -NoNewline -Path ".\requirements.txt" -Value $requirements
Write-Host "✔ requirements.txt mis à jour (YTDLP-only)."

# 3) Write new app.py (YTDLP-only, no Whisper fallback)
$app = @'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YTDLP-only pipeline:
- Télécharge l'audio + sous-titres .vtt depuis YouTube (yt-dlp)
- Parse le .vtt en "mots" approximés (timestamps répartis)
- Diarization (pyannote)
- Aligne mots ↔ speakers
- Écrit un JSON final (utterances: speaker/start/end/text)

Exemples:
  python app.py --input "https://www.youtube.com/watch?v=VIDEO_ID" --output outputs/json/out.json
  python app.py --healthcheck

Notes:
  - ffmpeg doit être installé (ffmpeg -version)
  - Accepter les termes du modèle Hugging Face: pyannote/speaker-diarization-3.1
  - Définir HUGGINGFACE_TOKEN (ou HF_TOKEN) dans l'environnement
"""

import argparse
import os
import sys
import json
import shutil
import subprocess
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Any

def print_step(msg: str): print(f"[STEP] {msg}", flush=True)
def print_check(msg: str): print(f"[CHECK] {msg}", flush=True)
def fail(msg: str, code: int = 1):
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)

def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        fail("ffmpeg non trouvé dans le PATH. Installez-le (ex: choco install ffmpeg) puis réessayez.")

def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

# ---------- YT-DLP: audio + sous-titres ----------
def download_youtube_audio(url: str, out_dir: Path) -> Path:
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        fail("yt-dlp n'est pas installé. pip install yt-dlp")
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

def download_youtube_subs(url: str, out_dir: Path) -> Path | None:
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        fail("yt-dlp n'est pas installé. pip install yt-dlp")
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["fr","fr-CA","fr-FR","en","en-US","en-GB"],
        "subtitlesformat": "vtt",
        "noplaylist": True,
        "outtmpl": str(out_dir / "%(title).200s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)
    vtts = list(out_dir.glob("*.vtt"))
    return vtts[0] if vtts else None

# ---------- Conversion audio ----------
def convert_to_wav_mono16k(src: Path, dst: Path) -> Path:
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", "-vn", str(dst)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        fail(f"Echec conversion audio (ffmpeg): {proc.stderr[:400]}")
    return dst

# ---------- VTT parsing en "mots" approximés ----------
def to_seconds(ts: str) -> float:
    # "HH:MM:SS.mmm" ou "HH:MM:SS,mmm"
    ts = ts.replace(',', '.')
    h, m, s = ts.split(':')
    return int(h)*3600 + int(m)*60 + float(s)

def parse_vtt_to_words(vtt_path: Path) -> List[Dict[str, Any]]:
    import webvtt
    words: List[Dict[str, Any]] = []
    for cue in webvtt.read(str(vtt_path)):
        text = " ".join(cue.text.strip().split())
        if not text:
            continue
        start = to_seconds(cue.start)
        end = to_seconds(cue.end)
        tokens = text.split()
        dur = max(1e-3, end - start)
        step = dur / len(tokens)
        for i, tok in enumerate(tokens):
            ws = start + i*step
            we = start + (i+1)*step
            words.append({"start": ws, "end": we, "text": tok})
    return words

# ---------- Diarization ----------
def diarize_pyannote(wav_path: Path) -> List[Dict[str, Any]]:
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        fail("HUGGINGFACE_TOKEN (ou HF_TOKEN) non défini. Ajoutez votre token Hugging Face.")
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        fail("pyannote.audio n'est pas installé. pip install 'pyannote.audio>=3.1.1'")
    print_step("Diarization (pyannote.audio)...")
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    except Exception as e:
        fail("Impossible de charger le pipeline. Avez-vous cliqué 'Agree to terms' sur la page du modèle Hugging Face ?")
    diarization = pipeline(str(wav_path))
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})
    segments.sort(key=lambda x: x["start"])
    print_check(f"Diarization ok. {len(segments)} segments.")
    return segments

# ---------- Alignement mots ↔ speakers ----------
def align_words_to_diarization(words: List[Dict[str, Any]], spk_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not words:
        return []
    if not spk_segments:
        joined = " ".join(w["text"] for w in words).strip()
        return [{"speaker":"SPEAKER_00","start":words[0]["start"],"end":words[-1]["end"],"text":joined}]
    spk_segments = sorted(spk_segments, key=lambda s:(s["start"], s["end"]))

    def speaker_at(t: float) -> str:
        for seg in spk_segments:
            if seg["start"] <= t <= seg["end"]:
                return seg["speaker"]
        # nearest fallback
        nearest = min(spk_segments, key=lambda s: min(abs(s["start"]-t), abs(s["end"]-t)))
        return nearest["speaker"]

    utterances = []
    cur_speaker, cur_start, buf = None, None, []
    for w in words:
        mid = (w["start"] + w["end"]) / 2.0
        spk = speaker_at(mid)
        if cur_speaker is None:
            cur_speaker, cur_start, buf = spk, w["start"], [w["text"]]
        elif spk == cur_speaker:
            buf.append(w["text"])
        else:
            text = re.sub(r"\s+"," "," ".join(buf)).strip()
            if text:
                utterances.append({"speaker":cur_speaker,"start":float(cur_start),"end":float(w["start"]),"text":text})
            cur_speaker, cur_start, buf = spk, w["start"], [w["text"]]
    if buf:
        text = re.sub(r"\s+"," "," ".join(buf)).strip()
        utterances.append({"speaker":cur_speaker,"start":float(cur_start),"end":float(words[-1]["end"]),"text":text})

    # fusion consécutifs de même speaker
    merged = []
    for u in utterances:
        if merged and merged[-1]["speaker"] == u["speaker"]:
            merged[-1]["text"] = (merged[-1]["text"] + " " + u["text"]).strip()
            merged[-1]["end"] = u["end"]
        else:
            merged.append(u)
    return merged

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="YouTube → Subtitles(.vtt) → Diarization → JSON (YTDLP-only)")
    parser.add_argument("--input", type=str, required=True, help="URL YouTube")
    parser.add_argument("--output", type=str, default="outputs/json/result.json", help="Chemin du JSON de sortie")
    parser.add_argument("--healthcheck", action="store_true", help="Vérifie l'environnement")
    args = parser.parse_args()

    if args.healthcheck:
        print_step("Healthcheck...")
        # ffmpeg
        if shutil.which("ffmpeg"): print_check("ffmpeg OK")
        else: fail("ffmpeg manquant (installez-le et ajoutez-le au PATH)")
        # yt-dlp
        try:
            import yt_dlp  # noqa: F401
            print_check("yt-dlp OK")
        except Exception:
            fail("yt-dlp manquant (pip install yt-dlp)")
        # webvtt
        try:
            import webvtt  # noqa: F401
            print_check("webvtt-py OK")
        except Exception:
            fail("webvtt-py manquant (pip install webvtt-py)")
        # pyannote
        try:
            import pyannote.audio  # noqa: F401
            print_check("pyannote.audio OK")
        except Exception:
            fail("pyannote.audio manquant (pip install 'pyannote.audio>=3.1.1')")
        # token
        if os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN"):
            print_check("Hugging Face token trouvé")
        else:
            fail("Hugging Face token introuvable (HUGGINGFACE_TOKEN/HF_TOKEN)")
        print_check("Healthcheck terminé ✅")
        sys.exit(0)

    if not is_url(args.input):
        fail("--input doit être une URL YouTube (YTDLP-only).")

    ensure_ffmpeg()

    with tempfile.TemporaryDirectory() as tmpdir_:
        tmpdir = Path(tmpdir_)
        # 1) Audio
        print_step("Téléchargement audio YouTube...")
        media_path = download_youtube_audio(args.input, tmpdir)
        print_check(f"Fichier téléchargé: {media_path.name}")

        # 2) Convert WAV mono 16k
        wav_path = tmpdir / "audio_16k_mono.wav"
        print_step("Conversion en WAV mono 16k...")
        convert_to_wav_mono16k(media_path, wav_path)
        print_check(f"Audio prêt: {wav_path}")

        # 3) Sous-titres .vtt obligatoires
        print_step("Récupération des sous-titres .vtt (yt-dlp)...")
        vtt = download_youtube_subs(args.input, tmpdir)
        if vtt is None:
            fail("Aucun sous-titre .vtt disponible pour cette vidéo (YTDLP-only, pas de Whisper).")
        print_check(f"Sous-titres trouvés: {vtt.name}")

        # 4) Parse VTT → mots approximés
        words = parse_vtt_to_words(vtt)
        print_check(f"{len(words)} 'mots' horodatés générés depuis le .vtt.")

        # 5) Diarization
        spk_segments = diarize_pyannote(wav_path)

        # 6) Alignement
        print_step("Alignement mots ↔ locuteurs...")
        utterances = align_words_to_diarization(words, spk_segments)
        print_check(f"{len(utterances)} blocs speaker+texte.")

        # 7) JSON
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "source": args.input,
            "utterances": utterances,
            "summary": {
                "num_words": len(words),
                "num_speaker_segments": len(spk_segments),
                "num_utterances": len(utterances)
            }
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print_check(f"JSON écrit: {out_path.resolve()}")
        print("Terminé ✅")

if __name__ == "__main__":
    main()
'@
Set-Content -NoNewline -Path ".\app.py" -Value $app
Write-Host "✔ app.py (YTDLP-only) écrit."

# 4) README addition
$readme = @'
# Mode YTDLP-only (sous-titres YouTube)

## 1) Activer le venv (PowerShell)
.\venv\Scripts\Activate.ps1

## 2) Installer les dépendances
pip install -r requirements.txt

## 3) Vérifier l'environnement
python app.py --healthcheck

## 4) Lancer (URL YouTube avec sous-titres)
python app.py --input "https://www.youtube.com/watch?v=VIDEO_ID" --output outputs/json/out.json

## Résultat
Le JSON est écrit dans `outputs/json/` avec des blocs `{speaker, start, end, text}`.
'@
if (Test-Path ".\README.md") {
  Add-Content -Path ".\README.md" -Value "`n`n$readme"
  Write-Host "✔ README.md complété"
} else {
  Set-Content -NoNewline -Path ".\README.md" -Value $readme
  Write-Host "✔ README.md créé"
}

Write-Host "`n🎯 Setup YTDLP-only terminé. Exécute maintenant:"
Write-Host "   .\venv\Scripts\Activate.ps1"
Write-Host "   pip install -r requirements.txt"
Write-Host "   python app.py --healthcheck"
Write-Host "   python app.py --input ""https://www.youtube.com/watch?v=VIDEO_ID"" --output outputs/json/out.json"
