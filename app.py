# /content/app.py
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# ----------------------------- petites utilitaires ----------------------------

def sh(cmd: list[str], allow_fail: bool = False) -> str:
    """Exécute une commande shell en capturant stdout/stderr."""
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode != 0 and not allow_fail:
        raise RuntimeError(
            f"[CMD ERROR] {' '.join(cmd)}\nSTDERR:\n{p.stderr}\nSTDOUT:\n{p.stdout}"
        )
    return p.stdout

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def seconds_from_timestamp(ts: str) -> float:
    """'HH:MM:SS.mmm' -> secondes flottantes."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def audio_duration_seconds(wav_path: str) -> float:
    out = sh(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            wav_path,
        ]
    ).strip()
    try:
        return float(out)
    except Exception:
        return 0.0

# -------------------------- téléchargement YouTube ----------------------------

def download_youtube(input_url: str, subs_lang: str, allow_auto: bool) -> Tuple[str, Optional[str], Path]:
    """
    - Télécharge l'audio en WAV.
    - Télécharge les sous-titres en .vtt si dispos (FR prioritaire; auto si allow_auto=True).
    Retourne (wav_16k_mono_path, vtt_path_or_None, workdir)
    """
    workdir = Path(tempfile.mkdtemp(prefix="yt_"))
    base = workdir / "media"

    # 1) AUDIO → WAV (qualité max)
    cmd_audio = [
        "yt-dlp",
        "-q",
        "-x",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "-o",
        f"{base}.%(ext)s",
        input_url,
    ]
    print("[STEP] Téléchargement audio YouTube…")
    sh(cmd_audio)

    wav = next(workdir.glob("media.wav"), None)
    if wav is None:
        raise RuntimeError("Audio .wav introuvable après yt-dlp.")

    # 2) SOUS-TITRES → .vtt
    sub_args = [
        "--write-subs",
        "--sub-langs",
        subs_lang,
        "--sub-format",
        "vtt",
        "--skip-download",
        "-o",
        f"{base}.%(ext)s",
    ]
    if allow_auto:
        # IMPORTANT : option séparée, pas une valeur du param précédent
        sub_args.insert(0, "--write-auto-subs")

    print("[STEP] Récupération des sous-titres .vtt (FR préférés)…")
    sh(["yt-dlp", "-q", *sub_args, input_url], allow_fail=True)
    vtt = next(workdir.glob("media*.vtt"), None)
    if vtt is None:
        print("[WARN] Sous-titres .vtt introuvables.")

    # 3) Conversion en 16k mono
    wav_16k = workdir / "audio_16k_mono.wav"
    print("[STEP] Conversion en WAV mono 16k…")
    sh(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(wav),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_16k),
        ]
    )
    print(f"[CHECK] Audio prêt: {wav_16k}")
    return str(wav_16k), (str(vtt) if vtt else None), workdir

# ---------------------------- parsing des sous-titres --------------------------

@dataclass
class Caption:
    start: float
    end: float
    text: str

def parse_vtt(vtt_path: str) -> List[Caption]:
    try:
        import webvtt
    except Exception as e:
        raise RuntimeError(
            "Le paquet 'webvtt-py' est requis. Installe :\n  pip install webvtt-py"
        ) from e

    print("[STEP] Parsing VTT → segments horodatés…")
    caps: List[Caption] = []
    for c in webvtt.read(vtt_path):
        start = seconds_from_timestamp(c.start)
        end = seconds_from_timestamp(c.end)
        text = " ".join(str(c.text).split())
        if text:
            caps.append(Caption(start, end, text))
    return caps

# ---------------------------- diarisation pyannote -----------------------------

@dataclass
class SpkSeg:
    start: float
    end: float
    speaker: str

def diarize_pyannote(wav_path: str, num_speakers: Optional[int]) -> List[SpkSeg]:
    try:
        import torch
        from pyannote.audio import Pipeline
    except Exception as e:
        raise RuntimeError(
            "pyannote.audio est manquant ou incompatible. Installe :\n  pip install 'pyannote.audio>=3.1.1'"
        ) from e

    # Token HF (doit avoir accepté les conditions du modèle)
    hf_token = os.environ.get("HF_TOKEN", None)

    print("[STEP] Diarization (pyannote.audio)…")
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    try:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
        ).to(dev)
    except Exception as e:
        raise RuntimeError(
            "Échec chargement pipeline pyannote. Vérifie HF_TOKEN et l'accès au modèle.\n"
            "Accepte les conditions ici : https://hf.co/pyannote/speaker-diarization-3.1"
        ) from e

    diarization = pipe(
        {"audio": wav_path},
        num_speakers=num_speakers,      # force 2 si fourni
        min_speech_duration=0.25,
        min_silence_duration=0.25,
        collar=0.05,
    )

    segs: List[SpkSeg] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segs.append(SpkSeg(float(turn.start), float(turn.end), str(speaker)))
    segs.sort(key=lambda x: x.start)

    if not segs:
        # fallback: un seul locuteur plein cadre
        dur = audio_duration_seconds(wav_path)
        segs = [SpkSeg(0.0, dur, "SPEAKER_00")]
    return segs

# ----------------------- alignement captions ↔ speakers ------------------------

@dataclass
class Utt:
    speaker: str
    start: float
    end: float
    text: str

def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def build_utterances(spk: List[SpkSeg], caps: List[Caption]) -> List[Utt]:
    """
    Pour chaque segment locuteur, concatène les captions chevauchantes.
    """
    utts: List[Utt] = []
    j = 0
    n = len(caps)
    for s in spk:
        buf = []
        # avance j jusqu'à la première cap susceptible de chevaucher
        while j < n and caps[j].end <= s.start:
            j += 1
        k = j
        while k < n and caps[k].start < s.end:
            if overlap(s.start, s.end, caps[k].start, caps[k].end) > 0:
                buf.append(caps[k].text)
            k += 1
        text = " ".join(buf).strip()
        if text:
            utts.append(Utt(s.speaker, s.start, s.end, text))
    return utts

# ------------------------- filtre anti-répétitions -----------------------------

_WORD = re.compile(r"\w+", re.UNICODE)

def _normalize_text(t: str) -> str:
    t = t.replace("’", "'")
    t = re.sub(r"\s+", " ", t.lower()).strip()
    return t

def remove_near_duplicates(
    utterances: List[Utt], ngram_max: int = 6, lookback_seconds: float = 6.0
) -> List[Utt]:
    """
    Supprime les répétitions proches inter- & intra-locuteur par n-grammes.
    Conserve la première occurrence dans une fenêtre glissante temporelle.
    """
    kept: List[Utt] = []
    memory: deque[Tuple[str, float]] = deque()

    def seen(ng: str, t: float) -> bool:
        while memory and t - memory[0][1] > lookback_seconds:
            memory.popleft()
        return any(ng == k for k, _ in memory)

    for u in utterances:
        txt = _normalize_text(u.text)
        words = [m.group(0) for m in _WORD.finditer(txt)]
        t_end = float(u.end)

        drop = False
        for n in range(min(ngram_max, len(words)), 1, -1):  # du long vers le court
            for i in range(len(words) - n + 1):
                ng = " ".join(words[i : i + n])
                if len(ng) < 6:
                    continue
                if seen(ng, t_end):
                    drop = True
                    break
            if drop:
                break

        if not drop:
            kept.append(u)
            # mémoriser quelques n-grammes
            for n in range(1, min(ngram_max, len(words)) + 1):
                for i in range(len(words) - n + 1):
                    ng = " ".join(words[i : i + n])
                    if len(ng) >= 6:
                        memory.append((ng, t_end))

    return kept

# ------------------------------- main pipeline --------------------------------

def run(
    input_url: str,
    output_path: str,
    subs_lang: str,
    allow_auto: bool,
    num_speakers: Optional[int],
    keep_metadata: bool,
):
    print("[CHECK] URL entrée:", input_url)
    wav_path, vtt_path, workdir = download_youtube(input_url, subs_lang, allow_auto)

    caps: List[Caption] = []
    if vtt_path:
        caps = parse_vtt(vtt_path)
    if not caps:
        print("[WARN] Aucune caption trouvée : le texte risque d’être vide.")

    spk = diarize_pyannote(wav_path, num_speakers=num_speakers)

    # Alignement + filtre de redites
    raw_utts = build_utterances(spk, caps) if caps else []
    print(f"[CHECK] {len(raw_utts)} segments alignés avant filtrage.")
    utts = remove_near_duplicates(raw_utts, ngram_max=6, lookback_seconds=6.0)
    print(f"[CHECK] {len(utts)} segments après filtrage des répétitions.")

    # JSON final
    ensure_parent(Path(output_path))
    meta = {
        "source": "YouTube",
        "video_url": input_url,
        "subs_lang": subs_lang,
        "allow_auto_subs": allow_auto,
        "diarization_model": "pyannote/speaker-diarization-3.1",
        "num_speakers_request": num_speakers,
        "device": "cuda" if shutil.which("nvidia-smi") else "cpu",
    }

    out = {
        "meta": meta if keep_metadata else {},
        "text": " ".join(u.text for u in utts).strip(),
        "utterances": [
            {
                "speaker": u.speaker,
                "start": round(float(u.start), 3),
                "end": round(float(u.end), 3),
                "text": u.text,
            }
            for u in utts
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] JSON écrit → {output_path}")
    # Nettoyage temp
    try:
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="URL YouTube ou chemin audio")
    ap.add_argument("--output", required=True, help="Chemin du JSON de sortie")
    ap.add_argument("--subs-lang", default="fr,fr-FR,fr-CA", help="Langues VTT (priorité FR)")
    ap.add_argument("--allow-auto", action="store_true", help="Autoriser les sous-titres auto")
    ap.add_argument("--num-speakers", type=int, default=None, help="Nombre de locuteurs attendus")
    ap.add_argument("--keep-metadata", action="store_true", help="Conserver l'en-tête meta")
    ap.add_argument("--healthcheck", action="store_true", help="Vérifie que le script se lance")
    args = ap.parse_args()

    if args.healthcheck:
        print("ok")
        return

    run(
        input_url=args.input,
        output_path=args.output,
        subs_lang=args.subs_lang,
        allow_auto=args.allow_auto,
        num_speakers=args.num_speakers,
        keep_metadata=args.keep_metadata,
    )


if __name__ == "__main__":
    main()
