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

# ----------------------------- utilitaires shell ------------------------------

def sh(cmd: list[str], allow_fail: bool = False) -> str:
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode != 0 and not allow_fail:
        raise RuntimeError(
            f"[CMD ERROR] {' '.join(cmd)}\nSTDERR:\n{p.stderr}\nSTDOUT:\n{p.stdout}"
        )
    return p.stdout

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def seconds_from_timestamp(ts: str) -> float:
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

def download_youtube(input_url: str, subs_lang: str, allow_auto: bool):
    # Télécharge l'audio + récupère les VTT via yt-dlp avec cookies + convertit en mono 16 kHz.
    import uuid, os
    from pathlib import Path

    def _pick_cookies():
        env = os.environ.get('COOKIES', '').strip()
        candidates = []
        if env:
            candidates.append(Path(env))
        candidates.append(Path('/content/drive/MyDrive/mon_projet_politique/cookies.txt'))
        for c in candidates:
            if c and c.exists():
                return c
        return None

    print('[STEP] Téléchargement audio YouTube…')
    workdir = Path(f"/tmp/yt_{uuid.uuid4().hex[:8]}")
    workdir.mkdir(parents=True, exist_ok=True)

    cookies_file = _pick_cookies()
    if cookies_file is None:
        print('[WARN] Aucun fichier cookies trouvé. Certains sous-titres peuvent manquer.')
        cookie_args = []
    else:
        print(f'[CHECK] Cookies: {cookies_file}')
        cookie_args = ['--cookies', str(cookies_file)]

    # Audio → WAV
    audio_out = workdir / 'media.%(ext)s'
    cmd_audio = [
        'yt-dlp','-q','-4', *cookie_args,
        '-x','--audio-format','wav','--audio-quality','0',
        '-o', str(audio_out), input_url
    ]
    sh(cmd_audio)

    wav_src = next(iter(workdir.glob('*.wav')), None)
    if wav_src is None:
        raise RuntimeError('Audio WAV introuvable après yt-dlp.')

    # Sous-titres
    print('[STEP] Récupération des sous-titres .vtt (FR préférés)…')
    vtt_path = None
    clients = ['tvhtml5ios', 'ios', 'android', 'web']  # priorité à tvhtml5ios (ton test OK)
    sub_flags = ['--write-subs', '--write-auto-sub'] if allow_auto else ['--write-subs']

    # ménage initial
    for old in workdir.glob('*.vtt'):
        try: old.unlink()
        except: pass

    for cli in clients:
        print(f'[INFO] Essai sous-titres avec client: {cli}')
        # ménage avant essai
        for old in list(workdir.glob('*.vtt')) + list(workdir.glob('*.srt')) + list(workdir.glob('*.srv*')):
            try: old.unlink()
            except: pass

        cmd_vtt = [
            'yt-dlp','-4', *cookie_args,
            '--extractor-args', f'youtube:player_client={cli}',
            '--skip-download', *sub_flags,
            '--convert-subs','vtt',
            '--sub-format','vtt/srt',
            '--sub-langs', subs_lang,
            '-o', str(workdir / '%(id)s.%(ext)s'),
            input_url
        ]
        try:
            sh(cmd_vtt)
        except Exception as e:
            print(f'[WARN] Essai client {cli} a levé une erreur (on continue): {e}')
            continue

        found = sorted(workdir.glob('*.vtt'))
        print(f"[DEBUG] Fichiers .vtt trouvés ({cli}): {[f.name for f in found]}")
        if found:
            pref_order = ['fr', 'fr-FR', 'fr-CA']
            chosen = None
            for tag in pref_order:
                for f in found:
                    name = f.name.lower()
                    if f'.{tag.lower()}.' in name or name.endswith(f'.{tag.lower()}.vtt'):
                        chosen = f; break
                if chosen: break
            vtt_path = chosen or found[0]
            break

    if vtt_path is None:
        print('[WARN] Sous-titres .vtt introuvables.')

    # Conversion → mono 16 kHz
    out_wav = workdir / 'audio_16k_mono.wav'
    sh(['ffmpeg','-y','-i', str(wav_src), '-ac','1','-ar','16000', str(out_wav)])
    print(f'[CHECK] Audio prêt: {out_wav}')

    return str(out_wav), (str(vtt_path) if vtt_path else None), str(workdir)

from dataclasses import dataclass

@dataclass
class Caption:
    start: float
    end: float
    text: str

def parse_vtt(vtt_path: str) -> List[Caption]:
    try:
        import webvtt
    except Exception as e:
        raise RuntimeError("Installe 'webvtt-py' :  pip install webvtt-py") from e

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

def _smooth_segments(segs: List[SpkSeg], min_speech: float = 0.35, min_pause: float = 0.25) -> List[SpkSeg]:
    """Lissage simple : fusionne petits silences & absorbe micro-segments."""
    if not segs:
        return segs
    segs = sorted(segs, key=lambda s: s.start)

    # 1) fusionner les gaps courts entre segments du même locuteur
    fused: List[SpkSeg] = [SpkSeg(segs[0].start, segs[0].end, segs[0].speaker)]
    for s in segs[1:]:
        last = fused[-1]
        if s.speaker == last.speaker and (s.start - last.end) <= min_pause:
            last.end = max(last.end, s.end)
        else:
            fused.append(SpkSeg(s.start, s.end, s.speaker))

    # 2) absorber les segments trop courts
    i = 0
    while i < len(fused):
        cur = fused[i]
        dur = cur.end - cur.start
        if dur < min_speech and len(fused) > 1:
            if i == 0:
                fused[1].start = min(fused[1].start, cur.start)
                fused.pop(0)
                continue
            if i == len(fused) - 1:
                fused[-2].end = max(fused[-2].end, cur.end)
                fused.pop()
                break
            prev, nxt = fused[i - 1], fused[i + 1]
            # préférence pour voisin même locuteur sinon le plus long
            if prev.speaker == cur.speaker or (prev.end - prev.start) >= (nxt.end - nxt.start):
                prev.end = max(prev.end, cur.end)
                fused.pop(i)
                continue
            else:
                nxt.start = min(nxt.start, cur.start)
                fused.pop(i)
                continue
        i += 1
    return fused

def diarize_pyannote(wav_path: str, num_speakers: Optional[int]) -> List[SpkSeg]:
    try:
        import torch
        from pyannote.audio import Pipeline
    except Exception as e:
        raise RuntimeError("Installe 'pyannote.audio>=3.1.1'") from e

    hf_token = os.environ.get("HF_TOKEN", None)
    print("[STEP] Diarization (pyannote.audio)…")
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    try:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
        ).to(dev)
    except Exception as e:
        raise RuntimeError(
            "Échec chargement pipeline. Vérifie HF_TOKEN et accepte les conditions du modèle."
        ) from e

    # >>> IMPORTANT : sur 3.1, ne PAS passer min_speech_duration/min_silence_duration/collar
    diarization = pipe({"audio": wav_path}, num_speakers=num_speakers)

    segs: List[SpkSeg] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segs.append(SpkSeg(float(turn.start), float(turn.end), str(speaker)))
    segs.sort(key=lambda x: x.start)

    # lissage côté code pour éviter les micro-blips
    segs = _smooth_segments(segs, min_speech=0.35, min_pause=0.25)

    if not segs:
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
    utts: List[Utt] = []
    j = 0
    n = len(caps)
    for s in spk:
        buf = []
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
        for n in range(min(ngram_max, len(words)), 1, -1):
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

    raw_utts = build_utterances(spk, caps) if caps else []
    print(f"[CHECK] {len(raw_utts)} segments alignés avant filtrage.")
    utts = remove_near_duplicates(raw_utts, ngram_max=6, lookback_seconds=6.0)
    print(f"[CHECK] {len(utts)} segments après filtrage des répétitions.")

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
    try:
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--subs-lang", default="fr,fr-FR,fr-CA")
    ap.add_argument("--allow-auto", action="store_true")
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--keep-metadata", action="store_true")
    ap.add_argument("--healthcheck", action="store_true")
    args = ap.parse_args()

    if args.healthcheck:
        print("ok"); return

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

def download_youtube(input_url: str, subs_lang: str, allow_auto: bool):
    """Télécharge l'audio + récupère les VTT (fallback de clients yt-dlp) + convertit en mono 16 kHz.

    Args:
        input_url: URL YouTube
        subs_lang: ex. "fr,fr-FR,fr-CA"
        allow_auto: True pour autoriser les auto-captions
    Returns:
        (wav_path, vtt_path_or_None, workdir)
    """
    import uuid, os
    from pathlib import Path

    print("[STEP] Téléchargement audio YouTube…")
    workdir = Path(f"/tmp/yt_{uuid.uuid4().hex[:8]}")
    workdir.mkdir(parents=True, exist_ok=True)

    cookies = Path("/content/drive/MyDrive/mon_projet_politique/cookies.txt")
    cookie_args = ["--cookies", str(cookies)] if cookies.exists() else []

    # 1) Audio → WAV
    audio_out = workdir / "media.%(ext)s"
    cmd_audio = ["yt-dlp","-q","-4",*cookie_args,
                 "-x","--audio-format","wav","--audio-quality","0",
                 "-o", str(audio_out), input_url]
    sh(cmd_audio)

    wav_path = next(iter(workdir.glob("*.wav")), None)
    if wav_path is None:
        raise RuntimeError("Audio WAV introuvable après yt-dlp.")

    # 2) Sous-titres (fallback clients)
    print("[STEP] Récupération des sous-titres .vtt (FR préférés)…")
    vtt_path = None
    clients = ["tvhtml5ios", "ios", "android", "web"]
    sub_flags = (["--write-subs","--write-auto-sub"] if allow_auto else ["--write-subs"])

    for cli in clients:
        # Nettoyage .vtt précédents
        for old in workdir.glob("*.vtt"):
            try: old.unlink()
            except: pass

        # Tentative avec ce client
        # NOTE: on évite les PO tokens → clients TV/IOS/Android passent souvent sans.
        cmd_vtt = [
            "yt-dlp","-4",*cookie_args,
            "--extractor-args", f"youtube:player_client={cli}",
            "--skip-download", *sub_flags,
            "--sub-format","vtt/srt",
            "--sub-langs", subs_lang,
            "-o", str(workdir / "%(id)s.%(ext)s"),
            input_url
        ]
        try:
            sh(cmd_vtt)
        except Exception as e:
            print(f"[WARN] Essai sous-titres avec client {cli} a échoué: {e}")
            continue

        found = sorted(workdir.glob("*.vtt"))
        if found:
            # Sélection préférentielle
            pref = ["fr","fr-FR","fr-CA"]
            chosen = None
            for tag in pref:
                for f in found:
                    if tag.lower() in f.name.lower():
                        chosen = f; break
                if chosen: break
            vtt_path = chosen or found[0]
            break

    if vtt_path is None:
        print("[WARN] Sous-titres .vtt introuvables.")

    # 3) Conversion audio en mono 16 kHz
    out_wav = workdir / "audio_16k_mono.wav"
    sh(["ffmpeg","-y","-i", str(wav_path), "-ac","1","-ar","16000", str(out_wav)])
    print(f"[CHECK] Audio prêt: {out_wav}")

    return str(out_wav), (str(vtt_path) if vtt_path else None), str(workdir)
