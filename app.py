#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube → transcript (VTT) complet + diarization (extrait ou full) [YTDLP-only]
+ Anti-doublons VTT
+ Ponctuation & fusion micro-segments
+ Suppression des redites consécutives (même entre speakers)
+ Export --export-srt / --export-csv / --export-md

Prérequis :
  - ffmpeg dans le PATH
  - HUGGINGFACE_TOKEN défini (et terms pyannote acceptés)
"""

import argparse, json, os, re, shutil, subprocess, sys, tempfile, string
from pathlib import Path
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher

# ---------------- Logs ----------------
def print_step(msg: str) -> None: print(f"[STEP] {msg}", flush=True)
def print_check(msg: str) -> None: print(f"[CHECK] {msg}", flush=True)
def fail(msg: str, code: int = 1) -> None: print(f"[ERROR] {msg}", file=sys.stderr, flush=True); sys.exit(code)

# -------------- Sanity ---------------
def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None: fail("ffmpeg non trouvé dans le PATH")

def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

# ------------ yt-dlp I/O -------------
def download_youtube_audio(url: str, out_dir: Path) -> Path:
    from yt_dlp import YoutubeDL
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {"format": "bestaudio/best","outtmpl": str(out_dir / "%(title).200s.%(ext)s"),
                "noplaylist": True,"quiet": True,"no_warnings": True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded = ydl.prepare_filename(info)
    return Path(downloaded)

def download_youtube_subs(url: str, out_dir: Path) -> Optional[Path]:
    from yt_dlp import YoutubeDL, utils
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "skip_download": True, "writesubtitles": True, "writeautomaticsub": True,
        "subtitleslangs": ["fr","fr-FR","fr-CA"],
        "subtitlesformat": "vtt",
        "noplaylist": True, "quiet": True, "no_warnings": True,
        "extractor_retries": 3, "sleep_interval_requests": 1,
        "outtmpl": str(out_dir / "%(title).200s.%(ext)s"),
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
    except utils.DownloadError:
        return None
    vtts = sorted(out_dir.glob("*.vtt"))
    return vtts[0] if vtts else None

# ----------- Audio (ffmpeg) ----------
def convert_to_wav_mono16k(src: Path, dst: Path, start: Optional[float] = None, duration: Optional[int] = None) -> Path:
    cmd = ["ffmpeg", "-y"]
    if start is not None: cmd += ["-ss", str(start)]
    cmd += ["-i", str(src)]
    if duration is not None: cmd += ["-t", str(duration)]
    cmd += ["-ac","1","-ar","16000","-vn", str(dst)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0: fail(f"Echec conversion audio (ffmpeg): {p.stderr[:400]}")
    return dst

# ----------- VTT parsing + dedup ----
def to_seconds(ts: str) -> float:
    ts = ts.replace(",", ".")
    h,m,s = ts.split(":")
    return int(h)*3600 + int(m)*60 + float(s)

def parse_vtt_raw(vtt_path: Path) -> List[Dict[str,Any]]:
    import webvtt
    cues=[]
    for cue in webvtt.read(str(vtt_path)):
        text = " ".join(cue.text.strip().split())
        if not text: continue
        cues.append({"start": to_seconds(cue.start), "end": to_seconds(cue.end), "text": text})
    return cues

_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation + "’“”«»…—–"})
def _norm_tok(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower().translate(_PUNCT_TABLE)).strip()
def _split_words(text: str) -> List[str]:
    return [w for w in _norm_tok(text).split() if w]
def _dedup_concat_words(prev_words: List[str], new_words: List[str], max_overlap: int = 8) -> int:
    max_k = min(max_overlap, len(prev_words), len(new_words))
    for k in range(max_k, 0, -1):
        if prev_words[-k:] == new_words[:k]:
            return k
    return 0

def build_clean_captions_and_words(vtt_path: Path, dedup: bool = True):
    """
    Enlève les répétitions inter-captions, renvoie:
      - captions_clean: [{start,end,text}] (texte dédoublonné)
      - words_all: mots horodatés approx.
      - full_text: tout le texte sans répétitions
    """
    raw = parse_vtt_raw(vtt_path)
    captions_clean = []
    words_all: List[Dict[str,Any]] = []
    full_text_words: List[str] = []

    for cue in raw:
        cue_words_norm = _split_words(cue["text"])
        if not cue_words_norm: continue

        start, end = cue["start"], cue["end"]
        dur = max(1e-3, end - start)
        step = dur / max(1, len(cue_words_norm))
        cue_words_timed = [{"start": start + j*step, "end": start + (j+1)*step, "text": w}
                           for j, w in enumerate(cue_words_norm)]

        if dedup and words_all:
            prev_tail = [x["text"] for x in words_all[-8:]]
            k = _dedup_concat_words(prev_tail, cue_words_norm, max_overlap=8)
        else:
            k = 0

        cue_words_timed = cue_words_timed[k:]
        cue_words_norm = cue_words_norm[k:]
        if not cue_words_timed:
            continue

        words_all.extend(cue_words_timed)
        full_text_words.extend(cue_words_norm)
        captions_clean.append({"start": start, "end": end, "text": " ".join(cue_words_norm)})

    full_text = " ".join(full_text_words)
    return captions_clean, words_all, full_text

# ------------- Diarization ----------
def diarize_pyannote(wav_path: Path, num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token: fail("HUGGINGFACE_TOKEN/HF_TOKEN non défini.")
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        fail("pyannote.audio manquant. pip install 'pyannote.audio>=3.1.1'")

    print_step("Diarization (pyannote.audio)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        ).to(device)
        print_check(f"pyannote device: {device.type}")
    except Exception as e:
        import traceback
        print("TRACEBACK:\n", traceback.format_exc())
        fail(f"Echec chargement pipeline: {e}")

    infer_args = {"audio": str(wav_path)}
    diar = pipeline(infer_args, num_speakers=num_speakers) if num_speakers is not None else pipeline(infer_args)

    segs = [{"start": float(t.start), "end": float(t.end), "speaker": spk}
            for t, _, spk in diar.itertracks(yield_label=True)]
    segs.sort(key=lambda x: x["start"])
    print_check(f"Diarization ok. {len(segs)} segments.")
    return segs

# ------------- Alignement -----------
def align_words_to_diarization(words: List[Dict[str, Any]], segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not words: return []
    if not segs:
        txt = " ".join(w["text"] for w in words).strip()
        return [{"speaker":"SPEAKER_00","start":words[0]["start"],"end":words[-1]["end"],"text":txt}]

    segs = sorted(segs, key=lambda s: (s["start"], s["end"]))
    def speaker_at(t: float) -> str:
        for s in segs:
            if s["start"] <= t <= s["end"]:
                return s["speaker"]
        nearest = min(segs, key=lambda s: min(abs(s["start"]-t), abs(s["end"]-t)))
        return nearest["speaker"]

    utt=[]; cur_spk=None; cur_start=None; buf=[]
    for w in words:
        mid = (w["start"]+w["end"])/2.0
        spk = speaker_at(mid)
        if cur_spk is None:
            cur_spk, cur_start, buf = spk, w["start"], [w["text"]]
        elif spk == cur_spk:
            buf.append(w["text"])
        else:
            text = re.sub(r"\s+"," "," ".join(buf)).strip()
            if text:
                utt.append({"speaker":cur_spk,"start":float(cur_start),"end":float(w["start"]),"text":text})
            cur_spk, cur_start, buf = spk, w["start"], [w["text"]]
    if buf:
        text = re.sub(r"\s+"," "," ".join(buf)).strip()
        utt.append({"speaker":cur_spk,"start":float(cur_start),"end":float(words[-1]["end"]),"text":text})

    # fusion consécutifs du même speaker
    merged=[]
    for u in utt:
        if merged and merged[-1]["speaker"]==u["speaker"]:
            merged[-1]["text"]=(merged[-1]["text"]+" "+u["text"]).strip()
            merged[-1]["end"]=u["end"]
        else:
            merged.append(u)
    return merged

# ------------- Lisibilité -----------
_PUNCT_MODEL = None
def punctuate_text(text: str) -> str:
    """Restaure ponctuation + majuscules. Fallback: retourne le texte brut."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text: return text
    try:
        global _PUNCT_MODEL
        if _PUNCT_MODEL is None:
            from deepmultilingualpunctuation import PunctuationModel
            _PUNCT_MODEL = PunctuationModel()
        punct = _PUNCT_MODEL.restore_punctuation(text)
    except Exception:
        punct = text
    parts = re.split(r'([\.!?…]+)\s*', punct)
    rebuilt=[]
    for i in range(0, len(parts), 2):
        sent = parts[i].strip()
        end = parts[i+1] if i+1 < len(parts) else ""
        if sent:
            sent = sent[0].upper() + sent[1:]
            rebuilt.append(sent + (end if end else ""))
    return " ".join(rebuilt).strip()

def merge_short_utterances(utts, min_len_chars=60, max_gap_sec=0.8):
    """Fusionne segments trop courts avec le suivant du même speaker (ou pause très courte)."""
    if not utts: return utts
    merged = [utts[0].copy()]
    for u in utts[1:]:
        last = merged[-1]
        same_spk = (u["speaker"] == last["speaker"])
        gap = max(0.0, u["start"] - last["end"])
        if same_spk and (gap <= max_gap_sec or len(last["text"]) < min_len_chars):
            last["text"] = (last["text"] + " " + u["text"]).strip()
            last["end"] = u["end"]
        else:
            merged.append(u.copy())
    return merged

def _norm_for_compare(t: str) -> str:
    t = t.lower().translate(_PUNCT_TABLE)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def dedup_consecutive_repetitions(utts, sim_ratio=0.97, max_len=140):
    """
    Supprime les redites consécutives même si speakers différents.
    - supprime u[i] si u[i] ~ u[i-1] (texte quasi identique)
    - plus agressif pour les courtes répliques (<= max_len)
    """
    if not utts: return utts
    pruned = [utts[0].copy()]
    for u in utts[1:]:
        prev = pruned[-1]
        a = _norm_for_compare(prev["text"])
        b = _norm_for_compare(u["text"])
        similar = False
        if a and b:
            if a == b:
                similar = True
            else:
                r = SequenceMatcher(None, a, b).ratio()
                if r >= sim_ratio:
                    similar = True
                # cas "echo" très court : b inclus dans a ou inverse
                if not similar and (len(a) <= max_len or len(b) <= max_len):
                    if a in b or b in a:
                        similar = True
        if similar:
            # on garde le premier; on peut éventuellement étendre l'intervalle
            prev["end"] = max(prev["end"], u["end"])
        else:
            pruned.append(u.copy())
    return pruned

# ------------- Exports --------------
def _fmt_srt(t: float) -> str:
    ms = int(round((t - int(t)) * 1000))
    h = int(t//3600); m = int((t%3600)//60); s = int(t%60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def export_srt(utterances: List[Dict[str,Any]], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as srt:
        for i, u in enumerate(utterances, 1):
            srt.write(f"{i}\n{_fmt_srt(u['start'])} --> {_fmt_srt(u['end'])}\n[{u['speaker']}] {u['text']}\n\n")
    print_check(f"SRT écrit: {out_path.resolve()}")

def export_csv(utterances: List[Dict[str,Any]], out_path: Path) -> None:
    import csv
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_sec","end_sec","speaker","text"])
        for u in utterances:
            w.writerow([u["start"], u["end"], u["speaker"], u["text"]])
    print_check(f"CSV écrit: {out_path.resolve()}")

def export_markdown(utterances: List[Dict[str,Any]], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Transcript nettoyé\n\n")
        current=None
        for u in utterances:
            if u["speaker"] != current:
                current = u["speaker"]
                f.write(f"\n## {current}\n\n")
            f.write(u["text"].strip() + " ")
    print_check(f"Markdown écrit: {out_path.resolve()}")

# ------------- Helpers --------------
def parse_hhmmss(s: Optional[str]) -> Optional[float]:
    if not s: return None
    s=s.strip()
    parts=s.split(":")
    if len(parts)==3:
        h,m,sec=parts; return int(h)*3600+int(m)*60+float(sec)
    if len(parts)==2:
        m,sec=parts; return int(m)*60+float(sec)
    return float(s)

# ---------------- Main --------------
def main() -> None:
    parser=argparse.ArgumentParser(description="YouTube → VTT (anti-doublons) + diarization + lisibilité + exports")
    parser.add_argument("--input", type=str, help="URL YouTube")
    parser.add_argument("--output", type=str, default="result.json", help="Chemin du JSON de sortie")
    parser.add_argument("--healthcheck", action="store_true", help="Vérifie l'environnement")
    parser.add_argument("--start", type=str, default=None, help="Début HH:MM:SS ou MM:SS de l'extrait à diariser")
    parser.add_argument("--duration", type=int, default=None, help="Durée en secondes de l'extrait à diariser")
    parser.add_argument("--num-speakers", type=int, default=None, help="Fixer le nombre de locuteurs")
    parser.add_argument("--no-dedup-vtt", action="store_true", help="Désactive l'anti-doublons VTT")
    parser.add_argument("--punctuate", action="store_true", help="Restaure la ponctuation des utterances")
    parser.add_argument("--min-utt-chars", type=int, default=60, help="Fusionne les segments trop courts (<N chars)")
    parser.add_argument("--merge-gap-sec", type=float, default=0.8, help="Fusion si pause <= N secondes")
    parser.add_argument("--no-cross-dedup", action="store_true", help="Ne pas supprimer les redites consécutives entre speakers")
    parser.add_argument("--export-srt", action="store_true", help="Écrit un .srt (utterances)")
    parser.add_argument("--export-csv", action="store_true", help="Écrit un .csv (utterances)")
    parser.add_argument("--export-md", action="store_true", help="Écrit un .md lisible (paragraphes par speaker)")
    args=parser.parse_args()

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
        print_check("Healthcheck terminé ✅"); sys.exit(0)

    if not args.input: fail("--input requis (sauf --healthcheck).")
    if not is_url(args.input): fail("--input doit être une URL YouTube.")
    ensure_ffmpeg()

    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td)

        # 1) Audio
        print_step("Téléchargement audio YouTube...")
        media_path=download_youtube_audio(args.input, tmp)
        print_check(f"Fichier téléchargé: {media_path.name}")

        # 2) Sous-titres (FR)
        print_step("Récupération des sous-titres .vtt (FR)...")
        vtt=download_youtube_subs(args.input, tmp)
        if vtt is None: fail("Pas de sous-titres FR disponibles (YTDLP-only).")
        print_check(f"Sous-titres trouvés: {vtt.name}")

        # 3) Parse + anti-doublons
        captions, words_all, full_text = build_clean_captions_and_words(vtt, dedup=(not args.no_dedup_vtt))

        # 4) Conversion audio (extrait si demandé)
        start_sec=parse_hhmmss(args.start) if args.start else None
        wav_path=tmp/"audio_16k_mono.wav"
        print_step("Conversion en WAV mono 16k...")
        convert_to_wav_mono16k(media_path, wav_path, start=start_sec, duration=args.duration)
        print_check(f"Audio prêt: {wav_path}")

        # 5) Restreindre les mots à l'extrait (si fourni)
        if start_sec is not None and args.duration is not None:
            end_sec=start_sec+args.duration
            words_clip=[w for w in words_all if start_sec <= (w["start"]+w["end"]) / 2.0 <= end_sec]
        else:
            words_clip=words_all
        print_check(f"{len(words_clip)} 'mots' à aligner.")

        # 6) Diarization
        spk_segments=diarize_pyannote(wav_path, num_speakers=args.num_speakers)

        # 7) Alignement
        print_step("Alignement mots ↔ locuteurs...")
        utterances=align_words_to_diarization(words_clip, spk_segments)
        print_check(f"{len(utterances)} blocs speaker+texte.")

        # 8) Lisibilité (ponctuation, fusion, anti-redites consécutives)
        if args.punctuate:
            for u in utterances:
                u["text"] = punctuate_text(u["text"])
        utterances = merge_short_utterances(
            utterances,
            min_len_chars=args.min_utt_chars,
            max_gap_sec=args.merge_gap_sec
        )
        if not args.no_cross_dedup:
            utterances = dedup_consecutive_repetitions(utterances)

        # 9) JSON
        out=Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
        result={"source": args.input,
                "params":{"start": args.start,"duration": args.duration,"num_speakers": args.num_speakers,
                          "dedup_vtt": not args.no_dedup_vtt, "punctuate": args.punctuate,
                          "min_utt_chars": args.min_utt_chars, "merge_gap_sec": args.merge_gap_sec,
                          "cross_dedup": not args.no_cross_dedup},
                "transcript":{"captions": captions, "text": full_text},
                "utterances": utterances,
                "summary":{"num_captions": len(captions),"num_words_all": len(words_all),
                           "num_words_clip": len(words_clip),"num_speaker_segments": len(spk_segments),
                           "num_utterances": len(utterances)}}
        with open(out,"w",encoding="utf-8") as f: json.dump(result,f,ensure_ascii=False,indent=2)
        print_check(f"JSON écrit: {out.resolve()}")

        # 10) Exports optionnels
        base = out.with_suffix("")
        if args.export_srt:
            export_srt(utterances, base.with_suffix(".srt"))
        if args.export_csv:
            export_csv(utterances, base.with_suffix(".csv"))
        if args.export_md:
            export_markdown(utterances, base.with_suffix(".md"))

        print("Terminé ✅")

if __name__=="__main__":
    main()
