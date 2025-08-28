#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
app.py — YouTube -> ASR (+diarisation si dispo) -> JSON propre (anti-répétitions agressif)
Sortie: {"meta": {...}, "utterances": [...], "text": "..."}
"""

import os
import re
import json
import argparse
import tempfile
import subprocess
from collections import deque
from difflib import SequenceMatcher

# ---------------------------
# Utils I/O
# ---------------------------

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{p.stderr}")
    return p.stdout.strip()

def download_audio(url, out_dir):
    """Télécharge l'audio via yt-dlp et renvoie le chemin wav."""
    audio_path = os.path.join(out_dir, "audio.wav")
    # Meilleur effort: 160k opus -> wav
    cmd = [
        "yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0",
        "-o", os.path.join(out_dir, "dl.%(ext)s"),
        url
    ]
    try:
        run(cmd)
    except Exception as e:
        raise RuntimeError(f"Echec yt-dlp: {e}")

    # chercher le wav
    cand = [f for f in os.listdir(out_dir) if f.endswith(".wav")]
    if not cand:
        raise RuntimeError("Aucun .wav trouvé après yt-dlp.")
    src = os.path.join(out_dir, cand[0])
    if src != audio_path:
        os.replace(src, audio_path)
    return audio_path

# ---------------------------
# ASR
# ---------------------------

def asr_faster_whisper(audio_path, lang, device):
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError("faster-whisper non disponible. Installez-le ou laissez l'app basculer sur 'whisper' : pip install faster-whisper") from e

    compute = "float16" if device == "cuda" else "int8"
    model_size = os.environ.get("WHISPER_MODEL", "large-v3")
    model = WhisperModel(model_size, device=device, compute_type=compute)

    segments_iter, info = model.transcribe(
        audio_path,
        language=lang,
        vad_filter=True,
        word_timestamps=True
    )

    words = []
    segments = []
    for seg in segments_iter:
        segments.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip()
        })
        if seg.words:
            for w in seg.words:
                if w.word is None: 
                    continue
                words.append({
                    "start": float(w.start) if w.start is not None else float(seg.start),
                    "end": float(w.end) if w.end is not None else float(seg.end),
                    "word": w.word
                })
        else:
            # fallback: découpe grossière au segment
            for token in seg.text.strip().split():
                words.append({
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "word": token
                })
    return words, segments, {"model": f"faster-whisper:{model_size}", "language": lang, "duration": getattr(info, "duration", None)}

def asr_whisper(audio_path, lang, device):
    try:
        import whisper
    except Exception as e:
        raise RuntimeError("openai-whisper non disponible. Installez-le : pip install -U openai-whisper") from e

    model_size = os.environ.get("WHISPER_MODEL", "large")
    model = whisper.load_model(model_size, device=device if device in ("cuda","cpu") else "cpu")
    # note: whisper officiel n'a pas de word timestamps fiables sans extra
    result = model.transcribe(audio_path, language=lang, verbose=False)
    words = []
    segments = []
    for s in result.get("segments", []):
        segments.append({
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": s.get("text","").strip()
        })
        # sans word-level, on tokenise grossièrement
        for token in s.get("text","").split():
            words.append({
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "word": token
            })
    return words, segments, {"model": f"openai-whisper:{model_size}", "language": lang, "duration": None}

def transcribe(audio_path, lang="fr", device="cuda"):
    # priorité à faster-whisper pour les timestamps de mots
    try:
        return asr_faster_whisper(audio_path, lang, device)
    except Exception as e_fw:
        print(f"[INFO] faster-whisper indisponible ({e_fw}). Bascule sur whisper.")
        return asr_whisper(audio_path, lang, device if device in ("cuda","cpu") else "cpu")

# ---------------------------
# Diarisation
# ---------------------------

def diarize_pyannote(audio_path, model_id="pyannote/speaker-diarization-3.1", num_speakers=None, hf_token=None):
    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        raise RuntimeError("pyannote.audio non disponible. Installez-le pour la diarisation, ou laissez l'app continuer sans.") from e

    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError("Token HuggingFace requis pour ce modèle pyannote (définissez HF_TOKEN).")

    pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)
    diar = pipeline(audio=audio_path, num_speakers=num_speakers)
    # Convertir en segments triés
    segs = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        segs.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker)
        })
    segs.sort(key=lambda x: (x["start"], x["end"]))
    # Normaliser label en SPEAKER_XX
    remap = {}
    next_id = 0
    for s in segs:
        if s["speaker"] not in remap:
            remap[s["speaker"]] = f"SPEAKER_{next_id:02d}"
            next_id += 1
        s["speaker"] = remap[s["speaker"]]
    return segs

def fallback_single_speaker(words):
    # un seul locuteur si pas de diarisation
    return [{"start": 0.0, "end": words[-1]["end"] if words else 0.0, "speaker": "SPEAKER_00"}]

def assign_words_to_speakers(words, diarization_segments):
    """Assigne chaque mot au locuteur dont le segment contient le milieu du mot."""
    if not diarization_segments:
        diarization_segments = fallback_single_speaker(words)

    diarization_segments = sorted(diarization_segments, key=lambda x: (x["start"], x["end"]))
    # Index linéaire
    utts = []
    cur_speaker = None
    cur_words = []
    cur_start = None
    cur_end = None

    def flush():
        nonlocal cur_words, cur_speaker, cur_start, cur_end
        if cur_words:
            utts.append({
                "speaker": cur_speaker,
                "start": float(cur_start),
                "end": float(cur_end),
                "text": " ".join(w["word"] for w in cur_words).strip()
            })
        cur_words = []
        cur_speaker = None
        cur_start = None
        cur_end = None

    # helper pour trouver le speaker courant par temps
    i = 0
    n = len(diarization_segments)
    for w in words:
        mid = (w["start"] + w["end"]) / 2.0
        # avancer i tant que segment i ne couvre pas mid
        while i < n and not (diarization_segments[i]["start"] <= mid <= diarization_segments[i]["end"]):
            if i+1 < n and diarization_segments[i+1]["start"] <= mid:
                i += 1
            else:
                break
        # choisir seg couvrant mid si possible, sinon le plus proche
        spk = None
        if 0 <= i < n and diarization_segments[i]["start"] <= mid <= diarization_segments[i]["end"]:
            spk = diarization_segments[i]["speaker"]
        else:
            # plus proche
            best = None
            for j in range(max(0,i-2), min(n, i+3)):
                d = min(abs(mid - diarization_segments[j]["start"]), abs(mid - diarization_segments[j]["end"]))
                if best is None or d < best[0]:
                    best = (d, diarization_segments[j]["speaker"])
            spk = best[1] if best else "SPEAKER_00"

        if cur_speaker is None:
            cur_speaker = spk
            cur_start = w["start"]
            cur_end = w["end"]
            cur_words = [w]
        elif spk == cur_speaker:
            cur_words.append(w)
            cur_end = w["end"]
        else:
            flush()
            cur_speaker = spk
            cur_start = w["start"]
            cur_end = w["end"]
            cur_words = [w]
    flush()
    # fusionner micro-utt très courtes (bruit)
    merged = []
    for u in utts:
        if merged and u["speaker"] == merged[-1]["speaker"] and (u["end"] - u["start"] < 0.5):
            merged[-1]["text"] = (merged[-1]["text"] + " " + u["text"]).strip()
            merged[-1]["end"] = u["end"]
        else:
            merged.append(u)
    return merged

# ---------------------------
# Nettoyage agressif anti-répétitions
# ---------------------------

_SENT_SPLIT = re.compile(r'(?<=[\.\!\?\:;])\s+')
_WS = re.compile(r'\s+')

def normalize_text(s: str) -> str:
    s = s.replace("\u00A0", " ")
    s = re.sub(r'[’`]', "'", s)
    s = re.sub(r'\s+([,;:\.\!\?])', r'\1', s)   # pas d'espace avant ponctuation finale
    s = _WS.sub(" ", s)
    return s.strip()

def collapse_stutters(s: str) -> str:
    # supprime répétitions immédiates de mots courts / syllabes (e.g., "qu'on qu'on", "le le")
    s = re.sub(r'\b(\w{1,4})(?:\s+\1){1,}\b', r'\1', s, flags=re.IGNORECASE)
    # supprime les doublons immédiats de groupes de mots (2–8 tokens)
    tokens = s.split()
    n = len(tokens)
    i = 0
    out = []
    while i < n:
        consumed = False
        # essayer du plus long au plus court
        for L in range(min(8, n - i), 1, -1):
            a = tokens[i:i+L]
            b = tokens[i+L:i+2*L]
            if a and b and a == b:
                out.extend(a)
                i += 2*L
                consumed = True
                break
        if not consumed:
            out.append(tokens[i])
            i += 1
    return " ".join(out)

def remove_internal_ngrams(s: str, max_ng=8):
    # supprime n-grammes internes répétés (même s'ils ne sont pas adjacents) de manière agressive
    tokens = s.split()
    seen_spans = set()
    i = 0
    out = []
    while i < len(tokens):
        removed = False
        for L in range(min(max_ng, len(tokens) - i), 3, -1):
            ngram = tuple(tokens[i:i+L])
            # si cet ngram apparaît déjà dans la sortie récente -> skip
            if L >= 4 and ngram in seen_spans:
                i += L
                removed = True
                break
        if not removed:
            out.append(tokens[i])
            # mémoriser n-grammes récents pour future détection
            for L in range(4, min(max_ng, len(out)) + 1):
                seen_spans.add(tuple(out[-L:]))
            i += 1
    return " ".join(out)

def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def cross_utterance_dedup(utterances, sim_thr=0.92, window=60):
    """
    Supprime les phrases quasi-identiques à celles déjà dites récemment (tous locuteurs confondus).
    """
    recent = deque(maxlen=window)
    cleaned = []
    for u in utterances:
        sents = [x.strip() for x in _SENT_SPLIT.split(normalize_text(u["text"])) if x.strip()]
        kept = []
        for sent in sents:
            dup = any(similar(sent, old) >= sim_thr for old in recent)
            if not dup:
                kept.append(sent)
                recent.append(sent)
        text_clean = " ".join(kept).strip()
        new_u = dict(u)
        new_u["text"] = text_clean
        cleaned.append(new_u)
    # enlever utt vides
    cleaned = [u for u in cleaned if u["text"]]
    return cleaned

def clean_utterance_text(s: str, ultra=True):
    s = normalize_text(s)
    s = collapse_stutters(s)
    if ultra:
        s = remove_internal_ngrams(s, max_ng=10)
    return normalize_text(s)

def aggressively_clean(utterances, ultra=True):
    # 1) intra-utt
    for u in utterances:
        u["text"] = clean_utterance_text(u["text"], ultra=ultra)
    # 2) inter-utt
    utterances = cross_utterance_dedup(utterances, sim_thr=0.92, window=80)
    # 3) fusionne consécutifs même speaker après nettoyage
    merged = []
    for u in utterances:
        if merged and merged[-1]["speaker"] == u["speaker"]:
            # si proche et court, fusionne
            if u["start"] - merged[-1]["end"] < 1.5:
                merged[-1]["text"] = (merged[-1]["text"] + " " + u["text"]).strip()
                merged[-1]["end"] = u["end"]
            else:
                merged.append(u)
        else:
            merged.append(u)
    # 4) deuxième passe légère pour retirer duplications introduites par fusion
    for u in merged:
        u["text"] = collapse_stutters(u["text"])
    # drop vides
    merged = [u for u in merged if u["text"]]
    return merged

# ---------------------------
# Build final text
# ---------------------------

def build_full_text(utterances):
    # Concaténation simple par ordre temporel, sans tags locuteur dans "text"
    parts = []
    for u in utterances:
        if u["text"]:
            parts.append(u["text"])
    text = " ".join(parts).strip()
    return normalize_text(text)

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="YouTube -> ASR -> (Diarisation) -> JSON propre")
    ap.add_argument("--url", type=str, required=True, help="URL YouTube")
    ap.add_argument("--lang", type=str, default="fr", help="Langue forcée pour l'ASR (ex: fr)")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"], help="cuda ou cpu")
    ap.add_argument("--num_speakers", type=int, default=None, help="Hint nombre de locuteurs pour la diarisation")
    ap.add_argument("--diarization_model", type=str, default="pyannote/speaker-diarization-3.1", help="Modèle pyannote")
    ap.add_argument("--out", type=str, default="/content/output.json", help="Chemin du JSON de sortie")
    ap.add_argument("--no_diarization", action="store_true", help="Désactiver la diarisation pyannote")
    ap.add_argument("--ultra_clean", action="store_true", help="Nettoyage hyper agressif (par défaut True)")
    args = ap.parse_args()

    ultra_flag = True if args.ultra_clean or True else False  # agressif par défaut

    with tempfile.TemporaryDirectory() as tmp:
        # 1) download
        audio_path = download_audio(args.url, tmp)

        # 2) ASR
        words, segments, asr_info = transcribe(audio_path, lang=args.lang, device=args.device)

        # 3) Diarisation
        diar_segments = []
        if not args.no_diarization:
            try:
                diar_segments = diarize_pyannote(
                    audio_path,
                    model_id=args.diarization_model,
                    num_speakers=args.num_speakers,
                    hf_token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
                )
            except Exception as e:
                print(f"[INFO] Diarisation indisponible ({e}). On continue avec un seul locuteur.")
                diar_segments = fallback_single_speaker(words)
        else:
            diar_segments = fallback_single_speaker(words)

        # 4) mots -> utterances par speaker
        utterances = assign_words_to_speakers(words, diar_segments)

        # 5) nettoyage agressif anti-répétitions
        utterances = aggressively_clean(utterances, ultra=ultra_flag)

        # 6) texte global
        full_text = build_full_text(utterances)

        # 7) meta
        meta = {
            "source": "YouTube",
            "video_url": args.url,
            "subs_lang": os.environ.get("SUBS_LANG", "fr,fr-FR,fr-CA"),
            "allow_auto_subs": True,
            "diarization_model": args.diarization_model if not args.no_diarization else None,
            "num_speakers_request": args.num_speakers,
            "device": args.device,
            "asr_model": asr_info.get("model"),
            "language": asr_info.get("language"),
            "duration_sec": asr_info.get("duration"),
            "cleaning": {
                "ultra_aggressive": True,
                "cross_speaker_dedup_similarity": 0.92,
                "window_sentences": 80,
                "intra_ngram_max": 10
            }
        }

        # 8) sortie
        out = {
            "meta": meta,
            "utterances": utterances,
            "text": full_text
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        print(f"[OK] JSON écrit -> {args.out}")

if __name__ == "__main__":
    main()
