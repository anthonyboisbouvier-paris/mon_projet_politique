import argparse, json, re, unicodedata, difflib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# --------- Normalisation & utilitaires ---------
def _normalize(txt: str) -> str:
    if not txt:
        return ""
    t = txt.lower().replace("’", "'")
    t = unicodedata.normalize("NFD", t)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

_WORD = r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:['’-][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)?"
_PHRASE_REPEAT = re.compile(rf"(?i)\b(({_WORD})(?:\s+{_WORD}){{0,3}})\b(?:\s+\1\b)+")

def _collapse_internal_repeats(text: str) -> str:
    if not text: return text
    cur = re.sub(rf"(?i)\b({_WORD})\b(?:\s+\1\b)+", r"\1", text)
    prev = None
    while cur != prev:
        prev = cur
        cur = _PHRASE_REPEAT.sub(r"\1", cur)
    cur = re.sub(r"\s+", " ", cur).strip()
    return cur

def _sim(a: str, b: str) -> float:
    if not a or not b: return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

# --------- Chargement / sauvegarde ---------
def _pick_utterances_schema(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Retourne (utterances, schema) où schema ∈ {'utterances','segments','segments_by_speaker','none'}.
    Utterance dict attendu: {speaker?, start?, end?, text}
    """
    if isinstance(data.get("utterances"), list):
        return data["utterances"], "utterances"
    if isinstance(data.get("segments"), list):
        # segments style Whisper/pyannote; parfois sans speaker
        out = []
        for s in data["segments"]:
            out.append({
                "speaker": s.get("speaker", "SPEAKER_00"),
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", s.get("start", 0.0))),
                "text": s.get("text", "") or "",
            })
        return out, "segments"
    if isinstance(data.get("segments_by_speaker"), dict):
        out = []
        for spk, segs in data["segments_by_speaker"].items():
            for s in segs:
                out.append({
                    "speaker": spk,
                    "start": float(s.get("start", 0.0)),
                    "end": float(s.get("end", s.get("start", 0.0))),
                    "text": s.get("text", "") or "",
                })
        out.sort(key=lambda x: (x.get("start", 0.0), x.get("end", 0.0)))
        return out, "segments_by_speaker"
    return [], "none"

def _write_back(data: Dict[str, Any], utterances: List[Dict[str, Any]], new_text: str, schema: str) -> Dict[str, Any]:
    # On ne touche pas aux métadonnées existantes ; on met à jour utterances + text
    data = dict(data)  # shallow copy
    data["utterances"] = utterances
    data["text"] = new_text
    # On laisse intacts les autres structures si elles existent (au besoin tu pourras les supprimer)
    return data

# --------- Coeur du dédoublonnage ---------
def dedupe_utterances(
    utts: List[Dict[str, Any]],
    window_sec: float = 30.0,
    sim_drop: float = 0.90,
    near_merge_gap: float = 1.5,
    near_merge_sim: float = 0.80,
    cross_speaker_drop: bool = True,
) -> List[Dict[str, Any]]:
    """
    - Supprime les répétitions internes ('bonjour bonjour', 'il est il est', etc.).
    - Enlève les segments quasi identiques apparus dans une *fenêtre glissante* (même entre locuteurs si voulu).
    - Fusionne les doublons très proches *si même locuteur*.
    """
    if not utts: return utts
    # tri temporel (sécurité)
    utts = sorted(utts, key=lambda u: (float(u.get("start", 0.0)), float(u.get("end", 0.0))))

    # 1) nettoyage intra-segment
    for u in utts:
        u["text"] = _collapse_internal_repeats(u.get("text",""))

    cleaned: List[Dict[str, Any]] = []
    recent = []  # [{end, ntxt, spk}]
    def purge_recent(now: float):
        cutoff = now - window_sec
        while recent and recent[0]["end"] < cutoff:
            recent.pop(0)

    for u in utts:
        start = float(u.get("start", 0.0))
        end   = float(u.get("end", start))
        spk   = u.get("speaker", "SPEAKER_00")
        txt   = u.get("text","") or ""
        ntxt  = _normalize(txt)

        purge_recent(start)

        drop = False
        if ntxt:
            # Construit un "tail" du contexte récent (uniquement mêmes spk si cross_speaker_drop=False)
            pool = recent if cross_speaker_drop else [r for r in recent if r["spk"] == spk]
            tail = " ".join(r["ntxt"] for r in pool)
            tail = tail[-max(300, len(ntxt)*2):] if tail else ""
            if tail:
                if ntxt in tail:
                    drop = True
                elif _sim(ntxt, tail) >= sim_drop:
                    drop = True

        if drop:
            continue

        # Fusion si quasi contigu ET même speaker ET très similaire
        if cleaned:
            last = cleaned[-1]
            gap = start - float(last.get("end", start))
            if (last.get("speaker","") == spk) and (gap <= near_merge_gap):
                last_ntxt = _normalize(last.get("text",""))
                if _sim(ntxt, last_ntxt) >= near_merge_sim:
                    # garde le texte le plus long (évite de perdre de l'info)
                    if len(txt) > len(last.get("text","")):
                        last["text"] = txt
                    last["end"] = max(float(last.get("end", end)), end)
                    # maj mémoire
                    if recent:
                        recent[-1]["end"] = last["end"]
                        recent[-1]["ntxt"] = _normalize(last["text"])
                    continue

        kept = {
            "speaker": spk,
            "start": start,
            "end": end,
            "text": txt
        }
        cleaned.append(kept)
        recent.append({"end": end, "ntxt": ntxt, "spk": spk})

    # 3) second passage: encore enlever les micro-bégaiements internes
    for u in cleaned:
        u["text"] = _collapse_internal_repeats(u.get("text",""))

    return cleaned

def rebuild_text(utts: List[Dict[str, Any]]) -> str:
    raw = " ".join(u.get("text","") for u in utts)
    return _collapse_internal_repeats(raw)

# --------- CLI ---------
def main():
    ap = argparse.ArgumentParser(description="Nettoie un JSON d'app.py (suppression agressive des répétitions).")
    ap.add_argument("--input", "-i", required=True, help="Chemin du JSON en entrée (app.py).")
    ap.add_argument("--output", "-o", required=True, help="Chemin du JSON nettoyé.")
    ap.add_argument("--window-sec", type=float, default=30.0, help="Fenêtre de détection des doublons (sec).")
    ap.add_argument("--sim-drop", type=float, default=0.90, help="Seuil de similarité pour DROP (0..1).")
    ap.add_argument("--near-merge-gap", type=float, default=1.5, help="Gap max pour fusionner (sec).")
    ap.add_argument("--near-merge-sim", type=float, default=0.80, help="Seuil de similarité pour fusion (0..1).")
    ap.add_argument("--no-cross-speaker-drop", action="store_true",
                    help="Ne supprime pas un doublon si dit par un autre locuteur (par défaut on supprime quand même).")
    ap.add_argument("--aggressive", action="store_true",
                    help="Raccourcis de réglages plus durs (sim_drop=0.93, near_merge_sim=0.85).")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(inp.read_text(encoding="utf-8"))
    utts, schema = _pick_utterances_schema(data)

    if not utts and data.get("text"):
        # Fallback texte seul : on fabrique une pseudo-utterance unique
        utts = [{"speaker":"SPEAKER_00","start":0.0,"end":0.0,"text": data["text"]}]
        schema = "text-only"

    if args.aggressive:
        args.sim_drop = max(args.sim_drop, 0.93)
        args.near_merge_sim = max(args.near_merge_sim, 0.85)

    cleaned = dedupe_utterances(
        utts,
        window_sec=args.window_sec,
        sim_drop=args.sim_drop,
        near_merge_gap=args.near_merge_gap,
        near_merge_sim=args.near_merge_sim,
        cross_speaker_drop=(not args.no_cross_speaker_drop),
    )
    new_text = rebuild_text(cleaned)

    new_data = _write_back(data, cleaned, new_text, schema)
    out.write_text(json.dumps(new_data, ensure_ascii=False, indent=2), encoding="utf-8")

    dropped = max(0, len(utts) - len(cleaned))
    print(f"[OK] {len(utts)} → {len(cleaned)} utterances (−{dropped}). Sortie: {out}")

if __name__ == "__main__":
    main()
