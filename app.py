# app.py
# Nettoyage agressif des répétitions dans un JSON {meta, utterances[], text}
# Post-traitement sûr : ne modifie pas l'étape d'ASR/diarisation.
# Usage typique (Colab) :
#   python /content/app.py \
#       --input-json  /content/outputs/json/jancovici_full.json \
#       --output-json /content/outputs/json/jancovici_full.cleaned.json \
#       --sim-drop 0.95 --seq-ratio 0.95 --ngram-n 3 --max-ngram-repeat 24 --min-clause-chars 18

import argparse
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Any

# ----------------------------
# Utils de normalisation/matching
# ----------------------------
SENT_SPLIT = re.compile(r'(?<=[\.\?\!\u2026;:])\s+|[\n\r]+')  # fins de phrases usuelles ou sauts de ligne
WS = re.compile(r'\s+')
PUNCT = re.compile(r'[^\w]+', re.UNICODE)

def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalize_for_match(s: str) -> str:
    s = strip_accents(s.lower())
    s = PUNCT.sub(' ', s)
    return WS.sub(' ', s).strip()

def sent_tokenize(text: str) -> List[str]:
    parts = [p.strip() for p in SENT_SPLIT.split(text) if p and p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r',(?!\d)', text) if p.strip()]
    return parts

def tokens(text: str) -> List[str]:
    return [t for t in re.split(r'\s+', text) if t]

def collapse_adjacent_repeats(text: str, max_ngram: int = 20) -> str:
    """Supprime A A A... où A est un n-gramme contigu (jusqu'à max_ngram tokens)."""
    toks = tokens(text)
    out = []
    i = 0
    while i < len(toks):
        kmax = min(max_ngram, (len(toks) - i) // 2)
        chosen_k = 0
        for k in range(kmax, 0, -1):
            left = ' '.join(toks[i:i+k])
            right = ' '.join(toks[i+k:i+2*k])
            if normalize_for_match(left) == normalize_for_match(right):
                chosen_k = k
                break
        if chosen_k:
            out.extend(toks[i:i+chosen_k])
            i += 2 * chosen_k
            # A A A... : saute les occurrences suivantes
            while i + chosen_k <= len(toks):
                seg = ' '.join(toks[i:i+chosen_k])
                if normalize_for_match(seg) == normalize_for_match(' '.join(out[-chosen_k:])):
                    i += chosen_k
                else:
                    break
        else:
            out.append(toks[i])
            i += 1
    return ' '.join(out)

def ngram_set(s: str, n: int = 3):
    w = normalize_for_match(s).split()
    if len(w) < n:
        n = 2 if len(w) >= 2 else 1
    return {' '.join(w[i:i+n]) for i in range(0, max(0, len(w)-n+1))}

def jaccard(a, b) -> float:
    if not a or not b: return 0.0
    inter = len(a & b)
    return 0.0 if inter == 0 else inter / float(len(a | b))

def is_near_duplicate(s: str, recent_norms: List[Dict[str, Any]], sim_drop: float, seq_ratio: float, ngram_n: int) -> bool:
    A = ngram_set(s, n=ngram_n)
    s_norm = normalize_for_match(s)
    for r in recent_norms:
        if jaccard(A, r['ngrams']) >= sim_drop:
            return True
        if SequenceMatcher(None, r['norm'], s_norm).ratio() >= seq_ratio:
            return True
    return False

def clean_utterance_text(raw: str, recent_norms: List[Dict[str, Any]], sim_drop: float, seq_ratio: float,
                         ngram_n: int, max_ngram_repeat: int, min_clause_chars: int) -> str:
    # 1) écrase les répétitions contiguës
    collapsed = collapse_adjacent_repeats(raw, max_ngram=max_ngram_repeat)
    # 2) découpe en phrases/propositions
    clauses = sent_tokenize(collapsed)
    cleaned: List[str] = []
    local_recent: List[str] = []

    for c in clauses:
        c = c.strip()
        if not c:
            continue
        if len(c) < min_clause_chars:
            if cleaned:
                cleaned[-1] = (cleaned[-1] + ' ' + c).strip()
                continue
            else:
                cleaned.append(c)
                continue

        # dédoublonnage intra-énoncé
        local_pack = [{'norm': normalize_for_match(x), 'ngrams': ngram_set(x, n=ngram_n)} for x in local_recent]
        if is_near_duplicate(c, local_pack, sim_drop, seq_ratio, ngram_n):
            continue
        # dédoublonnage global (tous locuteurs)
        if is_near_duplicate(c, recent_norms, sim_drop, seq_ratio, ngram_n):
            continue

        cleaned.append(c)
        local_recent.append(c)
        recent_norms.append({'norm': normalize_for_match(c), 'ngrams': ngram_set(c, n=ngram_n)})
        if len(recent_norms) > 800:
            del recent_norms[:-800]

    out = ' '.join(cleaned).strip()
    out = re.sub(r'\s+([,;:\.\?\!\u2026])', r'\1', out)
    return out

def aggressive_clean(data: Dict[str, Any],
                     sim_drop: float = 0.95,
                     seq_ratio: float = 0.95,
                     ngram_n: int = 3,
                     max_ngram_repeat: int = 24,
                     min_clause_chars: int = 18) -> Dict[str, Any]:
    """Nettoie data['utterances'][*]['text'] + reconstruit data['text']."""
    utts = data.get('utterances', []) or []
    recent_norms: List[Dict[str, Any]] = []
    new_utts = []
    for u in utts:
        t = u.get('text', '') or ''
        cleaned = clean_utterance_text(
            t, recent_norms,
            sim_drop=sim_drop,
            seq_ratio=seq_ratio,
            ngram_n=ngram_n,
            max_ngram_repeat=max_ngram_repeat,
            min_clause_chars=min_clause_chars
        )
        if cleaned:
            nu = dict(u)
            nu['text'] = cleaned
            new_utts.append(nu)

    data['utterances'] = new_utts
    data['text'] = ' '.join(u['text'] for u in new_utts).strip()

    # On garde les métadonnées existantes et on trace les options de nettoyage
    opts = data.get('options', {})
    opts['cleaning'] = {
        'sim_drop': sim_drop,
        'seq_ratio': seq_ratio,
        'ngram_n': ngram_n,
        'max_ngram_repeat': max_ngram_repeat,
        'min_clause_chars': min_clause_chars,
        'aggressive': True
    }
    data['options'] = opts
    return data

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Post-traitement agressif pour supprimer les répétitions et reconstituer un texte propre."
    )
    ap.add_argument('--input-json', required=True, help='Chemin du JSON brut (sortie de ton pipeline ASR/diarisation).')
    ap.add_argument('--output-json', required=False, help='Chemin du JSON nettoyé (défaut: *.cleaned.json à côté).')

    # Seuils/paramètres (par défaut agressifs)
    ap.add_argument('--sim-drop', type=float, default=0.95, help='Seuil Jaccard n-gram pour drop.')
    ap.add_argument('--seq-ratio', type=float, default=0.95, help='Seuil difflib SequenceMatcher pour drop.')
    ap.add_argument('--ngram-n', type=int, default=3, help='n pour les n-grammes du Jaccard.')
    ap.add_argument('--max-ngram-repeat', type=int, default=24, help='Longueur max des n-grammes contigus à effondrer.')
    ap.add_argument('--min-clause-chars', type=int, default=18, help='Longueur minimale d’une clause conservée.')
    return ap.parse_args()

def main():
    args = parse_args()
    in_path = Path(args.input_json)
    if not in_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {in_path}")

    out_path = Path(args.output_json) if args.output_json else in_path.with_suffix('').with_name(in_path.stem + '.cleaned.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Conserve meta en-tête si présente ; sinon ne crée rien de plus.
    # (Le nettoyeur ne touche pas data['meta'].)

    cleaned = aggressive_clean(
        data,
        sim_drop=args.sim_drop,
        seq_ratio=args.seq_ratio,
        ngram_n=args.ngram_n,
        max_ngram_repeat=args.max_ngram_repeat,
        min_clause_chars=args.min_clause_chars
    )

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"[OK] JSON nettoyé écrit dans : {out_path}")

if __name__ == '__main__':
    main()
