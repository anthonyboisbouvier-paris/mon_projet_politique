%%bash
set -e
cat <<'PY' > /content/clean_json_v2.py
# -*- coding: utf-8 -*-
import json, re, unicodedata, argparse
from difflib import SequenceMatcher
from pathlib import Path

SENT_SPLIT = re.compile(r'(?<=[\.\?\!\u2026;:])\s+|[\n\r]+')  # . ? ! … ; : ou sauts de ligne
WS = re.compile(r'\s+')
PUNCT = re.compile(r'[^\w]+', re.UNICODE)

def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalize_for_match(s: str) -> str:
    s = strip_accents(s.lower())
    s = PUNCT.sub(' ', s)
    return WS.sub(' ', s).strip()

def sent_tokenize(text: str):
    parts = [p.strip() for p in SENT_SPLIT.split(text) if p and p.strip()]
    # Si aucun séparateur n’a fonctionné, on retombe sur une découpe douce par virgules longues
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r',(?!\d)', text) if p.strip()]
    return parts

def tokens(text: str):
    return [t for t in re.split(r'\s+', text) if t]

def equal_norm(a: str, b: str) -> bool:
    return normalize_for_match(a) == normalize_for_match(b)

def collapse_adjacent_repeats(text: str, max_ngram: int = 20) -> str:
    """Supprime A A (xN) où A peut être un n-gramme long (jusqu’à max_ngram tokens)."""
    toks = tokens(text)
    out = []
    i = 0
    while i < len(toks):
        # Cherche le plus long n-gramme qui se répète immédiatement
        kmax = min(max_ngram, (len(toks) - i) // 2)
        chosen_k = 0
        for k in range(kmax, 0, -1):
            left = ' '.join(toks[i:i+k])
            right = ' '.join(toks[i+k:i+2*k])
            if normalize_for_match(left) == normalize_for_match(right):
                chosen_k = k
                break
        if chosen_k:
            # Garde une seule occurrence, saute toutes les répétitions suivantes
            out.extend(toks[i:i+chosen_k])
            i += 2*chosen_k
            # Cas A A A… : saute les occurrences suivantes
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
        # pour les petites phrases, on retombe sur des 2-grammes / 1-grammes
        n = 2 if len(w) >= 2 else 1
    return {' '.join(w[i:i+n]) for i in range(0, max(0, len(w)-n+1))}

def jaccard(a, b) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))

def is_near_duplicate(s: str, recent_norms, sim_drop: float, seq_ratio: float, ngram_n: int):
    # Jaccard sur n-grammes
    A = ngram_set(s, n=ngram_n)
    for r in recent_norms:
        if jaccard(A, r['ngrams']) >= sim_drop:
            return True
        # ratio de séquence (Levenshtein approximé)
        if SequenceMatcher(None, r['norm'], normalize_for_match(s)).ratio() >= seq_ratio:
            return True
    return False

def clean_utterance_text(raw: str, recent_norms, sim_drop: float, seq_ratio: float,
                         ngram_n: int, max_ngram_repeat: int, min_clause_chars: int):
    # 1) collapse de répétitions contiguës à base de n-grammes
    collapsed = collapse_adjacent_repeats(raw, max_ngram=max_ngram_repeat)
    # 2) split en phrases/propositions
    clauses = sent_tokenize(collapsed)
    cleaned = []
    local_recent = []  # tampon intra-énoncé
    for c in clauses:
        c = c.strip()
        if not c:
            continue
        if len(c) < min_clause_chars:
            # petites miettes : on tente la fusion avec la clause précédente si possible
            if cleaned:
                merged = cleaned[-1] + ' ' + c
                # remplace la dernière clause, puis continue
                cleaned[-1] = merged.strip()
                continue
            else:
                # sinon on stocke quand même
                cleaned.append(c)
                continue
        # dédoublonnage intra-énoncé (local) puis global (recent_norms)
        if is_near_duplicate(c, [{'norm': normalize_for_match(x),
                                  'ngrams': ngram_set(x, n=ngram_n)} for x in local_recent],
                             sim_drop, seq_ratio, ngram_n):
            continue
        if is_near_duplicate(c, recent_norms, sim_drop, seq_ratio, ngram_n):
            continue
        cleaned.append(c)
        local_recent.append(c)
        recent_norms.append({'norm': normalize_for_match(c),
                             'ngrams': ngram_set(c, n=ngram_n)})
        # on bride la taille du tampon global
        if len(recent_norms) > 800:
            recent_norms[:] = recent_norms[-800:]
    # 3) recolle avec des espaces propres
    out = ' '.join(cleaned).strip()
    # petit nettoyage d’espaces avant ponctuation
    out = re.sub(r'\s+([,;:\.\?\!\u2026])', r'\1', out)
    return out

def process(data, sim_drop: float, seq_ratio: float, ngram_n: int,
            max_ngram_repeat: int, min_clause_chars: int):
    utts = data.get('utterances', [])
    recent_norms = []  # tampon global de phrases récentes (tous locuteurs)
    new_utts = []
    for u in utts:
        t = u.get('text', '')
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
    # reconstruit le champ "text"
    data['text'] = ' '.join(u['text'] for u in new_utts).strip()
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--input', required=True)
    ap.add_argument('-o','--output', required=True)
    # Seuils très agressifs par défaut :
    ap.add_argument('--sim-drop', type=float, default=0.94, help='Seuil Jaccard n-gram pour drop')
    ap.add_argument('--seq-ratio', type=float, default=0.94, help='Seuil difflib ratio pour drop')
    ap.add_argument('--ngram-n', type=int, default=3, help='n pour n-grammes (Jaccard)')
    ap.add_argument('--max-ngram-repeat', type=int, default=20, help='Longueur max des n-grammes à effondrer')
    ap.add_argument('--min-clause-chars', type=int, default=20, help='Taille mini d’une clause')
    args = ap.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned = process(
        data,
        sim_drop=args.sim_drop,
        seq_ratio=args.seq_ratio,
        ngram_n=args.ngram_n,
        max_ngram_repeat=args.max_ngram_repeat,
        min_clause_chars=args.min_clause_chars
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
PY
echo "OK"

