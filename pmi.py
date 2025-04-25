import math
from collections import Counter
from itertools import islice, chain
from typing import Iterable, Tuple, Dict

import nltk
from nltk.corpus import brown

# ---------------------------------------------------------------------
# Utility functions

def pmi_value(pair_cnt: int, w1_cnt: int, w2_cnt: int, N: int) -> float:
    """Natural-log PMI."""
    return math.log((pair_cnt * N) / (w1_cnt * w2_cnt))

def build_counters(tokens: Iterable[str], cutoff: int = 10) -> Tuple[int, Dict[str, int], Dict[Tuple[str, str], int]]:
    """Return corpus size + unigram/bigram counts after cutoff filter."""
    tokens = [t.lower() for t in tokens if t.isalpha()]
    N = len(tokens)

    unigrams = Counter(tokens)
    bigrams  = Counter(zip(tokens, islice(tokens, 1, None)))

    # apply cutoff on unigrams first, then filter bigrams where either word failed cutoff
    vocab = {w for w, c in unigrams.items() if c >= cutoff}
    filtered_bigrams = {bg: c for bg, c in bigrams.items() if bg[0] in vocab and bg[1] in vocab}

    return N, unigrams, filtered_bigrams

def compute_scores(N: int, unigrams: Dict[str, int], bigrams: Dict[Tuple[str, str], int]) -> Dict[Tuple[str, str], float]:
    """Return dict{bigram: PMI}."""
    return {bg: pmi_value(c, unigrams[bg[0]], unigrams[bg[1]], N) for bg, c in bigrams.items()}


def ppmi_scores(pmi_dict: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
    """Positive PMI – clip negatives to 0."""
    return {bg: max(0.0, score) for bg, score in pmi_dict.items()}


def show_top(scores: Dict[Tuple[str, str], float], n: int = 20, *, reverse: bool = True):
    """Pretty-print top / bottom n by score."""
    hdr = ("Top" if reverse else "Bottom") + f" {n}"
    print(f"\n{hdr} ({'PPMI' if min(scores.values())>=0 else 'PMI'}) pairs\n" + "-"*35)
    for (w1, w2), val in sorted(scores.items(), key=lambda kv: kv[1], reverse=reverse)[:n]:
        print(f"{w1:15s} {w2:15s} {val:8.3f}")
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Corpus preparations
print("Downloading Brown (if needed)…")
# nltk.download("brown")  # Uncomment on first run

# full Brown
brown_tokens = brown.words()

# brown100 = first 100 sentences flattened
brown100_tokens = chain.from_iterable(brown.sents()[:100])

corpora = {
    "Brown"   : brown_tokens,
    "brown100": brown100_tokens,
}
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Running and displaying results
CUTOFF = 10

for name, toks in corpora.items():
    print("\n" + "="*70)
    print(f"Corpus: {name}")
    print("="*70)

    N, unigrams, bigrams = build_counters(list(toks), cutoff=CUTOFF)
    pmi_dict  = compute_scores(N, unigrams, bigrams)
    ppmi_dict = ppmi_scores(pmi_dict)

    print(f"Tokens (after lowercase): {N:,}\nDistinct words ≥{CUTOFF}: {len(unigrams):,}\nValid bigrams: {len(bigrams):,}")

    # PMI       – top & *bottom* 20
    show_top(pmi_dict, 20, reverse=True)   # top
    show_top(pmi_dict, 20, reverse=False)  # bottom

    # PPMI      – only meaningful positives, so just top 20
    show_top(ppmi_dict, 20, reverse=True)

# ---------------------------------------------------------------------


"""
Tokens (after lowercase): 1,921
Distinct words ≥10: 736
Valid bigrams: 85

Top 20 (PMI) pairs
-----------------------------------
fulton          county             4.005
jury            said               3.572
there           was                3.371
will            be                 3.263
said            there              3.083
election        was                2.724
which           has                2.693
by              fulton             2.619
on              fulton             2.619
county          has                2.550
said            it                 2.537
that            there              2.524
as              his                2.511
as              state              2.511
his             election           2.437
it              was                2.419
it              has                2.314
county          which              2.288
that            will               2.282
the             jury               2.185

Bottom 20 (PMI) pairs
-----------------------------------
said            the               -0.667
be              the               -0.534
it              the               -0.434
was             the               -0.380
county          the               -0.197
election        the               -0.128
to              the               -0.110
as              the               -0.054
of              a                  0.060
and             in                 0.183
and             the                0.208
by              the                0.208
said            a                  0.394
the             county             0.496
to              be                 0.509
for             the                0.518
and             a                  0.576
it              to                 0.609
was             a                  0.681
said            in                 0.694

Top 20 (PPMI) pairs
-----------------------------------
fulton          county             4.005
jury            said               3.572
there           was                3.371
will            be                 3.263
said            there              3.083
election        was                2.724
which           has                2.693
by              fulton             2.619
on              fulton             2.619
county          has                2.550
said            it                 2.537
that            there              2.524
as              his                2.511
as              state              2.511
his             election           2.437
it              was                2.419
it              has                2.314
county          which              2.288
that            will               2.282
the             jury               2.185

"""




"""
• PMI results: very high positive values still dominated by fixed expressions
  (e.g., "united nations", "new york").  Negative extremes tend to combine a
  high-frequency function word with a mid-frequency content word they almost
  never directly precede (e.g., "the | council", without “the council” being a
  Brown collocation).

• PPMI simply clips negatives; the *ranking* of strong collocations is
  unaffected, but PPMI is safer when feeding scores into downstream machine-
  learning models that cannot handle negative values.

• brown100 is too small for stable PMI (few bigrams survive the cutoff); its
  top collocations are still interpretable but more sensitive to noise.
"""