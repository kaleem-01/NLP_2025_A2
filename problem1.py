
"""
problem1_all.py
===============

Comprehensive solution for A2.1 Text Processing & Zipf’s Law

Features
--------
* Loads Brown corpus (NLTK).
* Computes:
    1. Unique‑word frequency lists (corpus + two genres).
    2. Corpus statistics (tokens, types, words, etc.).
    3. Ten most frequent POS tags (full corpus).
* Plots:
    • Linear rank‑frequency curve.
    • Log–log rank‑frequency curve.
    • Log–log curve **with power‑law fit** (using `powerlaw` library).
* Saves:
    - linear_plot.png
    - loglog_plot.png
    - rankloglog_powerlaw.png
    - corpus_stats.txt   (pretty table of all statistics)
    - pos_top10.txt      (tag → freq)
    - powerlaw_stats.txt (α, γ, C for each dataset)

Edit the `GENRES` tuple to choose any two Brown genres.

Requirements
------------
```
pip install nltk powerlaw matplotlib
```
(The script auto‑downloads required NLTK data at first run.)

Run
---
```
python problem1_all.py
```
"""

import nltk
from nltk.corpus import brown
from nltk import pos_tag, sent_tokenize, WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
import os
import textwrap

# ----------------------- CONFIGURATION --------------------------------
GENRES = ('news', 'romance')   # choose any two Brown genres
TOP_RANK = 20000               # ranks to display on plots
# ----------------------------------------------------------------------

# Ensure resources
for pkg in ('brown', 'punkt', 'averaged_perceptron_tagger',
            'wordnet', 'omw-1.4'):
    nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def tokens_for(genres=None):
    toks = brown.words(categories=genres) if genres else brown.words()
    return [w.lower() for w in toks]

def corpus_stats(tokens):
    types = set(tokens)
    words = [t for t in tokens if t.isalpha()]
    sents = sent_tokenize(' '.join(tokens))

    return {
        'tokens': len(tokens),
        'types': len(types),
        'words': len(words),
        'avg_words_per_sentence': len(words)/len(sents),
        'avg_word_length': sum(len(w) for w in words)/len(words),
        'lemmas': len({lemmatizer.lemmatize(w) for w in words})
    }

def pos_top(tokens, n=10):
    tags = [tag for _, tag in pos_tag(tokens)]
    return Counter(tags).most_common(n)

def write_stats_file(stats_dict, path):
    with open(path, 'w') as f:
        for name, st in stats_dict.items():
            f.write(f"\n=== {name} ===\n")
            for k, v in st.items():
                f.write(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
                f.write("\n")

def save_pos_file(pos_list, path):
    with open(path, 'w') as f:
        for tag, freq in pos_list:
            f.write(f"{tag}: {freq}\n")

def rank_freq(tokens):
    return [freq for _, freq in Counter(tokens).most_common()]

def fit_powerlaw(counts):
    fit = powerlaw.Fit(counts, xmin=1, discrete=True, estimate_discrete=True)
    alpha = fit.power_law.alpha
    gamma = 1/(alpha-1) if alpha > 1 else np.nan
    return fit, alpha, gamma

def plot_rank_curves(datasets, labels):
    # Linear
    plt.figure()
    for data, lbl in zip(datasets, labels):
        freqs = rank_freq(data)
        ranks = np.arange(1, len(freqs)+1)
        plt.plot(ranks, freqs, label=lbl)
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.ylim(0, 200)       # show first 1000 frequencies	
    plt.title('Rank–Frequency (Linear)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/linear_plot.png', dpi=300)

    # Log–log
    plt.figure()
    for data, lbl in zip(datasets, labels):
        freqs = rank_freq(data)
        ranks = np.arange(1, len(freqs)+1)
        plt.loglog(ranks, freqs, label=lbl)
    plt.xlabel('Rank (log)')
    plt.ylabel('Frequency (log)')
    plt.title('Rank–Frequency (Log–Log)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('resultsloglog_plot.png', dpi=300)

def plot_rank_with_powerlaw(datasets, labels, top_rank=TOP_RANK):
    plt.figure()
    stats_lines = []
    for data, lbl in zip(datasets, labels):
        freqs = rank_freq(data)
        ranks = np.arange(1, len(freqs)+1)
        # empirical
        plt.loglog(ranks[:top_rank], freqs[:top_rank], label=lbl, alpha=0.6)
        # powerlaw fit
        counts = list(Counter(data).values())
        _, alpha, gamma = fit_powerlaw(counts)
        C = freqs[0]  # scale from most frequent word
        fit_freqs = C * ranks[:top_rank] ** (-gamma)
        plt.loglog(ranks[:top_rank], fit_freqs, linestyle='--')
        stats_lines.append(f"{lbl}: alpha={alpha:.3f}, gamma={gamma:.3f}, C={C}")
    plt.xlabel('Rank (log)')
    plt.ylabel('Frequency (log)')
    plt.title('Rank–Frequency with Power‑Law Fit')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/rankloglog_powerlaw.png', dpi=300)

    with open('results/powerlaw_stats.txt', 'w') as f:
        f.write('\n'.join(stats_lines))

def main():

    corpora = {'Brown corpus': tokens_for()}
    for g in GENRES:
        corpora[f"{g} genre"] = tokens_for(g)

    # Stats
    stats = {name: corpus_stats(toks) for name, toks in corpora.items()}
    write_stats_file(stats, 'corpus_stats.txt')

    # POS
    pos_top10 = pos_top(corpora['Brown corpus'])
    save_pos_file(pos_top10, 'pos_top10.txt')

    # Plots
    datasets = list(corpora.values())
    labels   = list(corpora.keys())
    plot_rank_curves(datasets, labels)
    plot_rank_with_powerlaw(datasets, labels)

    print("""        Done!
        Plots saved: linear_plot.png | loglog_plot.png | rankloglog_powerlaw.png
        Text files : corpus_stats.txt | pos_top10.txt | powerlaw_stats.txt
    """)

if __name__ == '__main__':
    main()
