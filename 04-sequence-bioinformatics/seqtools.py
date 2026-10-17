"""Sequence bioinformatics from scratch (no Biopython): composition, translation,
ORF finding, k-mer statistics and Needleman-Wunsch / Smith-Waterman alignment.

Run directly to reproduce the figures:  python seqtools.py
"""
from __future__ import annotations

import itertools
import os
from collections import Counter

import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

_BASES = "TCAG"
_AMINO = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE = {a + b + c: _AMINO[i] for i, (a, b, c) in enumerate(itertools.product(_BASES, repeat=3))}
_COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def _clean(seq: str) -> str:
    seq = seq.upper().replace("U", "T").replace(" ", "").replace("\n", "")
    bad = set(seq) - set("ACGTN")
    if bad:
        raise ValueError(f"Invalid nucleotide characters: {sorted(bad)}")
    return seq


# ----------------------------------------------------------------------------
# Basic sequence operations
# ----------------------------------------------------------------------------
def gc_content(seq: str) -> float:
    seq = _clean(seq)
    if not seq:
        raise ValueError("Empty sequence.")
    return (seq.count("G") + seq.count("C")) / len(seq)


def reverse_complement(seq: str) -> str:
    return _clean(seq).translate(_COMP)[::-1]


def translate(seq: str, to_stop: bool = False) -> str:
    seq = _clean(seq)
    prot = []
    for i in range(0, len(seq) - len(seq) % 3, 3):
        aa = CODON_TABLE.get(seq[i : i + 3], "X")
        if aa == "*" and to_stop:
            break
        prot.append(aa)
    return "".join(prot)


def find_orfs(seq: str, min_aa: int = 30) -> list[dict]:
    """Find ATG-initiated ORFs on both strands in all six frames.

    Returns dicts with strand, frame, 0-based start/end on the *forward* coordinates
    (end exclusive, includes the stop codon) and the protein sequence.
    """
    seq = _clean(seq)
    n = len(seq)
    orfs = []
    for strand, s in (("+", seq), ("-", reverse_complement(seq))):
        for frame in range(3):
            i = frame
            while i + 3 <= n:
                if s[i : i + 3] == "ATG":
                    j = i
                    while j + 3 <= n and CODON_TABLE[s[j : j + 3]] != "*":
                        j += 3
                    if j + 3 <= n:  # stop codon found
                        prot = translate(s[i:j])
                        if len(prot) >= min_aa:
                            a, b = (i, j + 3) if strand == "+" else (n - (j + 3), n - i)
                            orfs.append({"strand": strand, "frame": frame, "start": a, "end": b, "protein": prot})
                        i = j + 3
                        continue
                    break
                i += 3
    return sorted(orfs, key=lambda o: o["start"])


def kmer_counts(seq: str, k: int) -> Counter:
    seq = _clean(seq)
    if k < 1 or k > len(seq):
        raise ValueError("k must satisfy 1 <= k <= len(seq).")
    return Counter(seq[i : i + k] for i in range(len(seq) - k + 1) if "N" not in seq[i : i + k])


def shannon_entropy(seq: str, k: int = 1) -> float:
    c = kmer_counts(seq, k)
    p = np.array(list(c.values()), float)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def sliding_gc(seq: str, window: int = 100, step: int = 10) -> tuple[np.ndarray, np.ndarray]:
    seq = _clean(seq)
    pos = np.arange(0, len(seq) - window + 1, step)
    return pos + window // 2, np.array([gc_content(seq[p : p + window]) for p in pos])


# ----------------------------------------------------------------------------
# Pairwise alignment
# ----------------------------------------------------------------------------
def _score_matrix(a: str, b: str, match: int, mismatch: int, gap: int, local: bool):
    n, m = len(a), len(b)
    H = np.zeros((n + 1, m + 1), dtype=int)
    T = np.zeros((n + 1, m + 1), dtype="U1")  # traceback: D, U, L, or 0
    if not local:
        H[:, 0] = gap * np.arange(n + 1)
        H[0, :] = gap * np.arange(m + 1)
        T[1:, 0], T[0, 1:] = "U", "L"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = H[i - 1, j - 1] + (match if a[i - 1] == b[j - 1] else mismatch)
            up = H[i - 1, j] + gap
            left = H[i, j - 1] + gap
            best = max(diag, up, left, 0 if local else -10**9)
            H[i, j] = best
            T[i, j] = "0" if (local and best == 0) else ("D" if best == diag else "U" if best == up else "L")
    return H, T


def align(a: str, b: str, mode: str = "global", match: int = 1, mismatch: int = -1, gap: int = -2) -> dict:
    """Pairwise alignment with linear gap penalties.

    mode='global' -> Needleman-Wunsch; mode='local' -> Smith-Waterman.
    """
    if mode not in ("global", "local"):
        raise ValueError("mode must be 'global' or 'local'.")
    a, b = a.upper(), b.upper()
    local = mode == "local"
    H, T = _score_matrix(a, b, match, mismatch, gap, local)
    if local:
        i, j = np.unravel_index(np.argmax(H), H.shape)
    else:
        i, j = len(a), len(b)
    score = int(H[i, j])
    end_a, end_b = i, j
    out_a, out_b = [], []
    while i > 0 or j > 0:
        t = T[i, j]
        if local and (t == "0" or H[i, j] == 0):
            break
        if t == "D":
            out_a.append(a[i - 1]); out_b.append(b[j - 1]); i -= 1; j -= 1
        elif t == "U":
            out_a.append(a[i - 1]); out_b.append("-"); i -= 1
        else:
            out_a.append("-"); out_b.append(b[j - 1]); j -= 1
    aa, bb = "".join(reversed(out_a)), "".join(reversed(out_b))
    ident = sum(x == y for x, y in zip(aa, bb) if x != "-" and y != "-")
    return {"score": score, "aligned_a": aa, "aligned_b": bb, "identity": ident / max(len(aa), 1),
            "start_a": i, "end_a": end_a, "start_b": j, "end_b": end_b}


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("Sequences must have equal length.")
    return sum(x != y for x, y in zip(a, b))


def synthetic_genome(length=3000, seed=0, gc_island=(1200, 1600)) -> str:
    """Random sequence with an embedded GC-rich island and one planted ORF, for demos."""
    rng = np.random.default_rng(seed)
    seq = list(rng.choice(list("ACGT"), length, p=[0.32, 0.18, 0.18, 0.32]))
    a, b = gc_island
    seq[a:b] = rng.choice(list("ACGT"), b - a, p=[0.15, 0.35, 0.35, 0.15])
    orf = "ATG" + "GCT" * 60 + "TAA"
    seq[300 : 300 + len(orf)] = list(orf)
    return "".join(seq)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def make_figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    g = synthetic_genome()
    x, y = sliding_gc(g, 150, 10)
    orfs = find_orfs(g, min_aa=50)

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(x, y)
    axes[0].axhline(gc_content(g), c="gray", ls="--", lw=0.8, label="genome mean")
    axes[0].set_ylabel("GC fraction (150-bp window)")
    axes[0].set_title("Synthetic genome: GC island detection and ORFs")
    axes[0].legend()
    for o in orfs:
        axes[1].barh(0 if o["strand"] == "+" else 1, o["end"] - o["start"], left=o["start"], height=0.8)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["+", "-"])
    axes[1].set_xlabel("Position (bp)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "genome_scan.png"), dpi=150)
    plt.close(fig)

    # Alignment score heatmap for a short pair
    a, b = "GATTACAGATTACA", "GCATGCUAGATTA".replace("U", "T")
    H, _ = _score_matrix(a, b, 1, -1, -2, local=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(H, cmap="viridis")
    ax.set_xticks(range(len(b) + 1), ["-"] + list(b))
    ax.set_yticks(range(len(a) + 1), ["-"] + list(a))
    ax.set_title("Needleman-Wunsch DP matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dp_matrix.png"), dpi=150)
    plt.close(fig)

    print(f"Genome GC = {gc_content(g):.3f}; ORFs found (>=50 aa): {len(orfs)}")
    res = align("ACACACTA", "AGCACACA", mode="global")
    print(res["aligned_a"]); print(res["aligned_b"]); print("score", res["score"])


if __name__ == "__main__":
    make_figures()
