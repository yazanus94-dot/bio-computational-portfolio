"""Population genetics toolkit: Wright-Fisher simulation, Hardy-Weinberg testing,
and comparison of simulated fixation probabilities with Kimura's diffusion result.

Run directly to reproduce the figures:  python popgen.py
"""
from __future__ import annotations

import os

import numpy as np
from scipy import stats

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ----------------------------------------------------------------------------
# Hardy-Weinberg equilibrium
# ----------------------------------------------------------------------------
def hardy_weinberg_test(n_AA: int, n_Aa: int, n_aa: int) -> dict:
    """Chi-square goodness-of-fit test for Hardy-Weinberg equilibrium.

    Degrees of freedom = 1 (3 genotype classes - 1 - 1 estimated allele frequency).
    """
    n = n_AA + n_Aa + n_aa
    if n <= 0:
        raise ValueError("Sample size must be positive.")
    p = (2 * n_AA + n_Aa) / (2 * n)
    q = 1.0 - p
    expected = np.array([p * p, 2 * p * q, q * q]) * n
    observed = np.array([n_AA, n_Aa, n_aa], dtype=float)
    if np.any(expected == 0):
        raise ValueError("Monomorphic sample: Hardy-Weinberg test is undefined.")
    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    p_value = float(stats.chi2.sf(chi2, df=1))
    return {"p": p, "q": q, "expected": expected, "chi2": chi2, "p_value": p_value}


# ----------------------------------------------------------------------------
# Wright-Fisher model
# ----------------------------------------------------------------------------
def wright_fisher(
    N: int,
    p0: float,
    generations: int,
    s: float = 0.0,
    h: float = 0.5,
    mu: float = 0.0,
    nu: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate allele frequency of A in a diploid Wright-Fisher population.

    Fitnesses: AA = 1+s, Aa = 1+h*s, aa = 1.  Mutation A->a at rate mu, a->A at rate nu.
    Selection acts first, then mutation, then binomial sampling of 2N gametes.
    Returns the frequency trajectory (length generations+1); stops early if absorbed
    (only when mu = nu = 0), padding with the absorbed value.
    """
    if not 0.0 <= p0 <= 1.0:
        raise ValueError("p0 must lie in [0, 1].")
    rng = rng or np.random.default_rng()
    traj = np.empty(generations + 1)
    p = p0
    traj[0] = p
    for t in range(1, generations + 1):
        q = 1.0 - p
        w_bar = p * p * (1 + s) + 2 * p * q * (1 + h * s) + q * q
        p_sel = (p * p * (1 + s) + p * q * (1 + h * s)) / w_bar
        p_mut = p_sel * (1 - mu) + (1 - p_sel) * nu
        p = rng.binomial(2 * N, p_mut) / (2 * N)
        traj[t] = p
        if mu == 0.0 and nu == 0.0 and (p == 0.0 or p == 1.0):
            traj[t:] = p
            break
    return traj


def simulate_fixation_probability(
    N: int, s: float, replicates: int, h: float = 0.5, seed: int = 0
) -> tuple[float, float]:
    """Estimate fixation probability of a single new mutant copy (p0 = 1/2N).

    Returns (estimate, binomial standard error).
    """
    rng = np.random.default_rng(seed)
    p0 = 1.0 / (2 * N)
    fixed = 0
    for _ in range(replicates):
        p = p0
        while 0.0 < p < 1.0:
            q = 1.0 - p
            w_bar = p * p * (1 + s) + 2 * p * q * (1 + h * s) + q * q
            p_sel = (p * p * (1 + s) + p * q * (1 + h * s)) / w_bar
            p = rng.binomial(2 * N, p_sel) / (2 * N)
        fixed += p == 1.0
    est = fixed / replicates
    return est, float(np.sqrt(est * (1 - est) / replicates))


def kimura_fixation_probability(N: int, s: float, p0: float | None = None, Ne: int | None = None) -> float:
    """Kimura (1962) diffusion approximation for additive selection (h = 1/2).

    Convention matches ``wright_fisher``: fitnesses AA = 1+s, Aa = 1+s/2, aa = 1, so the
    per-allele advantage is s/2.  With s_a = s/2:

        u(p) = (1 - exp(-4 Ne s_a p)) / (1 - exp(-4 Ne s_a)),  neutral limit u = p.
    """
    Ne = Ne or N
    p0 = 1.0 / (2 * N) if p0 is None else p0
    s_a = s / 2.0
    if abs(s_a) < 1e-12:
        return p0
    return float((1 - np.exp(-4 * Ne * s_a * p0)) / (1 - np.exp(-4 * Ne * s_a)))


def heterozygosity_decay(N: int, generations: int, loci: int = 500, seed: int = 1) -> np.ndarray:
    """Mean expected heterozygosity 2p(1-p) across independent neutral loci starting at p = 0.5."""
    rng = np.random.default_rng(seed)
    p = np.full(loci, 0.5)
    H = np.empty(generations + 1)
    H[0] = np.mean(2 * p * (1 - p))
    for t in range(1, generations + 1):
        p = rng.binomial(2 * N, p) / (2 * N)
        H[t] = np.mean(2 * p * (1 - p))
    return H


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def make_figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    # Fig 1: drift vs selection trajectories
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, s, title in zip(axes, [0.0, 0.02], ["Neutral drift (s = 0)", "Positive selection (s = 0.02)"]):
        for _ in range(20):
            ax.plot(wright_fisher(100, 0.1, 400, s=s, rng=rng), lw=0.9, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Generation")
    axes[0].set_ylabel("Allele frequency of A")
    fig.suptitle("Wright-Fisher trajectories, N = 100, p0 = 0.1")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "trajectories.png"), dpi=150)
    plt.close(fig)

    # Fig 2: heterozygosity decay vs theory (1 - 1/2N)^t
    fig, ax = plt.subplots(figsize=(6, 4))
    for N, c in [(25, "C0"), (100, "C1"), (400, "C2")]:
        gens = 300
        H = heterozygosity_decay(N, gens, loci=600)
        ax.plot(H / H[0], c=c, lw=1.2, label=f"N = {N} (simulated)")
        ax.plot((1 - 1 / (2 * N)) ** np.arange(gens + 1), c=c, ls="--", lw=1)
    ax.set_xlabel("Generation")
    ax.set_ylabel("H(t) / H(0)")
    ax.set_title("Loss of heterozygosity: simulation (solid) vs theory (dashed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "heterozygosity.png"), dpi=150)
    plt.close(fig)

    # Fig 3: fixation probability vs s
    N = 50
    s_vals = np.array([-0.02, -0.01, 0.0, 0.01, 0.02, 0.04])
    sims = [simulate_fixation_probability(N, s, 4000, seed=i) for i, s in enumerate(s_vals)]
    est = np.array([e for e, _ in sims])
    se = np.array([e for _, e in sims])
    s_grid = np.linspace(-0.025, 0.045, 200)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(s_grid, [kimura_fixation_probability(N, s) for s in s_grid], label="Kimura diffusion")
    ax.errorbar(s_vals, est, yerr=1.96 * se, fmt="o", capsize=3, label="Simulation (95% CI)")
    ax.set_yscale("log")
    ax.set_xlabel("Selection coefficient s")
    ax.set_ylabel("P(fixation) of a single new mutant")
    ax.set_title(f"Fixation probability, N = {N}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fixation_probability.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print("Hardy-Weinberg example (AA=298, Aa=489, aa=213):")
    res = hardy_weinberg_test(298, 489, 213)
    print(f"  p(A) = {res['p']:.3f}, chi2 = {res['chi2']:.3f}, p-value = {res['p_value']:.3f}")
    make_figures()
    print("Figures written to", FIG_DIR)
