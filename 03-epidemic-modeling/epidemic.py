"""Compartmental epidemic modelling: SEIR dynamics, R0, final-size relation,
vaccination thresholds, parameter inference and a stochastic comparison.

Run directly to reproduce the figures:  python epidemic.py
"""
from __future__ import annotations

import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ----------------------------------------------------------------------------
# Deterministic SEIR (fractions of population)
# ----------------------------------------------------------------------------
def seir_rhs(t, y, beta, sigma, gamma):
    S, E, I, R = y
    inf = beta * S * I
    return [-inf, inf - sigma * E, sigma * E - gamma * I, gamma * I]


def simulate_seir(beta, sigma, gamma, i0=1e-4, days=300, vaccinated=0.0, n_points=None):
    """Integrate SEIR. ``vaccinated`` is the fraction immunised before the outbreak."""
    if not 0 <= vaccinated < 1:
        raise ValueError("vaccinated must be in [0, 1).")
    S0 = 1.0 - i0 - vaccinated
    y0 = [S0, 0.0, i0, vaccinated]
    t_eval = np.linspace(0, days, n_points or days * 4 + 1)
    sol = solve_ivp(seir_rhs, (0, days), y0, args=(beta, sigma, gamma), t_eval=t_eval,
                    rtol=1e-9, atol=1e-12, method="LSODA")
    return sol.t, sol.y


def r0(beta, gamma):
    """Basic reproduction number for SIR/SEIR with a single infectious stage."""
    return beta / gamma


def herd_immunity_threshold(R0):
    if R0 <= 1:
        return 0.0
    return 1.0 - 1.0 / R0


def final_size(R0, s0=1.0):
    """Solve the final-size relation  z = s0 (1 - exp(-R0 z))  for the attack rate z.

    Returns 0 when R0*s0 <= 1 (no major outbreak).
    """
    if R0 * s0 <= 1.0:
        return 0.0
    f = lambda z: z - s0 * (1 - np.exp(-R0 * z))  # noqa: E731
    return brentq(f, 1e-12, s0)


# ----------------------------------------------------------------------------
# Parameter inference from noisy incidence-like data
# ----------------------------------------------------------------------------
def fit_beta_gamma(t_obs, infectious_obs, sigma, i0, days):
    """Least-squares estimate of (beta, gamma) from observed infectious prevalence."""

    def resid(theta):
        beta, gamma = theta
        t, y = simulate_seir(beta, sigma, gamma, i0=i0, days=days, n_points=len(t_obs))
        return y[2] - infectious_obs

    sol = least_squares(resid, x0=[0.5, 0.2], bounds=([1e-3, 1e-3], [5, 2]))
    return sol.x


# ----------------------------------------------------------------------------
# Stochastic SIR (Gillespie) to illustrate extinction / small-outbreak risk
# ----------------------------------------------------------------------------
def gillespie_sir(N, beta, gamma, I0=1, rng=None):
    """Exact stochastic SIR. Returns final number ever infected (including initial)."""
    rng = rng or np.random.default_rng()
    S, I = N - I0, I0
    total = I0
    while I > 0:
        rate_inf = beta * S * I / N
        rate_rec = gamma * I
        if rng.random() < rate_inf / (rate_inf + rate_rec):
            S -= 1
            I += 1
            total += 1
        else:
            I -= 1
    return total


def major_outbreak_probability(R0, N=2000, replicates=400, seed=0, threshold=0.1):
    rng = np.random.default_rng(seed)
    big = sum(gillespie_sir(N, R0 * 0.25, 0.25, rng=rng) > threshold * N for _ in range(replicates))
    return big / replicates


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def make_figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    sigma, gamma = 1 / 3.0, 1 / 5.0

    # Fig 1: epidemic curves for increasing vaccination coverage
    fig, ax = plt.subplots(figsize=(7, 4))
    beta = 0.6  # R0 = 3
    for vax in [0.0, 0.3, 0.5, 0.7]:
        t, y = simulate_seir(beta, sigma, gamma, vaccinated=vax, days=250)
        ax.plot(t, y[2], label=f"{int(vax * 100)}% vaccinated")
    ax.set_xlabel("Day")
    ax.set_ylabel("Infectious fraction")
    ax.set_title(f"SEIR, R0 = {r0(beta, gamma):.1f} - HIT = {herd_immunity_threshold(3):.0%}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "seir_vaccination.png"), dpi=150)
    plt.close(fig)

    # Fig 2: final size vs R0, simulation points on theory
    R0s = np.linspace(0.5, 5, 200)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(R0s, [final_size(x) for x in R0s], label="Final-size relation")
    sim_R0 = np.array([1.2, 1.5, 2, 3, 4, 5])
    sim_z = []
    for x in sim_R0:
        _, y = simulate_seir(x * gamma, sigma, gamma, i0=1e-6, days=1500, n_points=3000)
        sim_z.append(y[3][-1])
    ax.plot(sim_R0, sim_z, "ko", label="ODE simulation")
    ax.set_xlabel("R0")
    ax.set_ylabel("Final attack rate")
    ax.legend()
    ax.set_title("Attack rate vs R0")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "final_size.png"), dpi=150)
    plt.close(fig)

    # Fig 3: parameter recovery from noisy data
    rng = np.random.default_rng(7)
    t_obs, y = simulate_seir(0.55, sigma, 0.2, i0=1e-3, days=120, n_points=121)
    noisy = np.clip(y[2] * (1 + 0.08 * rng.standard_normal(len(t_obs))), 0, None)
    est = fit_beta_gamma(t_obs, noisy, sigma, 1e-3, 120)
    _, yfit = simulate_seir(est[0], sigma, est[1], i0=1e-3, days=120, n_points=121)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_obs, noisy, "k.", label="Noisy observations")
    ax.plot(t_obs, yfit[2], "C3-", label=f"Fit: beta={est[0]:.3f}, gamma={est[1]:.3f}")
    ax.set_xlabel("Day")
    ax.set_ylabel("Infectious fraction")
    ax.set_title("Parameter inference (truth: beta=0.55, gamma=0.20)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "inference.png"), dpi=150)
    plt.close(fig)

    print(f"Recovered beta={est[0]:.3f}, gamma={est[1]:.3f}, R0={est[0] / est[1]:.2f} (true 2.75)")
    print(f"P(major outbreak) R0=2: sim={major_outbreak_probability(2.0):.2f}, theory={1 - 1 / 2:.2f}")


if __name__ == "__main__":
    make_figures()
