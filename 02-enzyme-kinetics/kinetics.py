"""Enzyme kinetics: Michaelis-Menten fitting, inhibition models, model selection
and bootstrap confidence intervals.

Run directly to reproduce the figures:  python kinetics.py
"""
from __future__ import annotations

import os

import numpy as np
from scipy.optimize import curve_fit

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ----------------------------------------------------------------------------
# Rate laws
# ----------------------------------------------------------------------------
def michaelis_menten(S, Vmax, Km):
    return Vmax * S / (Km + S)


def competitive(S, I, Vmax, Km, Ki):
    return Vmax * S / (Km * (1 + I / Ki) + S)


def uncompetitive(S, I, Vmax, Km, Ki):
    return Vmax * S / (Km + S * (1 + I / Ki))


def noncompetitive(S, I, Vmax, Km, Ki):
    return Vmax * S / ((1 + I / Ki) * (Km + S))


def substrate_inhibition(S, Vmax, Km, Ksi):
    return Vmax * S / (Km + S * (1 + S / Ksi))


def hill(S, Vmax, K, n):
    return Vmax * S**n / (K**n + S**n)


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------
def fit_michaelis_menten(S, v, sigma=None):
    """Non-linear least squares fit. Returns dict with estimates, SEs and covariance."""
    S, v = np.asarray(S, float), np.asarray(v, float)
    if len(S) < 3:
        raise ValueError("Need at least 3 data points.")
    p0 = [v.max() * 1.2, np.median(S)]
    popt, pcov = curve_fit(michaelis_menten, S, v, p0=p0, sigma=sigma, bounds=(0, np.inf))
    se = np.sqrt(np.diag(pcov))
    return {"Vmax": popt[0], "Km": popt[1], "se_Vmax": se[0], "se_Km": se[1], "cov": pcov}


def lineweaver_burk_fit(S, v):
    """Classic double-reciprocal linear fit (biased by error weighting; shown for comparison)."""
    S, v = np.asarray(S, float), np.asarray(v, float)
    slope, intercept = np.polyfit(1 / S, 1 / v, 1)
    Vmax = 1 / intercept
    Km = slope * Vmax
    return {"Vmax": Vmax, "Km": Km}


def bootstrap_ci(S, v, n_boot=2000, alpha=0.05, seed=0):
    """Residual bootstrap percentile CIs for (Vmax, Km)."""
    rng = np.random.default_rng(seed)
    S, v = np.asarray(S, float), np.asarray(v, float)
    fit = fit_michaelis_menten(S, v)
    pred = michaelis_menten(S, fit["Vmax"], fit["Km"])
    resid = v - pred
    draws = []
    for _ in range(n_boot):
        v_star = pred + rng.choice(resid, size=len(resid), replace=True)
        try:
            f = fit_michaelis_menten(S, np.clip(v_star, 1e-9, None))
            draws.append((f["Vmax"], f["Km"]))
        except RuntimeError:
            continue
    draws = np.array(draws)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return {"Vmax": (lo[0], hi[0]), "Km": (lo[1], hi[1]), "n_valid": len(draws)}


def aic(y, y_pred, k):
    """Gaussian-error AIC with k fitted parameters (small-sample corrected AICc returned)."""
    n = len(y)
    rss = float(np.sum((np.asarray(y) - np.asarray(y_pred)) ** 2))
    a = n * np.log(rss / n) + 2 * k
    return a + (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else a


def fit_inhibition(S, I, v):
    """Fit competitive / uncompetitive / non-competitive models globally and rank by AICc."""
    S, I, v = (np.asarray(x, float) for x in (S, I, v))
    models = {"competitive": competitive, "uncompetitive": uncompetitive, "noncompetitive": noncompetitive}
    out = {}
    for name, f in models.items():
        popt, _ = curve_fit(lambda X, Vm, Km, Ki: f(X[0], X[1], Vm, Km, Ki), np.vstack([S, I]), v,
                            p0=[v.max() * 1.2, np.median(S), max(np.median(I[I > 0]), 1e-3)],
                            bounds=(1e-9, np.inf), maxfev=20000)
        pred = f(S, I, *popt)
        out[name] = {"params": dict(zip(["Vmax", "Km", "Ki"], popt)), "AICc": aic(v, pred, 3)}
    out["best"] = min(models, key=lambda m: out[m]["AICc"])
    return out


# ----------------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------------
def synthetic_mm(Vmax=100.0, Km=25.0, noise=0.05, seed=0, n=10):
    rng = np.random.default_rng(seed)
    S = np.geomspace(2, 400, n)
    v = michaelis_menten(S, Vmax, Km) * (1 + noise * rng.standard_normal(n))
    return S, v


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def make_figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    true_Vmax, true_Km = 100.0, 25.0
    S, v = synthetic_mm(true_Vmax, true_Km, noise=0.06, seed=3, n=12)
    nl = fit_michaelis_menten(S, v)
    lb = lineweaver_burk_fit(S, v)
    ci = bootstrap_ci(S, v, n_boot=1500)

    Sg = np.linspace(0, 420, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(S, v, "ko", label="Data")
    axes[0].plot(Sg, michaelis_menten(Sg, true_Vmax, true_Km), "g:", label="Truth")
    axes[0].plot(Sg, michaelis_menten(Sg, nl["Vmax"], nl["Km"]), "C0-",
                 label=f"Non-linear fit (Km={nl['Km']:.1f})")
    axes[0].plot(Sg, michaelis_menten(Sg, lb["Vmax"], lb["Km"]), "C3--",
                 label=f"Lineweaver-Burk (Km={lb['Km']:.1f})")
    axes[0].set_xlabel("[S]")
    axes[0].set_ylabel("v")
    axes[0].set_title("Michaelis-Menten fit")
    axes[0].legend(fontsize=8)

    axes[1].plot(1 / S, 1 / v, "ko")
    x = np.linspace(0, (1 / S).max(), 50)
    axes[1].plot(x, x * lb["Km"] / lb["Vmax"] + 1 / lb["Vmax"], "C3--")
    axes[1].set_xlabel("1/[S]")
    axes[1].set_ylabel("1/v")
    axes[1].set_title("Double-reciprocal plot")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mm_fit.png"), dpi=150)
    plt.close(fig)

    # Inhibition global fit
    rng = np.random.default_rng(8)
    Sg_, Ig_ = np.meshgrid(np.geomspace(5, 300, 8), [0, 10, 30])
    Sd, Id = Sg_.ravel(), Ig_.ravel()
    vd = competitive(Sd, Id, 100, 25, 12) * (1 + 0.04 * rng.standard_normal(len(Sd)))
    res = fit_inhibition(Sd, Id, vd)
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.linspace(1, 320, 300)
    for I_, c in zip([0, 10, 30], ["C0", "C1", "C2"]):
        m = Id == I_
        ax.plot(Sd[m], vd[m], "o", c=c)
        p = res[res["best"]]["params"]
        f = {"competitive": competitive, "uncompetitive": uncompetitive, "noncompetitive": noncompetitive}[res["best"]]
        ax.plot(xs, f(xs, I_, p["Vmax"], p["Km"], p["Ki"]), c=c, label=f"[I] = {I_}")
    ax.set_xlabel("[S]")
    ax.set_ylabel("v")
    ax.set_title(f"Global inhibition fit - best model by AICc: {res['best']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "inhibition.png"), dpi=150)
    plt.close(fig)

    print(f"Non-linear : Vmax={nl['Vmax']:.2f}+/-{nl['se_Vmax']:.2f}, Km={nl['Km']:.2f}+/-{nl['se_Km']:.2f}")
    print(f"Lineweaver : Vmax={lb['Vmax']:.2f}, Km={lb['Km']:.2f}")
    print(f"Bootstrap 95% CI: Vmax={tuple(round(float(x), 1) for x in ci["Vmax"])}, Km={tuple(round(float(x), 1) for x in ci["Km"])}")
    print("Best inhibition model:", res["best"])


if __name__ == "__main__":
    make_figures()
