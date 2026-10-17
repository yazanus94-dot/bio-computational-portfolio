# 2. Enzyme Kinetics: Fitting, Inhibition and Model Selection

Parameter estimation for Michaelis-Menten kinetics and discrimination between inhibition mechanisms.

## What it does
- Rate laws: Michaelis-Menten, competitive / uncompetitive / non-competitive inhibition,
  substrate inhibition, Hill.
- `fit_michaelis_menten` - bounded non-linear least squares with standard errors.
- `lineweaver_burk_fit` - double-reciprocal fit, kept to demonstrate its bias.
- `bootstrap_ci` - residual-bootstrap percentile confidence intervals.
- `fit_inhibition` - global fit of all three inhibition models, ranked by AICc.

## Example output (synthetic data, truth Vmax = 100, Km = 25, 6% noise)
```
Non-linear : Vmax=104.78+/-4.63, Km=29.31+/-4.48
Lineweaver : Vmax=87.22,  Km=20.86        <- noticeably more biased
Bootstrap 95% CI: Vmax=(98.1, 114.9), Km=(21.8, 38.3)
Best inhibition model: competitive         <- the true generating mechanism
```
The double-reciprocal transform over-weights low-substrate, low-rate points, which is why
direct non-linear regression is preferred.

## Validation (see `tests/`)
- Exact recovery on noise-free data; tolerance-bounded recovery on noisy data.
- Every inhibition law reduces to Michaelis-Menten at [I] = 0; competitive inhibition leaves Vmax unchanged.
- Hill with n = 1 equals Michaelis-Menten.
- Model selection identifies competitive inhibition and recovers Ki.

## Run
```bash
python -m unittest discover -s tests
python kinetics.py        # writes figures/
```
