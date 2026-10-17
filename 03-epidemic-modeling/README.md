# 3. Epidemic Modelling: SEIR Dynamics, Vaccination and Inference

Deterministic and stochastic compartmental models of infectious disease.

## What it does
- `simulate_seir` - SEIR ODEs (LSODA, tight tolerances) with optional pre-outbreak vaccination.
- `r0`, `herd_immunity_threshold` (1 - 1/R0), `final_size` (solves z = 1 - exp(-R0 z)).
- `fit_beta_gamma` - least-squares inference of transmission and recovery rates from noisy prevalence.
- `gillespie_sir` / `major_outbreak_probability` - exact stochastic simulation showing that
  even with R0 > 1 many introductions die out.

## Example output
```
Recovered beta=0.547, gamma=0.199, R0=2.74 (true 2.75)
P(major outbreak) R0=2: sim=0.52, theory=0.50
```

## Validation (see `tests/`)
- Population is conserved and compartments stay non-negative.
- ODE attack rate matches the final-size relation; R0 = 2 gives z = 0.7968.
- Vaccinating above the herd-immunity threshold prevents a further outbreak.
- Stochastic major-outbreak probability matches the branching-process result 1 - 1/R0.

## Run
```bash
python -m unittest discover -s tests
python epidemic.py        # writes figures/
```

## Scope
Homogeneous mixing and a single infectious stage. Natural extensions: age structure,
waning immunity, time-varying beta.
