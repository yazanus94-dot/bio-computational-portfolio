# 1. Population Genetics: Drift, Selection and Hardy-Weinberg

A diploid Wright-Fisher model (selection, dominance, two-way mutation, binomial drift) plus
Hardy-Weinberg testing.

## What it does
- `hardy_weinberg_test` - chi-square test (1 d.f.) from genotype counts.
- `wright_fisher` - allele-frequency trajectories; fitnesses AA = 1+s, Aa = 1+hs, aa = 1.
- `simulate_fixation_probability` - Monte-Carlo fixation probability of a single new mutant.
- `kimura_fixation_probability` - diffusion approximation (per-allele advantage s/2 under h = 1/2).
- `heterozygosity_decay` - neutral loss of heterozygosity across independent loci.

## Validation (see `tests/`)
- Neutral fixation probability equals the initial frequency 1/2N.
- Simulated fixation under selection agrees with Kimura's formula within Monte-Carlo error.
- Mean heterozygosity decays at the theoretical rate (1 - 1/2N) per generation.

> **Convention note.** An early version of the test disagreed with theory by roughly a factor of two.
> The cause was a parameterisation mismatch (the model's per-allele advantage is s/2, Kimura's
> formula assumes s per allele). The theory function was corrected and the convention documented,
> rather than loosening the test tolerance.

## Run
```bash
python -m unittest discover -s tests
python popgen.py          # writes figures/
```

Figures: `trajectories.png`, `heterozygosity.png`, `fixation_probability.png`.
