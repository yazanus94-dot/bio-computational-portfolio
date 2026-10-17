# 4. Sequence Bioinformatics From Scratch

Core sequence-analysis algorithms implemented without Biopython, to show understanding of the
algorithms rather than the libraries.

## What it does
- Composition: GC content, k-mer counts, Shannon entropy, sliding-window GC profile.
- Translation with the standard genetic code (RNA input accepted).
- `find_orfs` - ATG-initiated ORFs in all six reading frames, with forward-strand coordinates.
- `align` - Needleman-Wunsch (global) and Smith-Waterman (local) with linear gap penalties
  and full traceback.
- `synthetic_genome` - random genome with a GC-rich island and a planted ORF for demonstration.

## Validation (see `tests/`)
- Alignment scores verified against an independent memoised recursion;
  the returned alignment is re-scored from scratch and must equal the reported score.
- Gapped sequences reduce back to the inputs; global scoring is symmetric; local score is never negative.
- Local alignment recovers an embedded motif at the right offset.
- Reverse-strand ORFs are reported in forward coordinates; unterminated ORFs are ignored.
- The GC island and the planted ORF are recovered from the synthetic genome.

## Reading the genome-scan figure
The large GC peak near 1,400 bp is the embedded GC-rich island. The smaller bump near 400 bp is the
planted ORF itself: its repeated GCT codons are GC-rich, so it also shows up in the GC profile.

## Run
```bash
python -m unittest discover -s tests
python seqtools.py        # writes figures/
```

## Possible extensions
Affine gap penalties (Gotoh), BLOSUM62 for proteins, banded alignment for long sequences.
