import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import seqtools as st  # noqa: E402


class TestBasics(unittest.TestCase):
    def test_gc_content(self):
        self.assertAlmostEqual(st.gc_content("ATGC"), 0.5)
        self.assertAlmostEqual(st.gc_content("GGGG"), 1.0)

    def test_rna_input_accepted(self):
        self.assertEqual(st.translate("AUGGCUUAA"), "MA*")

    def test_reverse_complement(self):
        self.assertEqual(st.reverse_complement("AAACCCGGGT"), "ACCCGGGTTT")

    def test_revcomp_is_involution(self):
        s = "ATGCGTACGTTAGC"
        self.assertEqual(st.reverse_complement(st.reverse_complement(s)), s)

    def test_invalid_character_rejected(self):
        with self.assertRaises(ValueError):
            st.gc_content("ATGXZ")

    def test_codon_table_is_complete_and_standard(self):
        self.assertEqual(len(st.CODON_TABLE), 64)
        self.assertEqual(st.CODON_TABLE["ATG"], "M")
        self.assertEqual({c for c, a in st.CODON_TABLE.items() if a == "*"}, {"TAA", "TAG", "TGA"})
        self.assertEqual(st.CODON_TABLE["TGG"], "W")

    def test_translate_to_stop(self):
        self.assertEqual(st.translate("ATGAAATAGGGG", to_stop=True), "MK")


class TestKmers(unittest.TestCase):
    def test_kmer_counts(self):
        c = st.kmer_counts("ATATAT", 2)
        self.assertEqual(c["AT"], 3)
        self.assertEqual(c["TA"], 2)

    def test_entropy_bounds(self):
        self.assertAlmostEqual(st.shannon_entropy("AAAAAAAA"), 0.0)
        self.assertAlmostEqual(st.shannon_entropy("ACGTACGT"), 2.0)


class TestORF(unittest.TestCase):
    def test_finds_planted_orf(self):
        orf = "ATG" + "GCT" * 40 + "TAA"
        seq = "CCCC" + orf + "GGGG"
        res = st.find_orfs(seq, min_aa=30)
        self.assertEqual(len(res), 1)
        self.assertEqual((res[0]["start"], res[0]["end"]), (4, 4 + len(orf)))
        self.assertEqual(res[0]["protein"], "M" + "A" * 40)

    def test_finds_reverse_strand_orf_with_forward_coordinates(self):
        orf = "ATG" + "GCT" * 40 + "TAA"
        seq = "CCCC" + st.reverse_complement(orf) + "GGGG"
        res = st.find_orfs(seq, min_aa=30)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["strand"], "-")
        self.assertEqual((res[0]["start"], res[0]["end"]), (4, 4 + len(orf)))

    def test_unterminated_orf_ignored(self):
        self.assertEqual(st.find_orfs("ATG" + "GCT" * 40, min_aa=10), [])


class TestAlignment(unittest.TestCase):
    def test_identical_sequences(self):
        r = st.align("GATTACA", "GATTACA")
        self.assertEqual(r["score"], 7)
        self.assertEqual(r["identity"], 1.0)

    def test_global_known_score(self):
        # Optimum (match=1, mismatch=-1, gap=-2) verified against an independent
        # memoised recursion: GCATGCT vs GATTACA -> -1
        r = st.align("GCATGCT", "GATTACA")
        self.assertEqual(r["score"], -1)
        self.assertEqual(len(r["aligned_a"]), len(r["aligned_b"]))

    def test_global_alignment_reconstructs_inputs(self):
        r = st.align("ACACACTA", "AGCACACA")
        self.assertEqual(r["aligned_a"].replace("-", ""), "ACACACTA")
        self.assertEqual(r["aligned_b"].replace("-", ""), "AGCACACA")

    def test_alignment_score_matches_recomputed_score(self):
        r = st.align("ACACACTA", "AGCACACA", match=2, mismatch=-1, gap=-2)
        s = 0
        for x, y in zip(r["aligned_a"], r["aligned_b"]):
            s += -2 if "-" in (x, y) else (2 if x == y else -1)
        self.assertEqual(s, r["score"])

    def test_local_finds_embedded_motif(self):
        r = st.align("TTTTGATTACATTTT", "CCGATTACACC", mode="local")
        self.assertEqual(r["score"], 7)
        self.assertEqual(r["aligned_a"], "GATTACA")
        self.assertEqual(r["aligned_b"], "GATTACA")
        self.assertEqual(r["start_a"], 4)

    def test_local_score_never_negative(self):
        self.assertEqual(st.align("AAAA", "TTTT", mode="local")["score"], 0)

    def test_global_symmetry(self):
        self.assertEqual(st.align("GATTACA", "GCATGCT")["score"], st.align("GCATGCT", "GATTACA")["score"])

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            st.align("A", "A", mode="semi")

    def test_hamming(self):
        self.assertEqual(st.hamming("GAGCCTACTAACGGGAT", "CATCGTAATGACGGCCT"), 7)


class TestSynthetic(unittest.TestCase):
    def test_gc_island_detected(self):
        g = st.synthetic_genome()
        x, y = st.sliding_gc(g, 150, 10)
        peak = x[y.argmax()]
        self.assertTrue(1200 <= peak <= 1600)

    def test_planted_orf_recovered(self):
        g = st.synthetic_genome()
        orfs = [o for o in st.find_orfs(g, min_aa=55) if o["strand"] == "+" and o["start"] == 300]
        self.assertEqual(len(orfs), 1)


if __name__ == "__main__":
    unittest.main()
