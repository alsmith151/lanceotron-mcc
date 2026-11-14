import pytest

from mcc_peaks.api import utils


def test_standard_chromosomes_include_chr10_and_chr20_only_exact_names():
    chromosomes = utils.Chromosomes(
        ["chr1", "chr10", "chr20", "chr22", "chrX", "chrY", "chrM", "chr23", "prefix_chr10"]
    )
    assert chromosomes.standard_chromosomes.chromosomes == {
        "chr1",
        "chr10",
        "chr20",
        "chr22",
        "chrX",
        "chrY",
    }


def test_sample_is_seeded_without_replacement():
    chromosomes = utils.Chromosomes(["chr1", "chr2", "chr3", "chr10"])
    first = chromosomes.sample(3, seed=17).chromosomes
    second = chromosomes.sample(3, seed=17).chromosomes
    assert first == second
    assert len(first) == 3
    assert first <= chromosomes.chromosomes
    assert chromosomes.sample(0, seed=17).chromosomes == set()
    with pytest.raises(ValueError):
        chromosomes.sample(True)


def test_chromosome_stats_reject_nonfinite_or_invalid_values():
    with pytest.raises(ValueError):
        utils.ChromosomeStats(name="chr1", length=10, mean=float("nan"), std=0)
    with pytest.raises(ValueError):
        utils.ChromosomeStats(name="chr1", length=-1, mean=0, std=0)


def test_from_bigwig_uses_reader_chroms_property(monkeypatch):
    class Reader:
        chroms = {"chr1": 10, "chr10": 20}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(utils, "BigWigReader", lambda path: Reader())
    assert utils.Chromosomes.from_bigwig("example.bw").chromosomes == {"chr1", "chr10"}
