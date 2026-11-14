import numpy as np
import pandas as pd
import pydantic
import pyranges as pr

from .io import BigWigReader


class Chromosomes:
    def __init__(self, chromosomes: list[str]):
        self.chromosomes = set(chromosomes)

    def __repr__(self):
        return f"Chromosomes({self.chromosomes})"

    def __len__(self):
        return len(self.chromosomes)

    @property
    def standard_chromosomes(self):
        include = r"^(?:chr)?(?:[1-9]|1[0-9]|2[0-2]|X|Y)$"
        return self.filter_chromosomes(include=include)

    def filter_chromosomes(self, exclude: str = None, include: str = None):
        chrom_ser = pd.Series(sorted(self.chromosomes), dtype="string")

        if exclude is not None:
            chrom_ser = chrom_ser[~chrom_ser.str.contains(exclude, regex=True, na=False)]

        if include is not None:
            chrom_ser = chrom_ser[chrom_ser.str.contains(include, regex=True, na=False)]

        return Chromosomes(chrom_ser.tolist())

    @classmethod
    def from_bed(cls, bed_file: str):
        bed = pr.read_bed(bed_file)
        bed_df = bed.df
        chromosomes = bed_df.Chromosome.drop_duplicates()
        return cls(chromosomes=chromosomes.tolist())

    @classmethod
    def from_bigwig(cls, bigwig_file: str):
        with BigWigReader(str(bigwig_file)) as reader:
            return cls(chromosomes=list(reader.chroms))

    def sample(self, n: int, seed: int | None = None):
        if isinstance(n, bool) or not isinstance(n, int) or n < 0 or n > len(self.chromosomes):
            raise ValueError("n must be between zero and the number of chromosomes")
        chroms = sorted(self.chromosomes)
        rng = np.random.default_rng(seed)
        chroms_sampled = rng.choice(chroms, size=n, replace=False).tolist()
        return Chromosomes(chromosomes=chroms_sampled)

    def __str__(self):
        return f"Chromosomes({self.chromosomes})"

    def __format__(self, format_spec: str) -> str:
        return format(str(self), format_spec)

    def __add__(self, other):

        chroms_left = self.chromosomes.copy()
        chroms_right = other.chromosomes.copy()

        return Chromosomes(chromosomes={*chroms_left, *chroms_right})


class ChromosomeStats(pydantic.BaseModel):
    name: str
    length: int
    mean: float
    std: float

    @pydantic.model_validator(mode="after")
    def _validate_stats(self):
        if self.length < 0:
            raise ValueError("length must be non-negative")
        if not np.isfinite(self.mean) or not np.isfinite(self.std) or self.std < 0:
            raise ValueError("mean and std must be finite, and std must be non-negative")
        return self


def get_device(use_gpu: bool = True) -> str:
    """
    Get the device type (CPU or GPU) based on the environment variable.
    """
    import torch

    device = "cpu"
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
    elif use_gpu and torch.backends.mps.is_available():
        device = "mps"

    return device
