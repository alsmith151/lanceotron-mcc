import pathlib
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
import pyBigWig
import pydantic
import pyranges as pr
import ray
import scipy.ndimage
import scipy.sparse
import scipy.sparse.lil
import tqdm
from loguru import logger


class Chromosomes:
    def __init__(self, chromosomes: List[str]):
        self.chromosomes = set(chromosomes)

    def __repr__(self):
        return f"Chromosomes({self.chromosomes})"

    @property
    def standard_chromosomes(self):
        include = r"(chr)?([1-9]+|X|Y)$"
        exclude = r"(chr)?(M|Un|Random|Alt|HLA|EBV|NT|KI|GL|JH|AC|NC|NT|NW|NT_)"
        return self.filter_chromosomes(include=include, exclude=exclude)

    def filter_chromosomes(self, exclude: str = None, include: str = None):
        chrom_ser = pd.Series(list(self.chromosomes))

        import warnings

        warnings.filterwarnings("ignore")

        if exclude is not None:
            chrom_ser = chrom_ser[~chrom_ser.str.contains(exclude)]

        if include is not None:
            chrom_ser = chrom_ser[chrom_ser.str.contains(include)]

        warnings.filterwarnings("default")

        return Chromosomes(chrom_ser.tolist())

    @classmethod
    def from_bed(cls, bed_file: str):
        bed = pr.read_bed(bed_file)
        bed_df = bed.df
        chromosomes = bed_df.Chromosome.drop_duplicates()
        return cls(chromosomes=chromosomes.tolist())

    @classmethod
    def from_bigwig(cls, bigwig_file: str):
        with pyBigWig.open(bigwig_file) as bw:
            chromosomes = list(bw.chroms())

        return cls(chromosomes=chromosomes)

    def sample(self, n: int):
        return Chromosomes(chromosomes=np.random.choice(list(self.chromosomes), n))
    
    def __add__(self, other):
        
        chroms_left = self.chromosomes.copy()
        chroms_right = other.chromosomes.copy()

        return Chromosomes(chromosomes={*chroms_left, *chroms_right})

class ChromosomeStats(pydantic.BaseModel):
    name: str
    length: int
    mean: float
    std: float