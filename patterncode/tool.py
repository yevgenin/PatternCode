import numpy as np
from pydantic import BaseModel, Field

from patterncode.config import *
from patterncode.dmc import error_probability_dmc
from patterncode.seq_utils import find_pattern_positions


class LikelihoodModel(BaseModel):
    tpr: float
    "True positive labeling rate per bin"

    fpr: float
    "False positive labeling rate per bin"

    p_y_given_x: list[list] = None
    "Label detection likelihood per bin"

    bin_size: int = DEFAULT_BIN_SIZE
    "Bin size in base pairs"

    def __init__(self, **data):
        super().__init__(**data)
        tpr = self.tpr
        fpr = self.fpr
        p01 = fpr
        p11 = tpr + (1 - tpr) * fpr
        self.p_y_given_x = [
            [1 - p01, p01],
            [1 - p11, p11],
        ]


class PatternTool(BaseModel):
    sequence: str = Field(..., repr=False)
    "Genome sequence"

    pattern: str
    "Labeled pattern"

    fragment_len: int | np.ndarray = DEFAULT_MOLECULE_LEN
    "Fragment length in base pairs"

    bin_size: int = DEFAULT_BIN_SIZE
    "Bin size in base pairs"

    genome_len: int = None
    "Genome length in base pairs"

    p_x: list = None
    "Genome per-bin pattern count distribution"

    p_y_given_x: list[list] = None
    "Label detection likelihood per bin"

    p_err: float = None
    "Error probability of a maximum-likelihood decoder"

    _count_cap: int = None

    positions: np.ndarray = Field(None, repr=False)
    "Pattern positions in the genome"

    class Config:
        underscore_attrs_are_private = True
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        p_y_given_x = np.asarray(self.p_y_given_x)
        assert p_y_given_x.shape[0] == p_y_given_x.shape[1]
        self._count_cap = p_y_given_x.shape[0] - 1

        positions = find_pattern_positions(self.sequence, self.pattern)
        x = self._get_x(positions)
        p_x = self._get_p_x(x)
        self.positions = positions
        self.genome_len = len(x) * self.bin_size
        self.p_x = p_x

        B = int(self.bin_size)
        G = int(self.genome_len)
        L = self.fragment_len
        p_x = np.asarray(self.p_x)

        M = 2 * G / B
        n = L / B

        p_x = p_x / p_x.sum()
        p_y_given_x = p_y_given_x / p_y_given_x.sum(axis=1, keepdims=True)

        I, V, p_err = error_probability_dmc(M, n, p_x, p_y_given_x)
        self.p_err = p_err

    def _get_x(self, positions):
        """
        :param positions: positions of pattern in genome
        :return: counts of pattern in bins
        """
        bin_counts = np.bincount(positions // self.bin_size)
        assert bin_counts.sum() == len(positions)
        x = bin_counts.clip(0, self._count_cap)
        return x

    def _get_p_x(self, x):
        """
        :param x: counts of pattern in bins
        :return: probability of counts of pattern in bins
        """
        n = self._count_cap + 1
        p_x = np.bincount(x, minlength=n) / len(x)
        assert len(p_x) == n
        assert np.allclose(p_x.sum(), 1)
        return p_x
