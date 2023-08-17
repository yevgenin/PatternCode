import os
from functools import lru_cache

# os.environ['PC_QUICK_RUN'] = '1'
from patterncode.config import *

from patterncode.seq_utils import all_strings_of_length

from typing import Any, Callable

from tqdm.auto import tqdm
import numpy as np
from scipy.interpolate import interp1d

from patterncode.figures import Evaluation, SimulationForPattern, BinnedOGMData, TheoryForVaryingPattern, \
    TheoryForVaryingChannelModel
from patterncode.genome_data import GenomeIndex, NCBIGenome

from patterncode.ogm_data import OGMData
from scipy.stats import median_abs_deviation
from pydantic import BaseModel

from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

from patterncode.utils import sample_uniformly, plot_grid, plot_ci, plot_text_annotations, Computation, cached_func


class MoleculeAnalysis(BaseModel):
    ref: Any
    locs: Any
    alignment: Any

    crop_length: int = 50000
    num_crops: int = 100
    bin_size: float = 1000

    def __init__(self, **data: Any):
        super().__init__(**data)
        r, q = self.alignment.T
        self.ref = self.ref[r] - self.ref[r[0]]
        self.locs = self.locs[q] - self.locs[q[0]]
        self._alignment_interp = interp1d(self.ref, self.locs)
        self._crop_starts = np.linspace(0, self.ref.max() - self.crop_length, self.num_crops)

    def _misaligned_bins_fraction(self, start: int):
        stop = start + self.crop_length
        x = self.ref[slice(*self.ref.searchsorted([start, stop]))]
        y = self._alignment_interp(x)
        if len(x) < 2 or len(y) < 2:
            return 0
        x = x - x[0]
        y = y - y[0]
        scale = (x[-1] - x[0]) / (y[-1] - y[0])
        x_bin_number = (x / self.bin_size).astype(int)
        y_bin_number = (y * scale / self.bin_size).astype(int)
        return np.mean(x_bin_number != y_bin_number)

    def misaligned_bins_fraction(self):
        return np.median([
            self._misaligned_bins_fraction(start) for start in self._crop_starts
        ])

    def stretch_factor_deviation(self):
        return np.median([
            self.deviation_metric(self.stretch_factor(start)) for start in self._crop_starts
        ])

    def stretch_factor(self, start: int):
        ref = np.arange(start, start + self.crop_length, self.bin_size)
        qry = self._alignment_interp(ref)
        return np.diff(ref) / np.diff(qry)

    @staticmethod
    def deviation_metric(x):
        return median_abs_deviation(x) / np.median(x)


class OGMDataAnalysis:
    def __init__(self, num_samples=10):
        self.data = OGMData.get_molecules_data()

        self.lengths = np.geomspace(10000, 400000, 30).astype(int)
        self.molecules = self.data.molecules_df.sample(n=num_samples, random_state=0).to_dict(orient='records')

    def molecule(self):
        return MoleculeAnalysis(**self.molecules[0])

    def _average_misaligned_bins_fraction(self):
        return np.median([
            [MoleculeAnalysis(**molecule, crop_length=_).misaligned_bins_fraction()
             for _ in self.lengths]
            for molecule in tqdm(self.molecules)
        ], axis=0)

    def plot_average_misaligned_bins_fraction(self):
        plt.plot(self.lengths, self._average_misaligned_bins_fraction(), 'k.-', )

        plt.xscale('log')
        plt.xlabel('DNA fragment length (bp)')

        plt.ylabel('Misaligned bins fraction')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))

        plt.grid(which='both', axis='both')

    def _average_stretch(self):
        return np.median([
            [MoleculeAnalysis(**molecule, crop_length=_).stretch_factor_deviation()
             for _ in self.lengths]
            for molecule in tqdm(self.molecules)
        ], axis=0)

    def plot_average_stretch(self):
        plt.plot(self.lengths, self._average_stretch(), 'k.-', )

        plt.xscale('log')
        plt.xlabel('DNA fragment length (bp)')

        plt.ylabel('Stretch factor deviation')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))

        plt.grid(which='both', axis='both')


rng = np.random.default_rng(seed=0)


class PErrVsPatternVsBinSize(Computation):

    def __init__(self):
        super().__init__()
        self.patterns = list({*all_strings_of_length(length=6), *SPECIAL_OGM_PATTERNS})
        self.extra_patterns = None
        self.annotated_patterns = None
        self.bin_size = None
        self._genome = None
        self._ogm_data = None

    @property
    def name_prefix(self):
        return f'pattern_bin_size={self.bin_size}'

    def _compute(self):
        self.theory_df = self._swipe(TheoryForVaryingChannelModel.init(
            _genome=self._genome,
            _ogm_data=self._ogm_data,
            bin_size=self.bin_size,
            molecule_len=DEFAULT_MOLECULE_LEN_PER_GENOME[HUMAN_GENOME],
        ), 'pattern', self.patterns)

    def plot_theory(self, marker_size=.5, color='k', marker='o'):
        """
        plot p_err vs density for theory
        """
        plt.xlabel(DENSITY)
        plt.ylabel(f'{ERROR_PROBABILITY} ({THEORY})')
        plt.xscale('log', base=2)
        plt.yscale('log', base=10)

        df = self.theory_df
        df = df.copy()

        x = 'density'
        y = 'p_err'

        df = df.sort_values(x)
        line = plt.scatter(df[x], df[y], marker=marker, s=marker_size, c=color, alpha=.8)

        if self.extra_patterns is not None:
            df_extra = df[df['pattern'].isin(self.extra_patterns)]
            plt.scatter(df_extra[x], df_extra[y], marker='x', c='r', s=10, alpha=.7)

        if self.annotated_patterns is not None:
            self._annotate(x, y, df, self.annotated_patterns, color=color)

        plt.ylim(1e-4, 5)
        plot_grid()
        return line

    def _annotate(self, x, y, df, annotated_patterns, **kwargs):
        df = df[df['pattern'].isin(annotated_patterns)]
        df = df.sort_values(y)
        r = 100
        plot_text_annotations(df[x], df[y], df['pattern'], distance=(rng.integers(-r, r), rng.integers(-r, r)), **kwargs)

    @classmethod
    def plot_vs_bin_size(cls, genome=None):
        bins_sizes = [100, 500, 750, 1000]
        colors = ['C0', 'C1', 'C2', 'C3']
        if genome is None:
            genome = cls.get_genome()
        _ogm_data = OGMData.get_molecules_data()

        lines = []
        for bin_size, color in zip(bins_sizes, colors):
            line = cls().make(bin_size=bin_size, _genome=genome, _ogm_data=_ogm_data, plot=False, load=1,
                              annotated_patterns=[CTTAAG],
                              save=True).plot_theory(
                color=color)
            lines.append(line)
        legend = [f'bin_size={_} bp' for _ in bins_sizes]
        plt.legend(lines, legend, loc='lower left', fontsize=8, markerscale=3)

    @staticmethod
    def get_genome():
        return NCBIGenome.get_genome(HUMAN_GENOME)
