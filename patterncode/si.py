from patterncode.config import *

from patterncode.seq_utils import all_strings_of_length

from typing import Any

from tqdm.auto import tqdm
import numpy as np
from scipy.interpolate import interp1d

from patterncode.figures import TheoryForVaryingChannelModel, PErrVsFragmentLen
from patterncode.genome_data import NCBIGenome

from patterncode.ogm_data import OGMData
from scipy.stats import median_abs_deviation
from pydantic import BaseModel

from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

from patterncode.utils import plot_grid, plot_text_annotations, Computation
import matplotlib.patches as patches


class MoleculeAnalysis(BaseModel):
    ref: Any
    locs: Any
    alignment: Any

    crop_length: int = 50000
    num_crops: int = 64
    bin_size: float = 1000
    plot: bool = False

    class Config:
        arbitrary_types_allowed = True
        extra = 'allow'

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

        if self.plot:
            print(f'x_bin_number={x_bin_number}')
            print(f'y_bin_number={y_bin_number}')

        return np.mean(x_bin_number != y_bin_number)

    def misaligned_bins_fraction(self):
        return np.mean([
            self._misaligned_bins_fraction(start) for start in self._crop_starts
        ])

    def stretch_factor_deviation(self):
        return np.mean([
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
    class Config(BaseModel):
        plot = True
        num_molecules = 32
        num_lengths = 24
        figures_dir = '../../PatternCode-Paper/figures'
        save_figure = True

    def __init__(self, **config):
        self.config = self.Config(**config)
        self.data = OGMData.get_molecules_data()

        self.lengths = np.geomspace(10000, 400000, self.config.num_lengths).astype(int)
        self.molecules = self.data.molecules_df.sample(n=self.config.num_molecules, random_state=0).to_dict(
            orient='records')

    def average_misaligned_bins_fraction(self):
        results = [
            np.mean([
                MoleculeAnalysis(**molecule, crop_length=_).misaligned_bins_fraction()
                for molecule in self.molecules
            ])
            for _ in tqdm(self.lengths)
        ]

        if self.config.plot:
            plt.plot(self.lengths, results, 'k.-', )

            plt.xscale('log')
            plt.xlabel('DNA fragment length (bp)')

            plt.ylabel('Bin-misaligned labels fraction')
            plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))

            plt.grid(which='both', axis='both')

        if self.config.save_figure:
            self._savefig('misaligned_bins_fraction')

        return results

    def _savefig(self, name):
        print('saving figure: ', name)
        plt.savefig(self.config.figures_dir + f'/{name}.png', dpi=300)
        plt.savefig(self.config.figures_dir + f'/{name}.pdf')

    def average_stretch(self):
        results = [
            np.mean([
                MoleculeAnalysis(**molecule, crop_length=_).stretch_factor_deviation()
                for molecule in self.molecules
            ]) for _ in tqdm(self.lengths)
        ]

        if self.config.plot:
            plt.plot(self.lengths, results, 'k.-', )

            plt.xscale('log')
            plt.xlabel('DNA fragment length (bp)')

            plt.ylabel('Stretch factor deviation')
            plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))

            plt.grid(which='both', axis='both')

        if self.config.save_figure:
            self._savefig('stretch_factor_deviation')

        return results

    def example_molecule(self):
        return MoleculeAnalysis(**self.molecules[0], plot=True)


class PErrVsPatternVsBinSize(Computation):
    rng = np.random.default_rng(seed=0)

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
        plot_text_annotations(df[x], df[y], df['pattern'],
                              distance=(self.rng.integers(-r, r), self.rng.integers(-r, r)),
                              **kwargs)

    @classmethod
    def plot_vs_bin_size(cls, genome=None, load=True):
        bins_sizes = [100, 500, 750, 1000, 2000]
        colors = ['C0', 'C1', 'C2', 'C3', 'C4']
        _ogm_data = OGMData.get_molecules_data()

        lines = []
        for bin_size, color in zip(bins_sizes, colors):
            line = cls().make(bin_size=bin_size, _genome=genome, _ogm_data=_ogm_data, plot=False, load=load,
                              annotated_patterns=[CTTAAG],
                              save=True).plot_theory(
                color=color)
            lines.append(line)
        legend = [f'bin_size={_} bp' for _ in bins_sizes]
        plt.legend(lines, legend, loc='lower left', fontsize=8, markerscale=3)

    @staticmethod
    def get_genome():
        return NCBIGenome.get_genome(HUMAN_GENOME)


class SIFigures(OGMDataAnalysis):

    def bionano_comparison(self):
        PErrVsFragmentLen.make(
            genome_name=HUMAN_GENOME,
            _ogm_data=OGMData.get_molecules_data(),
            load=1,
            plot=0
        ).plot_theory()
        OGMData.make(
            data_file='/Users/user1/out/PatternCode/export_chosen_fig=fig_b_bionano_20230816T220617Z-melodic-mouse.pkl',
            load=0, plot=1
        )._plot(color='C3')
        lines = plt.gca().lines
        plt.legend(
            lines,
            ['Theory', 'Simulation', 'Experimental: DeepOM', 'Experimental: Bionano']
        )
        self._savefig('bionano_comparison')

    def bin_size_pattern_effect(self, load=True):

        if load:
            genome = None
        else:
            genome = PErrVsPatternVsBinSize.get_genome()

        PErrVsPatternVsBinSize.rng = np.random.default_rng(seed=0)
        ax = plt.gca()
        rect = patches.Rectangle((2 ** (-13), 3e-1), 2 ** (-12), .5, facecolor='none', edgecolor='black')
        ax.add_patch(rect)

        rect = patches.Rectangle((.01, .06), .3, .14, zorder=50,
                                 facecolor='none', edgecolor='black', transform=ax.transAxes)
        ax.add_patch(rect)
        PErrVsPatternVsBinSize.plot_vs_bin_size(genome, load=load)
        plt.xlim(None, 2 ** (-8))
        self._savefig('bin_size_pattern')
