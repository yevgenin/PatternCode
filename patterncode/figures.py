import itertools
from abc import ABC
from functools import lru_cache

import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from more_itertools import is_sorted
from scipy.interpolate import interp1d

from patterncode.config import *
from patterncode.dmc import error_probability_dmc
from patterncode.genome_data import GenomeIndex
from patterncode.illustration import plot_image_illustration
from patterncode.ogm_data import OGMData
from patterncode.utils import set_seed, filter_uniformly, plot_ci, plot_text_annotations, make_subplots, Computation, \
    plot_grid


class Evaluation(Computation, ABC):
    def __init__(self):
        super().__init__()
        self.genome_name = HUMAN_GENOME
        self.show_ci = SHOW_CI
        self.num_trials = DEFAULT_NUM_TRIALS

    @property
    def name(self):
        return f'{super().name}-{self.genome_name}'

    def _plot(self):
        plt.figure(figsize=(2 * FIG_SIZE, FIG_SIZE))

        plt.subplot(1, 2, 1)
        self.plot_theory()

        plt.subplot(1, 2, 2)
        self.plot_comparison()

        plt.subplots_adjust(top=0.8, wspace=.7)

    def plot_theory(self):
        pass

    def plot_comparison(self):
        pass


class BinnedFragment(Computation):
    def __init__(self):
        super().__init__()
        self.locs = None
        self.image = None
        self.ref = None
        self.alignment = None
        self.ogm_data_item = None
        self.x_binned = None
        self.y_binned = None

        self.bin_size = DEFAULT_BIN_SIZE
        self.align_length = DEFAULT_ALIGN_LENGTH

    def _compute(self):
        locs = self.ogm_data_item.locs
        image = self.ogm_data_item.image
        ref = self.ogm_data_item.ref
        alignment = self.ogm_data_item.alignment

        aligned_locs = self._get_aligned_localizations(alignment, locs, ref)

        bounds = aligned_locs[[0, -1]]
        start, stop = sorted(ref.searchsorted(bounds))
        ref_segment = ref[start: stop]

        B = self.bin_size
        x_pos = ref_segment
        y_pos = aligned_locs

        x0 = x_pos[0]
        x_pos = x_pos - x0
        y_pos = y_pos - x0

        assert x_pos[-1] <= y_pos[-1], 'x[-1]: %s, y[-1]: %s' % (x_pos[-1], y_pos[-1])
        assert x_pos[0] == 0 and y_pos[0] == 0, 'x[0] != 0 or y[0] != 0'

        y = np.bincount(y_pos.astype(int) // B)
        x = np.bincount(x_pos.astype(int) // B, minlength=len(y))

        self.x_binned = x
        self.y_binned = y
        self.aligned_locs = aligned_locs
        self.ref_segment = ref_segment
        self.image = image
        self.ref = ref
        self.alignment = alignment
        self.locs = locs

    def _get_aligned_localizations(self, alignment, locs, ref):
        """
        Get aligned localizations

        :param alignment: alignment
        :param locs: localizations
        :param ref: reference
        :return: aligned localizations
        """
        r, q = alignment.T

        alignment_ref = ref[r]
        alignment_locs = locs[q]
        assert is_sorted(alignment_ref)

        xmin, xmax = alignment_ref.min(), alignment_ref.max()

        i = alignment_ref.searchsorted(np.arange(xmin, xmax, self.align_length))

        # always include the last index
        indices = list({*i, len(alignment_ref) - 1})
        alignment_ref = alignment_ref[indices]
        alignment_locs = alignment_locs[indices]

        aligned_locs = interp1d(alignment_locs, alignment_ref, bounds_error=False, fill_value="extrapolate")(locs)
        return aligned_locs


class BinnedOGMData(Computation):
    def __init__(self):
        super().__init__()
        self.run_name = ''
        self.load = False
        self.y_binned_all = None
        self.x_binned_all = None
        self.count_cap = DEFAULT_COUNT_CAP
        self.bin_size = DEFAULT_BIN_SIZE
        self.align_length = DEFAULT_ALIGN_LENGTH
        self.likelihood = None
        self._report_df = None
        self._ogm_data = None

    @staticmethod
    @lru_cache
    def compute_likelihood():
        data = BinnedOGMData.compute(
            _ogm_data=OGMData.get_molecules_data(),
            verbose=1,
        )
        likelihood = data.likelihood
        return likelihood

    def _compute(self):
        assert self._ogm_data is not None, 'no molecule data'

        items = list(self._ogm_data.molecules_df.itertuples())

        fragment = BinnedFragment()
        fragment.bin_size = self.bin_size
        fragment.align_length = self.align_length
        report_df = self._swipe(fragment, 'ogm_data_item', items)

        x_b = np.concatenate(report_df['x_binned'])
        y_b = np.concatenate(report_df['y_binned'])

        h_xy = self._get_hist_xy(x_b, y_b, self.count_cap)
        p_xy = h_xy / np.sum(h_xy)
        p_y_given_x = p_xy / np.sum(p_xy, axis=1, keepdims=True)

        self._report_df = report_df
        self.likelihood = p_y_given_x
        self.x_binned_all = x_b
        self.y_binned_all = y_b

    def _plot(self):
        plt.figure(figsize=(FIG_SIZE, FIG_SIZE / 4))
        self.plot_images()

    def _plot_stats(self):
        assert self._report_df is not None, 'report_df is None'
        df = self._report_df
        plt.subplot(1, 3, 1)
        locs_len = df['locs'].apply(len)
        plt.hist(locs_len, bins='auto')
        plt.xlabel('locs len')
        plt.ylabel('count')
        plt.subplot(1, 3, 2)
        # hist of image widths
        image_widths = df['image'].apply(lambda x: x.shape[-1])
        plt.hist(image_widths, bins='auto')
        plt.xlabel('image width')
        plt.ylabel('count')
        plt.subplot(1, 3, 3)
        # hist of alignment lengths
        alignment_lens = df['alignment'].apply(len)
        plt.hist(alignment_lens, bins='auto')
        plt.xlabel('alignment len')
        plt.ylabel('count')

    def plot_images(self):
        assert self._report_df is not None, 'report_df is None'
        df = self._report_df
        row = df.iloc[0]
        image = row['image'][0]
        locs = row['locs']
        width = 7
        image = image[image.shape[0] // 2 - width // 2 + 1:image.shape[0] // 2 + width // 2 + 2]
        starts = np.arange(0, image.shape[1], 50).astype(int)

        segments = [
                       (start, end)
                       for start, end in zip(starts, starts[1:])
                   ][:5]
        crops = [
            image[:, start:end]
            for start, end in segments
        ]
        fig, axes = plt.subplots(len(crops), 1)
        for ax, crop, segment in zip(axes.flat, crops, segments):
            y = locs
            y = y[slice(*y.searchsorted(segment))]
            y = y - segment[0]
            ax: 'Axes'
            ax.invert_yaxis()
            ax.eventplot(y, colors='red', lineoffsets=0, linewidths=1)
            ax.imshow(crop, cmap='gray', interpolation='none', aspect='auto')
            ax.set_axis_off()

    @staticmethod
    @numba_parallel
    def _get_hist_xy(x_b: np.ndarray, y_b: np.ndarray, cap: int):
        x_b = x_b.clip(0, cap)
        y_b = y_b.clip(0, cap)
        hist = np.zeros((cap + 1, cap + 1), dtype=np.int64)
        for x, y in zip(x_b, y_b):
            hist[x, y] += 1
        return hist


class BinnedGenome(Computation):
    def __init__(self):
        super().__init__()
        self.genome_len = None
        self._positions = None
        self._bin_counts = None
        self._ref_x = None
        self.p_x = None
        self.count_cap = DEFAULT_COUNT_CAP
        self.bin_size = DEFAULT_BIN_SIZE
        self.pattern = DEFAULT_PATTERN
        self.add_rev_comp = True
        self._genome = None

    def _compute(self):
        assert self._genome is not None, 'genome is not set'
        assert self.pattern is not None, 'pattern is not set'
        assert self.bin_size is not None, 'bin_size is not set'
        assert self.count_cap is not None, 'count_cap is not set'

        positions = self._genome.get_pattern_positions(pattern=self.pattern, add_rev_comp=self.add_rev_comp)
        x = self._get_x(positions)
        p_x = self._get_p_x(x)

        self._ref_x = x
        self.positions_len = len(positions)
        self.genome_len = len(x) * self.bin_size
        self.p_x = p_x

    def _get_x(self, positions):
        """
        :param positions: positions of pattern in genome
        :return: counts of pattern in bins
        """
        bin_counts = np.bincount(positions // self.bin_size)
        assert bin_counts.sum() == len(positions)
        x = bin_counts.clip(0, self.count_cap)
        return x

    def _get_p_x(self, x):
        """
        :param x: counts of pattern in bins
        :return: probability of counts of pattern in bins
        """
        n = self.count_cap + 1
        p_x = np.bincount(x, minlength=n) / len(x)
        assert len(p_x) == n
        assert np.allclose(p_x.sum(), 1)
        return p_x


class Theory(Computation):
    def __init__(self):
        super().__init__()
        self.genome_len = None
        self.positions_len = None
        self.p_x = None
        self.likelihood = None
        self.bin_size = None
        self.molecule_len = None
        self.p_err = None

    def _compute(self):
        assert self.genome_len is not None, 'genome_len is not set'
        assert self.p_x is not None, 'p_x is not set'
        assert self.likelihood is not None, 'likelihood is not set'
        assert self.bin_size is not None, 'bin_size is not set'
        assert self.molecule_len is not None, 'molecule_len is not set'

        B = int(self.bin_size)
        G = int(self.genome_len)
        L = self.molecule_len
        p_x = np.asarray(self.p_x)
        p_y_given_x = np.asarray(self.likelihood)

        M = 2 * G / B
        n = L / B

        p_x = p_x / p_x.sum()
        p_y_given_x = p_y_given_x / p_y_given_x.sum(axis=1, keepdims=True)

        I, V, p_err = error_probability_dmc(M, n, p_x, p_y_given_x)

        self.info = I
        self.info_var = V
        self.p_err = p_err


class TheoryForVaryingPattern(Theory):
    def __init__(self):
        super().__init__()
        self.likelihood = None
        self.count_cap = None
        self.bin_size = None
        self.p_err = None
        self._genome = None
        self._binned_genome = None
        self.pattern = None

    def _compute(self):
        binned_genome = BinnedGenome.compute(
            _genome=self._genome,
            pattern=self.pattern,
            verbose=2,
        )
        self._set(
            p_x=binned_genome.p_x,
            genome_len=binned_genome.genome_len,
            positions_len=binned_genome.positions_len
        )
        super()._compute()

        self._binned_genome = binned_genome
        self.density = self.positions_len / self.genome_len


class TheoryForVaryingChannelModel(Theory):

    def __init__(self):
        super().__init__()
        self.pattern = None
        self._ogm_data = None
        self._genome = None
        self.align_length = DEFAULT_ALIGN_LENGTH
        self.bin_size = DEFAULT_BIN_SIZE

    def _compute(self):
        likelihood_estimator = BinnedOGMData.compute(
            _ogm_data=self._ogm_data,
            bin_size=self.bin_size,
            align_length=self.align_length,
            verbose=2,
        )
        binned_genome = BinnedGenome.compute(
            _genome=self._genome,
            bin_size=self.bin_size,
            pattern=self.pattern,
            verbose=2,
        )
        self._set(
            likelihood=likelihood_estimator.likelihood,
            p_x=binned_genome.p_x,
            genome_len=binned_genome.genome_len,
        )
        super()._compute()


class SimulationTrial(Computation):
    def __init__(self):
        super().__init__()
        self.likelihood = None
        self._ref_x = None
        self.bin_size = None
        self.molecule_len = None

    def _compute(self):
        """
        Simulate a molecule from the reference
        """
        assert self._ref_x is not None, 'ref_x is not set'
        assert self.bin_size is not None, 'bin_size is not set'
        assert self.molecule_len is not None, 'molecule_len is not set'
        assert self.likelihood is not None, 'likelihood is not set'

        N = int(self.molecule_len)
        b = int(self.bin_size)
        x = np.asarray(self._ref_x)
        p_y_given_x = np.asarray(self.likelihood)
        p_y_given_x = p_y_given_x / p_y_given_x.sum(axis=1, keepdims=True)

        n = N // b

        true_start = np.random.randint(0, len(x) - n)
        x_fragment = x[true_start:true_start + n]
        assert len(x_fragment) == n

        y = self._get_y(x_fragment, p_y_given_x)
        assert len(y) == n

        log_likelihood = np.log(self.likelihood)
        score_vector = np.concatenate([
            self._get_score(x, y_, log_likelihood)
            for y_ in [y, y[::-1]]
        ])
        assert len(score_vector) == 2 * (len(x) - n + 1)

        score_true = score_vector[true_start]
        match_start = score_vector.argmax()
        score_match = score_vector[match_start]
        is_error = match_start != true_start

        self._x_fragment = x_fragment
        self._true_start = true_start
        self._qry_y = y
        self._match_start = match_start
        self._score_true = score_true
        self._score_match = score_match
        self._score = score_vector
        self.is_error = is_error

    @staticmethod
    def _get_y(x: np.ndarray, p_y_given_x: np.ndarray):
        """
        samples y from p(y|x)

        :param x: reference sequence
        :param p_y_given_x: likelihood table
        :return: query sequence
        """
        y = np.zeros_like(x)
        n, m = p_y_given_x.shape
        for i, x_i in enumerate(x):
            assert 0 <= x_i < n
            y[i] = np.random.choice(m, p=p_y_given_x[x_i])
        return y

    @staticmethod
    @numba_parallel
    def _get_score(x: np.ndarray, y: np.ndarray, score_table: np.ndarray):
        """
        computes:
            \sum_{j=1}^m \log p(x_{i+j}|y_j)

        :param x: reference sequence
        :param y: query sequence
        :param score_table: log likelihood table
        :return: score vector
        """
        n, m = score_table.shape
        assert len(x) >= len(y)

        result = np.zeros(len(x) - len(y) + 1, dtype=np.float64)

        for i in range(len(result)):
            sum_j = 0
            for j in range(len(y)):
                x_val = x[i + j]
                y_val = y[j]
                assert 0 <= x_val < n and 0 <= y_val < m

                sum_j += score_table[x_val, y_val]
            result[i] = sum_j
        return result


class SimulationMixin(Computation):

    def __init__(self):
        super().__init__()
        self._binned_genome = None
        self.likelihood = None
        self.bin_size = None
        self.molecule_len = None
        self.simulate = True
        self.error_rate = None
        self.confidence_interval = None
        self.num_trials = DEFAULT_NUM_TRIALS

    def _compute(self):
        assert self._binned_genome is not None, 'binned_genome is not set'
        assert self.likelihood is not None, 'likelihood is not set'
        assert self.bin_size is not None, 'bin_size is not set'
        assert self.molecule_len is not None, 'molecule_len is not set'
        assert self.num_trials is not None, 'num_trials is not set'

        if not self.simulate:
            return
        # noinspection PyProtectedMember
        base = SimulationTrial.init(
            bin_size=self.bin_size,
            likelihood=self.likelihood,
            molecule_len=self.molecule_len,
            _ref_x=self._binned_genome._ref_x
        )
        num_trials = self.num_trials
        df = self._swipe(base, attr=None, vals=range(num_trials))
        error_count = df['is_error'].sum()
        num_trials = len(df)
        self.num_trials = num_trials
        self.error_count = error_count
        self.error_rate = error_count / num_trials


class SimulationForPattern(SimulationMixin, TheoryForVaryingPattern):
    def __init__(self):
        SimulationMixin.__init__(self)
        TheoryForVaryingPattern.__init__(self)
        self.verbose = 2

    def _compute(self):
        TheoryForVaryingPattern._compute(self)
        SimulationMixin._compute(self)


class SimulationForFragmentLen(SimulationMixin, Theory):
    def __init__(self):
        Theory.__init__(self)
        SimulationMixin.__init__(self)

    def _compute(self):
        Theory._compute(self)
        SimulationMixin._compute(self)


class PErrVsPattern(Evaluation):
    name_prefix = 'pattern'

    def __init__(self):
        super().__init__()
        self.molecule_len = None
        self._show_ci = SHOW_CI
        self.df = None
        self._simulation = None

        self.num_simulate_patterns = NUM_SIMULATE_PATTERNS
        self.limit_patterns = LIMIT_PATTERNS

    def _compute(self):
        self.molecule_len = DEFAULT_MOLECULE_LEN_PER_GENOME[self.genome_name]
        self._simulation = SimulationForPattern.init(
            _genome=GenomeIndex.get_genome(self.genome_name),
            bin_size=DEFAULT_BIN_SIZE,
            count_cap=DEFAULT_COUNT_CAP,
            likelihood=BinnedOGMData.compute_likelihood(),
            pattern=DEFAULT_PATTERN,
            molecule_len=self.molecule_len,
            num_trials=DEFAULT_NUM_TRIALS,
        )
        patterns = list(map(''.join, itertools.product(ACGT, repeat=6)))

        if self.limit_patterns is not None:
            np.random.shuffle(patterns)
            patterns = list({*patterns[:self.limit_patterns], DEFAULT_PATTERN})

        base = self._simulation.replace(simulate=False)
        theory_df = self._swipe(base, 'pattern', patterns)

        df = theory_df
        df = df.loc[filter_uniformly(theory_df, by='p_err', n=self.num_simulate_patterns)]
        patterns = list({*df['pattern'], DEFAULT_PATTERN})
        base = self._simulation.replace(simulate=True)
        sim_df = self._swipe(base, 'pattern', patterns)

        self.sim_df = sim_df
        self.theory_df = theory_df

    def data_table(self):
        """
        data export table for the theory plot, sorted by error probability from low to high
        :return: data frame
        """
        df = self.theory_df[['pattern', 'p_err', 'density']]
        df = df.sort_values('p_err')
        df = df.rename(columns={
            'pattern': PATTERN,
            'density': DENSITY,
            'p_err': ERROR_PROBABILITY,
        }).reset_index(drop=True)

        return df

    def plot_theory(self):
        plt.xlabel(DENSITY)
        plt.ylabel(f'{ERROR_PROBABILITY} ({THEORY})')
        plt.xscale('log', base=2)
        plt.yscale('log', base=10)
        df = self.theory_df
        df = df.copy()
        x = 'density'
        y = 'p_err'
        df = df.sort_values(x)
        plt.plot(df[x], df[y], label=THEORY, ls='', marker='o', alpha=1, ms=.5)

        if NUM_ANNOTATE_PATTERNS is not None:
            annotated_patterns = self._select_annotated(self.theory_df, by='p_err')
            self._annotate(x, y, df, annotated_patterns, xytext=(50, 0))

        plt.ylim(None, 1.1)
        plot_grid()

    def plot_comparison(self):
        df = self.sim_df
        x = 'p_err'
        y = 'error_rate'
        df = df.sort_values(x)
        x_eq = df[x].iloc[[0, -1]]

        plt.plot(x_eq, x_eq, '--k', alpha=.5)
        plt.plot(df[x], df['error_rate'], ls='', marker='.', alpha=1, ms=2, color='C1', label=SIMULATION)

        if self._show_ci:
            plot_ci(df[x], df['error_count'], df['num_trials'], alpha=0.5, color='C1', ls='-')

        if NUM_ANNOTATE_PATTERNS is not None:
            annotated_patterns = self._select_annotated(self.sim_df, by='error_rate')
            self._annotate(x, y, df, annotated_patterns, xytext=(-50, 0))
        plt.xlabel(f'{ERROR_PROBABILITY} ({THEORY})')
        plt.ylabel(f'{ERROR_PROBABILITY} ({SIMULATION})')
        plot_grid()

    def _annotate(self, x, y, df, annotated_patterns, **kwargs):
        df = df[df['pattern'].isin(annotated_patterns)]
        df = df.sort_values(y)
        plot_text_annotations(df[x], df[y], df['pattern'], highlight=DEFAULT_PATTERN, **kwargs)

    def _select_annotated(self, df, by):
        if NUM_ANNOTATE_PATTERNS == 0:
            return [DEFAULT_PATTERN]
        df = df[df['pattern'].isin(self.sim_df['pattern'])]
        log = plt.gca().get_yscale() == 'log'
        indices = filter_uniformly(df, by=by, n=NUM_ANNOTATE_PATTERNS, log=log)
        indices = list({*indices, df[df['pattern'] == DEFAULT_PATTERN].index[0]})
        annotated_patterns = df['pattern'].loc[indices]
        return annotated_patterns


class PErrVsFragmentLen(Evaluation):
    name_prefix = 'len'

    def __init__(self):
        super().__init__()
        self.lims = 10000, 400000
        self._ogm_data = None
        self.exp_df = None
        self.num_simulate_lens = NUM_SIMULATE_LENS

    def _compute(self):
        _binned_genome = BinnedGenome.compute(
            _genome=GenomeIndex.get_genome(self.genome_name),
            bin_size=DEFAULT_BIN_SIZE,
            count_cap=DEFAULT_COUNT_CAP,
            pattern=DEFAULT_PATTERN,
        )

        simulation = SimulationForFragmentLen.init(
            _binned_genome=_binned_genome,
            p_x=_binned_genome.p_x,
            genome_len=_binned_genome.genome_len,
            bin_size=_binned_genome.bin_size,
            likelihood=BinnedOGMData.compute_likelihood(),
            num_trials=self.num_trials,
        )
        self._simulation = simulation
        lens = np.geomspace(*self.lims, num=self.num_simulate_lens)
        theory_df = self._get_theory(lens)

        df = theory_df
        df = df.loc[filter_uniformly(theory_df, by='p_err', n=self.num_simulate_lens)]

        simulation = simulation.replace(
            simulate=True,
            verbose=2,
        )
        sim_df = self._swipe(simulation, 'molecule_len', df['molecule_len'])

        self.sim_df = sim_df
        self.theory_df = theory_df

        if self._ogm_data is not None:
            self.exp_df = self._ogm_data.df
            self.exp_theory_df = self._get_theory(self.exp_df['molecule_len'].values)

    def _get_theory(self, lens):
        base = self._simulation.replace(
            molecule_len=lens,
            simulate=False,
            verbose=2,
        )
        base._compute()
        theory_df = pd.DataFrame(vars(base), columns=['molecule_len', 'p_err'])
        return theory_df

    def plot_theory(self):
        df = self.theory_df
        df = self._filter_df(df)
        x = 'molecule_len'
        plt.plot(df[x], df['p_err'], label=THEORY, alpha=1, marker='', ls='-', lw=3)

        df = self.sim_df
        df = self._filter_df(df)

        alpha = 1
        ci_alpha = .3

        plt.plot(df[x], df['error_rate'], label=SIMULATION, alpha=alpha, color='C1', marker='.', ls='')

        if self.show_ci:
            plot_ci(df[x], df['error_count'], df['num_trials'], alpha=ci_alpha, color='C1', ls='-')

        plt.xlabel(MOLECULE_LENGTH)
        plt.ylabel(ERROR_PROBABILITY)

        if self.exp_df is not None:
            df = self.exp_df
            df = self._filter_df(df)
            plt.plot(df[x], df['error_rate'], label=EXPERIMENTAL, alpha=alpha, marker='x', ls='', color='C2')
            plot_ci(df[x], df['error_count'], df['num_trials'], alpha=ci_alpha, color='C2', ls='-')

        plot_grid()
        plt.xscale('log')

    def _filter_df(self, df):
        df = df[df['molecule_len'] <= PLOT_LEN_LIM]
        return df

    def plot_comparison(self):
        df = self.sim_df
        df = self._filter_df(df)
        x_eq = df['p_err'].iloc[[0, -1]]
        plt.plot(x_eq, x_eq, '--k', alpha=.5)
        plt.plot(df['p_err'], df['error_rate'], label=SIMULATION, alpha=.8, color='C1', marker='.', ls='')
        if self.show_ci:
            plot_ci(df['p_err'], df['error_count'], df['num_trials'], alpha=.5, color='C1', ls='-')
        plt.xlabel(f'{ERROR_PROBABILITY} ({THEORY})')
        plt.ylabel(f'{ERROR_PROBABILITY}')

        if self.exp_df is not None:
            df = self.exp_df
            df = self._filter_df(df)
            theory_df = self.exp_theory_df
            theory_df = self._filter_df(theory_df)

            x = theory_df['p_err']
            plt.plot(x, df['error_rate'], label=EXPERIMENTAL, alpha=.8, color='C2', marker='x', ls='')

            if self.show_ci:
                plot_ci(x, df['error_count'], df['num_trials'], alpha=.5, color='C2', ls='-')
        plot_grid()
        # plt.xscale('log')
        # plt.yscale('log')


class PErrVsChannelModel(Evaluation):
    def __init__(self):
        super().__init__()
        self.run_name = ''
        self.optimal_val = None
        self.attr = 'bin_size'
        self.vals = None
        self.genome_name = HUMAN_GENOME
        self._ogm_data = None
        self.verbose = 1

    @property
    def name_prefix(self):
        return f'{self.attr}'

    def _compute(self):
        molecules_data = OGMData.get_molecules_data()
        theory = TheoryForVaryingChannelModel.init(
            _genome=GenomeIndex.get_genome(self.genome_name),
            molecule_len=DEFAULT_MOLECULE_LEN_PER_GENOME[self.genome_name],
            pattern=DEFAULT_PATTERN,
            _ogm_data=molecules_data
        )

        df = self._swipe(theory, self.attr, self.vals)

        self.optimal_bin_size = df[self.attr].iloc[df['p_err'].argmin()]
        self.df = df

    def _plot(self):
        df = self.df
        x = self.attr
        df = df.sort_values(x)

        plt.figure(figsize=(FIG_SIZE, FIG_SIZE))
        plt.xlabel(CHANNEL_MODEL_ATTR[self.attr])
        plt.ylabel(ERROR_PROBABILITY)
        plt.plot(df[x], df['p_err'], ls='--', lw=.5, marker='.')
        plt.xscale('log')
        plot_grid()


def bacterial_genomes_table(**kwargs):
    genome = GenomeIndex.get_genome(BACTERIAL_GENOMES)
    df = genome.seq_df[[ORGANISM_NAME, ACCESSION, 'seq_id', 'seq_len']].copy()
    df.columns = ['Organism', 'Accession', 'Sequence ID', 'Sequence Length']
    df = df.sort_values('Organism').reset_index(drop=True)
    display(df)
    return df.style.to_latex()


def estimated_distributions_table(**kwargs):
    data = BinnedOGMData.compute(
        _ogm_data=OGMData.get_molecules_data(),
        verbose=1,
    )
    likelihood = data.likelihood
    binned_genome = BinnedGenome.compute(
        _genome=GenomeIndex.get_genome(HUMAN_GENOME),
        verbose=1,
    )
    p_x = binned_genome.p_x
    p_xy = likelihood
    p_x = p_x
    assert p_xy.shape == (3, 3)
    assert p_x.shape == (3,)

    df = pd.concat([
        pd.DataFrame(p_xy),
        pd.DataFrame(p_x),
    ], axis=1, ignore_index=True)
    df.columns = ['$y=0$', '$y=1$', '$y=2$', '$p_x$']
    df.index = ['$x=0$', '$x=1$', '$x=2$']

    df = df.rename_axis(columns='$p_{y|x}$')
    df = df.style.format(precision=5)

    display(df)
    return df.to_latex(
        hrules=True,
        column_format='|l|lll|l|',
    )


def dna_fragment_illustration_figure(**kwargs):
    set_seed()
    molecules_data = OGMData.get_molecules_data()
    mol = BinnedFragment.compute(
        ogm_data_item=molecules_data.molecules_df.iloc[3],
    )
    offset = 0
    image = mol.image[0, :, offset:offset + 150]
    plot_image_illustration(image, mol.locs, mol.ref_segment)


def p_err_vs_fragment_len_figure(**kwargs):
    figs = {
        genome_name: PErrVsFragmentLen.make(
            genome_name=genome_name,
            _ogm_data=OGMData.get_molecules_data() if genome_name == HUMAN_GENOME else None,
            **kwargs
        )
        for genome_name in GENOME_NAMES
    }
    make_subplots(figs, legend=True)


def p_err_vs_pattern_figure(**kwargs):
    figs = {
        genome_name: PErrVsPattern.make(
            genome_name=genome_name,
            **kwargs,
        )
        for genome_name in GENOME_NAMES
    }
    make_subplots(figs, legend=False)


def p_err_vs_pattern_tables(export_dir, **kwargs):
    tables = {
        genome_name: PErrVsPattern.make(
            genome_name=genome_name,
            plot=0,
            load=1,
            **kwargs,
        ).data_table()
        for genome_name in GENOME_NAMES
    }
    for genome_name, df in tables.items():
        csv_file = f'{genome_name}_p_err_vs_pattern.csv'
        file_path = Path(export_dir) / csv_file
        print(f'Exporting {genome_name} to {file_path}')
        df.to_csv(file_path)


def p_err_vs_bin_size_figure(**kwargs):
    plt.figure()
    PErrVsChannelModel.make(
        attr='bin_size',
        vals=np.geomspace(16, 8192, NUM_BIN_SIZES).astype(int), **kwargs
    )


def p_err_vs_align_length_figure(**kwargs):
    PErrVsChannelModel.make(
        attr='align_length',
        vals=np.geomspace(10e3, 400e3, 32).astype(int),
        **kwargs
    )

