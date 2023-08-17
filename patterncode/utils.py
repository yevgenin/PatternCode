import random
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import copy
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from Bio import SeqIO
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from tqdm.auto import tqdm
import shutil
from patterncode.config import *


def set_seed(seed: int = SEED):
    print('Setting random seed to: ', seed)
    random.seed(seed)
    np.random.seed(seed)


def clear_cache():
    shutil.rmtree(CACHE_DIR)


def cached_func(func: T) -> T:
    return joblib.Memory(CACHE_DIR).cache(func)


def tqdm2(iterable, verbose=VERBOSE, verbose_threshold=1, **kwargs):
    yield from tqdm(iterable, leave=verbose < 2, disable=verbose_threshold < verbose, **kwargs)


def map2(func, items, desc=None, verbose=VERBOSE, parallel=PARALLEL_FLAG, max_workers=MAX_WORKERS):
    @contextmanager
    def _mapper():
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                yield executor.map
        else:
            yield map

    with _mapper() as _map:
        def _func(item_seed):
            item, seed = item_seed
            np.random.seed(seed)
            return func(item)

        items = list(items)
        random_seeds = np.random.randint(0, MAX_INT32, size=len(items))
        results = _map(_func, zip(items, random_seeds))
        yield from tqdm2(
            results, verbose=verbose, desc=desc, postfix=f'parallel: {parallel}',
            total=len(items),
        )


def report_dict(self):
    return {
        key: val for key, val in vars(self).items()
        if not key.startswith('_')
    }


def sample_uniformly(df: pd.DataFrame, by: str, n: int, log=False):
    """
    Sample dataframe by column `by` such that the values are uniformly distributed.

    :param df:  dataframe to sample
    :param by:  column to sample by
    :param n:   number of samples
    :param log: sample uniformly in log space
    :return:    sampled dataframe index
    """
    assert n is not None
    if len(df) > n:
        x = df[by].sort_values()

        if log:
            func = np.geomspace
        else:
            func = np.linspace

        pos = func(x.min(), x.max(), n)
        i = np.unique(x.searchsorted(pos))
        return x.iloc[i].index
    else:
        return df.index


def plot_ci(x, k, n, **kwargs):
    ci = proportion_confint(count=k, nobs=n, alpha=CONFIDENCE_ALPHA, method='beta')
    plt.vlines(x, *ci, **kwargs)


def plot_text_annotations(xs, ys, texts, color='k', distance=(15, 15), **kwargs):
    for i, (x, y, text) in enumerate(zip(xs, ys, texts)):
        plt.plot(x, y, '.', color=color)
        sign_i = 1 if i % 2 == 0 else -1
        kw = dict(
            xy=(x, y),
            xycoords='data',
            ha='center',
            va='center',
            color=color,
            xytext=sign_i * np.stack(distance),
            textcoords='offset points',
            fontweight='bold',
            alpha=1,
            arrowprops=dict(
                arrowstyle="-|>",
                linestyle='-',
                alpha=1,
                color=color,
            ),
            fontsize=6,
        ) | kwargs
        plt.annotate(text, **kw)


def label_ax(label):
    plt.title(label, loc='center', fontweight='bold')


@lru_cache
def read_human_genome(file=HUMAN_GENOME_FASTA_FILE):
    records = SeqIO.parse(file, 'fasta')
    sequences = [
        record.seq for record in tqdm(records, desc='Reading human genome')
        if record.id.startswith('NC_')
    ]
    return sequences


def add_subplot_seps():
    offset = .01
    for x in [1 / 3 + offset, 2 / 3 + offset]:
        margin = .05
        plt.plot([x, x], [margin, 1 - margin], color='k', ls=':', lw=.5, transform=plt.gcf().transFigure, clip_on=False)


class Computation(ABC):
    def __init__(self):
        super().__init__()
        self.run_name = RUN_NAME
        self.quick_run = QUICK_RUN_FLAG
        self._init_override()

    def _init_override(self):
        self.load_run_name = LOAD_RUN_NAME
        self.verbose = VERBOSE
        self.parallel = PARALLEL_FLAG
        self.plot = PLOT_COMPUTATIONS_FLAG
        self.load = LOAD_FLAG
        self.save = SAVE_FLAG
        self.show = False
        self.caption = False

    @property
    def name(self):
        return self.name_prefix

    @property
    def name_prefix(self):
        return type(self).__name__

    def replace(self, **kwargs):
        new = copy(self)
        new._set(**kwargs)
        return new

    def _set(self, **kwargs):
        assert set(kwargs) <= set(vars(self)), f'Unknown kwargs: {set(kwargs) - set(vars(self))}'
        vars(self).update(kwargs)

    @classmethod
    def init(cls, **kwargs):
        self = cls()
        self._set(**kwargs)
        return self

    @classmethod
    def compute(cls, **kwargs):
        self = cls.init(**kwargs)
        self._compute()
        return self

    @classmethod
    def make(cls, **kwargs):
        self = cls()
        try:
            self._make(**kwargs)
        except Exception as e:
            print(f'Error in {self.name}._make: {e}')
            raise

        return self

    def _compute(self):
        raise NotImplementedError

    def _show(self):
        data = pd.Series(report_dict(self)).dropna()
        display(data)

    def _plot(self):
        pass

    def _swipe(self, base, attr, vals, apply='_compute', desc=None, parallel=None) -> pd.DataFrame:
        """
        Swipe over values and apply method, returning a dataframe of results

        :param base: base object to copy
        :param attr: attribute to set
        :param vals: values to set
        :param apply: method to apply
        :param desc: description for tqdm
        :param parallel: parallel flag
        :return: dataframe of results
        """
        if parallel is None:
            parallel = self.parallel

        if desc is None:
            desc = self.name

        def _func(val):
            new = copy(base)
            if attr is not None:
                setattr(new, attr, val)
            getattr(new, apply)()
            return report_dict(new)

        # check that _func works
        _func(vals[0])

        # run swipe
        items = map2(_func, vals, desc=desc, parallel=parallel, verbose=self.verbose)
        return pd.DataFrame(items)

    def _load(self):
        """
        Load data from file, and set attributes
        """
        out_dir = Path(OUT_DIR)
        assert out_dir.exists(), out_dir

        if self.load_run_name is None:
            print('Loading latest')
            files = list(out_dir.glob(self.name + '*.pkl'))
            assert len(files) > 0, f'no files found: {self.name} in {out_dir}'
            file_path = sorted(files)[-1]
            self.file_name = file_path.stem
        else:
            self.file_name = self.name + '_' + self.load_run_name
            file_path = (out_dir / (self.file_name + '.pkl'))

        print('Loading:', file_path)
        assert file_path.exists(), 'file not found: ' + str(file_path)
        loaded_data = pd.read_pickle(file_path)
        assert isinstance(loaded_data, pd.Series), 'loaded data is not a Series'
        vars(self).update(dict(loaded_data))

        self.file_path = file_path

    def _save(self):
        """
        Save data to file
        """
        if self.run_name is None:
            self.run_name = RUN_NAME

        out_dir = Path(OUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.file_name = self.name + '_' + self.run_name
        file_path = out_dir / (self.file_name + '.pkl')

        print('Saving:', file_path)
        pd.Series(report_dict(self)).to_pickle(file_path)

    def _make(self, **kwargs):
        """
        Make data, optionally loading, saving, showing, and plotting
        """
        if DETERMINISTIC_MAKE:
            set_seed()
        self._set(**kwargs)

        compute = True
        if self.load:
            compute = False
            try:
                self._load()
            except AssertionError as err:
                print('Failed to load: ', err)
                compute = True
            else:
                self._init_override()
                self._set(**kwargs)

        if compute:
            self._compute()
            if self.save:
                self._save()

        if self.show:
            self._show()

        if self.plot:
            self._plot()


def plot_grid():
    plt.grid(True, which='both', ls='-', alpha=.5)
