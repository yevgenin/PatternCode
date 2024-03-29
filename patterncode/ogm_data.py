from functools import lru_cache

import pandas as pd
from matplotlib import pyplot as plt

from patterncode.config import SHOW_CI, PROJECT_DATA_DIR, OGM_DATA_FILE, EXPERIMENTAL, MOLECULE_LENGTH, \
    ERROR_PROBABILITY
from patterncode.utils import Computation, plot_ci


class OGMData(Computation):
    correct_df: pd.DataFrame
    molecules_df: pd.DataFrame

    @staticmethod
    @lru_cache
    def get_molecules_data():
        return OGMData.compute()

    def __init__(self):
        super().__init__()
        self.show_ci = SHOW_CI
        self.data_file = PROJECT_DATA_DIR / OGM_DATA_FILE

    def _compute(self):
        data = pd.read_pickle(self.data_file)
        vars(self).update(data)

        correct = self.correct_df.groupby("len_bp")["correct"]
        num_successes = correct.sum()
        num_trials = correct.count()
        error_count = num_trials - num_successes
        error_rate = (error_count / num_trials)

        self.df = pd.DataFrame(
            {
                'molecule_len': list(error_rate.index),
                'error_count': error_count,
                'error_rate': error_rate,
                'num_trials': num_trials,
            }
        )

    def _plot(self, color='C2'):
        df = self.df

        x = 'molecule_len'
        plt.plot(df[x], df['error_rate'], label=EXPERIMENTAL, alpha=.8, marker='o', c=color, ls=':')

        if self.show_ci:
            plot_ci(df[x], df['error_count'], df['num_trials'], alpha=.5, color=color, ls='-')

        plt.xlabel(MOLECULE_LENGTH)
        plt.ylabel(ERROR_PROBABILITY)
        plt.xscale('log')
