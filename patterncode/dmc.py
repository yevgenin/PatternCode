import numpy as np
from scipy import stats


def error_probability_dmc(M, n, p_x, p_y_given_x):
    r"""
    computes
         \varepsilon = \Phi \left( \frac{log(M) - nI - \frac{1}{2} log(n)}{\sqrt{n V}} \right)
    where
        I = \sum_{x,y} p(x,y) log \frac{p(x,y)}{p(x)p(y)}
        V = \sum_{x,y} p(x,y) \left( log \frac{p(x,y)}{p(x)p(y)} - I \right)^2

    :param M: codebook size
    :param n: codeword length
    :param p_x: prior
    :param p_y_given_x: likelihood
    :return: I, V, \varepsilon
    """
    p_xy = p_y_given_x * p_x[:, None]
    p_xy = p_xy / p_xy.sum()
    p_y = p_xy.sum(axis=0)
    p_x = p_x[:, None]
    p_y = p_y[None, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        p_ratio = np.log(p_xy / (p_x * p_y))

    p_ratio = np.where((p_xy == 0) | (p_x == 0) | (p_y == 0), 0, p_ratio)

    I = (p_xy * p_ratio).sum()
    V = (p_xy * p_ratio ** 2).sum() - I ** 2

    phi = stats.norm.cdf
    p_err = phi((np.log(M) - n * I - .5 * np.log(n)) / np.sqrt(n * V))
    return I, V, p_err
