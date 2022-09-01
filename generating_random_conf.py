import numpy as np
import pandas as pd
import scipy.stats as stats

import parameters_restriction as params

np.random.seed(0)


def to_dict_from_module():
    return {k: getattr(params, k) for k in dir(params) if not k.startswith('_')}


def pre_process(data):
    return data.describe().T[['mean', 'std']]


def compound(x, n):
    # Function will break if n is too small < 1000
    # so that not enough columns will be sampled for PROCESSING_ACPS, for example
    param = to_dict_from_module()
    samples = pre_process(x)
    df = pd.DataFrame()
    data = dict()
    # Either choice or normal
    for p in param:
        if p in samples.index:
            if param[p]['distribution'] == 'normal':
                lower, upper = param[p]['min'], param[p]['max']
                mu, sigma = samples.loc[p, 'mean'], samples.loc[p, 'std'] * 3
                data[p] = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(n)
        else:
            choices = [i for i in samples.index if p in i]
            if choices:
                m = len(choices)
                choices_vector = pd.DataFrame(np.random.choice(choices, n, p=[1 / m] * m))
                temp = pd.get_dummies(choices_vector)
                temp.columns = choices
                df = pd.concat([temp, df], axis=1)
    return pd.concat([df, pd.DataFrame(data)], axis=1)
