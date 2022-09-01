import operator
import os

import numpy as np
import pandas as pd

from preparing_data import read_xy


def selecting_y(df, col):
    # Selects only results from last month of simulation
    return df[col]


def customizing_target(base, percentile=65, op=operator.gt):
    # Discrete results for a given percentile and a given operator (greater than or less than)
    return pd.DataFrame({'target': [1 if op.__call__(x, np.percentile(base, percentile)) else 0 for x in base]})


def averaging_targets(df1, df2):
    # Summarizes two target columns into one when both results are one
    return pd.DataFrame({'target': [1 if x == 1 and y == 1 else 0 for x, y in zip(df1['target'], df2['target'])]})


def getting_target(y, col1, col2):
    name = f'pre_processed_data/{col1[0]}_{col1[1]}_{col2[0]}_{col2[1]}.csv'
    if os.path.exists(name):
        print('Loading existing targets...')
        return pd.read_csv(name, sep=';')
    first_col = customizing_target(selecting_y(y, col1[0]), col1[1], col1[2])
    second_col = customizing_target(selecting_y(y, col2[0]), col2[1], col2[2])
    target = averaging_targets(first_col, second_col)
    target.to_csv(name, sep=';', index=False)
    return target


if __name__ == '__main__':
    data_x, data_y = read_xy('temp_stats')
    target1 = 'gdp_index', 65, operator.gt
    target2 = 'gini_index', 35, operator.lt
    t = getting_target(data_y, target1, target2)
