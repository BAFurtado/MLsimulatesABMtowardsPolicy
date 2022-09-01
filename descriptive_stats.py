import numpy as np
import pandas as pd

pd.options.display.float_format = '{:,.2f}'.format


def print_conf_stats(kwargs, name):
    # Dict contains X and Y as lists in a dictionary for current and each model
    df = pd.DataFrame()
    for key in kwargs.keys():
        kwargs[key][0].to_csv(f'output/{key}_{name}_0.csv', sep=';', index=False)
        kwargs[key][1].to_csv(f'output/{key}_{name}_1.csv', sep=';', index=False)
        try:
            temp1 = pd.DataFrame(index=np.arange(0, len(kwargs[key][0])),
                                 columns=[list(kwargs[key][0].columns) + [f'{key}_optimal']])
            n_cols = kwargs[key][0].shape[1]
            temp1.iloc[:, 0: n_cols] = kwargs[key][0]
            temp1.iloc[:, n_cols: n_cols + 1] = kwargs[key][1]
            temp1.to_csv(f'output/{key}_{name}.csv', sep=';', index=False)
        except MemoryError:
            print('MemoryError')
            continue
    df.to_csv(f'pre_processed_data/comparison_analysis_{name}.csv', sep=';', index=False, float_format='%.6f')
