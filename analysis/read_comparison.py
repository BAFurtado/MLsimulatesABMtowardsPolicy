import numpy as np
import pandas as pd

import groups_cols

# Produce IQR (interquartile) table from output of random forest (Tree)


if __name__ == '__main__':
    """
    IMPORTANT
    1. You have to run main.py to produce the output file.
    2. Alternatively, you can download it (500 MB) from here:
    https://drive.google.com/file/d/1xsKQQr5EGxv7iI1esF6cjvaNk8Hlo7QO/view?usp=sharing
    
    """

    csv = pd.read_csv('../output/Tree_gdp_index_75_gini_index_25_1000000_temp_stats.csv', sep=';')
    # The Tree column means optimal: 0 is non-optimal and 1 is optimal

    # We separate and aggregate, per characteristic and especially by ACP -->
    #   for each one we have to get median, q3, q1

    df = pd.DataFrame()

    acps = ['all'] + groups_cols.abm_dummies['acps']

    for ACP in acps:

        if ACP == 'all':
            plot_df = csv
        else:
            plot_df = csv.loc[csv[ACP] == 1]

        for pol in groups_cols.abm_dummies['policies']:

            if ACP == 'all':
                # if we are dealing with all acps, we just need to locate the rows in which the pol is the one being
                # analyzed
                results_df = csv.loc[csv[pol] == 1]
                optimal_count = csv.loc[(csv[pol] == 1) &
                                        (csv['Tree'] == 1)].count().values[0]
            else:
                results_df = csv.loc[(csv[ACP] == 1) & (csv[pol] == 1)]
                optimal_count = csv.loc[(csv[ACP] == 1) &
                                        (csv[pol] == 1) &
                                        (csv['Tree'] == 1)].count().values[0]

            q3, q1 = np.percentile(results_df['Tree'], [75, 25])
            median, mean, count, std = results_df['Tree'].median(), results_df['Tree'].mean(), results_df[
                'Tree'].count(), np.std(results_df['Tree'])

            df = df.append(
                {'ACP': ACP,
                 'pol': pol,
                 'q3': q3,
                 'q1': q1,
                 'median': median,
                 'IQR': q3 - q1,
                 'mean': mean,
                 'std': std,
                 'count': count,
                 'optimal_count': optimal_count},
                ignore_index=True)

    df.to_csv('IQR.csv', sep=';')
