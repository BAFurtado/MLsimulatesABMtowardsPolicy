import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def comparing_means(a, b, equal_var=False):
    """
    This is a test for the null hypothesis that 2 independent samples have identical average (expected) values.
    This test assumes that the populations have identical variances by default.
    equal_varbool, optional

    equal_var
    If True (default), perform a standard independent 2 sample test that assumes equal population variances [1].
    If False, perform Welch’s t-test, which does not assume equal population variance [2].

    f the p-value is smaller than our threshold, then we have evidence against the null hypothesis of
    equal population means. Ou seja, pequeno: médias diferentes. grande: médias iguais estatisticamente

    :return: statistics and p-value
    """
    return ttest_ind(a, b, equal_var=equal_var)


def normalize(data, col, output):
    inter_quartile = (data[col].max() - data[col].min())
    if inter_quartile == 0:
        output.loc[col, 'no_variance'] = True
        return data[col]
    else:
        return (data[col] - data[col].min()) / inter_quartile


def get_optimal(data, col):
    return data[data['Tree'] == 1][col]


def normalize_optimal(sim, ml, col, output, norm=True):
    if norm:
        sim.loc[:, col] = normalize(sim, col, output)
        ml.loc[:, col] = normalize(ml, col, output)
    sim = get_optimal(sim, col)
    ml = get_optimal(ml, col)
    return sim, ml, output


def optimal_non(ml, col, output, norm=True):
    if norm:
        ml.loc[:, col] = normalize(ml, col, output)
    optimal = ml[ml['Tree'] == 1][col]
    non_optimal = ml[ml['Tree'] == 0][col]
    return optimal, non_optimal, output


def plot_density(sim, ml, col, output):
    fig, ax = plt.subplots()
    try:
        sim.plot.density(ax=ax, label='ABM simulated', color='blue')
    except np.linalg.LinAlgError:
        output.loc[col, 'matrix_singular'] = True
    try:
        ml.plot.density(ax=ax, label='ML surrogate', color='red')
    except np.linalg.LinAlgError:
        # If any two rows are the same the matrix is singular...
        output.loc[col, 'matrix_singular'] = True
    else:
        ax.legend(frameon=False)
        plt.title(col)
        p_value = comparing_means(sim, ml)[1]
        output.loc[col, 'p_value'] = p_value
        if p_value < .001:
            output.loc[col, 'reject'] = 'different means'
            # plt.savefig(f'{col}_different.png')
        else:
            output.loc[col, 'reject'] = 'equal means'
            # plt.savefig(f'plots/{col}_equal.png')
        plt.close()
    return output


def main(sim, ml):
    out = pd.DataFrame(columns=['p_value', 'reject', 'matrix_singular'])
    for n in [True, False]:
        for p in sim.columns:
            s, m, out = normalize_optimal(sim, ml, p, out, norm=n)
            if n:
                out = plot_density(s, m, p, out)
        out.to_csv(f'means_comparison_output_norm{n}.csv', sep=';')
    out2 = pd.DataFrame(columns=['p_value', 'reject', 'matrix_singular'])
    for n in [True, False]:
        for p in ml.columns:
            opt, non, out2 = optimal_non(ml, p, out2, norm=n)
            p_value = comparing_means(opt, non)[1]
            out2.loc[p, 'p_value'] = p_value
            if p_value < .001:
                out2.loc[p, 'reject'] = 'different means'
            else:
                out2.loc[p, 'reject'] = 'equal means'
        out2.to_csv(f'means_comparison_optimal_non_optimal_norm{n}.csv', sep=';')


def different(file1, file2):
    output = list()
    for param in file1.param:
        if all(file1.loc[file1['param'] == param, 'reject'] == 'different means'):
            if all(file2.loc[file2['param'] == param, 'reject'] == 'different means'):
                output.append(param)
    return output


if __name__ == '__main__':
    th = pd.read_csv('../output/Tree_gdp_index_75_gini_index_25_1000000_temp_stats.csv', sep=';')
    c = pd.read_csv('../output/current_gdp_index_75_gini_index_25_1000000_temp_stats.csv', sep=';')
    c.rename(columns={'0': 'Tree'}, inplace=True)
    main(c, th)
