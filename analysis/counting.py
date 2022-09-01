import pandas as pd

import groups_cols
from groups_cols import abm_dummies as dummies
from groups_cols import abm_params as params


def getting_counting(data, name):
    """ Produces a csv with information regarding each dummy, i.e., when the dummy is active (=1).
    In the final csv we have three columns: sample size, optimal and non-optimal.
    All columns are in percentage of the total: sample size in relation to the whole sample, and optimal and
    non-optimal in relation to the sample of that specific dummy.

    For example, policies_buy has a sample size of 0.12 meaning that 12% of the sample had that dummy as true, an
    optimal 2% and a non-optimal of 97%, meaning that, when that dummy is active and policy used is buy, 97% of the
    samples fall under the non-optimal category.

    :param data: base csv
    :param name: name of the file
    :return: returns nothing, but saves the csv
    """
    table = pd.DataFrame(columns=['size', 'optimal', 'non_optimal', 'optimal_count', 'non_optimal_count'])
    for key in dummies:
        for each in dummies[key]:
            sample_size = len(data[data[each] == 1])/len(data)
            optimal = len(data[(data[each] == 1) & (data['Tree'] == 1)])
            non_optimal = len(data[(data[each] == 1) & (data['Tree'] == 0)])
            total = optimal + non_optimal
            print(f'{each}: size {sample_size:.04f}: optimal {optimal/total:.0f}: '
                  f'non-optimal {non_optimal/total:.04f}: optimal_count {optimal} non-optimal_count {non_optimal}')
            table.loc[each, 'size'] = sample_size
            table.loc[each, 'optimal'] = optimal/total
            table.loc[each, 'non_optimal'] = non_optimal/total
            table.loc[each, 'optimal_count'] = optimal
            table.loc[each, 'non_optimal_count'] = non_optimal
    table.to_csv(f'../pre_processed_data/counting_{name}.csv', sep=';')


# Parameters analysis
def coefficient_variation_comparison(simulated, ml):
    """ This function compares the ABM simulated results to the ML surrogate results in order to identify the
    differences between the two methods. How much of the cases fall under the optimal in relation to the mean?
    Added the column difference

    Using standard-score: (optimal value mean - full sample mean) / full sample standard-deviation

    Added absolute optimal value for simulated and ML

    :param simulated: the simulated database in csv
    :param ml: the ML surrogate database in csv
    :return: returns nothing, but saves the csv
    """
    table = pd.DataFrame(columns=['simulated_optimal', 'ml_optimal', 'difference'])
    for param in params:
        sim_mean = simulated[param].mean()
        sim_std = simulated[param].std()
        sim_optimal_mean = simulated[simulated['Tree'] == 1][param].mean()
        ml_mean = ml[param].mean()
        ml_std = ml[param].std()
        ml_optimal_mean = ml[ml['Tree'] == 1][param].mean()
        print(f'{param}: {(sim_optimal_mean - sim_mean) / sim_std:.06f}')
        print(f'{param}: {(ml_optimal_mean - ml_mean) / ml_std:.06f}')
        table.loc[param, 'simulated_optimal'] = (sim_optimal_mean - sim_mean) / sim_std
        table.loc[param, 'ml_optimal'] = (ml_optimal_mean - ml_mean) / ml_std
        table.loc[param, 'difference'] = table.loc[param, 'simulated_optimal'] - table.loc[param, 'ml_optimal']
        table.loc[param, 'abs_sim_optimal'] = sim_optimal_mean
        table.loc[param, 'abs_ml_optimal'] = ml_optimal_mean
    table.to_csv(f'../pre_processed_data/parameters_comparison.csv', sep=';')
    table.reset_index(inplace=True)
    table['Parameters'] = table['index'].map(groups_cols.abm_params_show)
    to_latex = table[['Parameters', 'abs_sim_optimal', 'abs_ml_optimal']]
    to_latex = to_latex.sort_values(by='Parameters')
    to_latex.set_index('Parameters', inplace=True)
    to_latex.to_latex('../pre_processed_data/parameters_comparison_latex.txt',
                      float_format="{:0.3f}".format)


# Parameters analysis
def normalize_and_optimal(simulated, ml):
    table = pd.DataFrame(columns=['z_simulated_optimal', 'z_ml_optimal'])
    for param in params:
        # normalize
        simulated.loc[:, f'n_{param}'] = (simulated[param] - simulated[param].min()) / \
                                         (simulated[param].max() - simulated[param].min())
        ml.loc[:, f'n_{param}'] = (ml[param] - ml[param].min()) / (ml[param].max() - ml[param].min())
        sim_optimal_mean = simulated[simulated['Tree'] == 1][f'n_{param}'].mean()
        ml_optimal_mean = ml[ml['Tree'] == 1][f'n_{param}'].mean()
        print(f'{param}: {sim_optimal_mean:.06f}')
        print(f'{param}: {ml_optimal_mean:.06f}')
        table.loc[param, 'z_simulated_optimal'] = sim_optimal_mean
        table.loc[param, 'z_ml_optimal'] = ml_optimal_mean
        table.loc[param, 'difference'] = sim_optimal_mean - ml_optimal_mean
        table.loc[param, 'abs_difference'] = abs(sim_optimal_mean - ml_optimal_mean)
    table.to_csv(f'../pre_processed_data/parameters_norm_optimal.csv', sep=';')
    table.reset_index(inplace=True)
    table['Parameters'] = table['index'].map(groups_cols.abm_params_show)
    to_latex = table[['Parameters', 'z_simulated_optimal', 'z_ml_optimal']]
    to_latex = to_latex.sort_values(by='Parameters')
    to_latex.set_index('Parameters', inplace=True)
    to_latex.to_latex('../pre_processed_data/parameters_norm_optimal_latex.txt',
                      float_format="{:0.3f}".format)


if __name__ == '__main__':
    th = pd.read_csv('../output/Tree_gdp_index_75_gini_index_25_1000000_temp_stats.csv', sep=';')
    c = pd.read_csv('../output/current_gdp_index_75_gini_index_25_1000000_temp_stats.csv', sep=';')
    c.rename(columns={'0': 'Tree'}, inplace=True)
    getting_counting(th, 'Tree')
    getting_counting(c, 'Current')
    coefficient_variation_comparison(c, th)
    normalize_and_optimal(c, th)

