import operator
import pickle

import numpy as np
import pandas as pd
from numpy import set_printoptions
from sklearn.model_selection import train_test_split

import choosing_targets
import descriptive_stats
import generating_random_conf
import machines
import preparing_data

set_printoptions(precision=6)

TEST_SIZE = .25


def get_data(path, datafile_name, col1, col2, output_name):
    try:
        x, y = preparing_data.read_xy(output_name)
        print('Loaded!')
    except FileNotFoundError:
        x, y = preparing_data.main(path, datafile_name, output_name)
    target = choosing_targets.getting_target(y, col1, col2)
    target = np.ravel(target)
    return x, target


def check_minimum_presence_parameter(x, y):
    temp_x, x_test, temp_y, temp_y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=10)
    for col in x_test.columns:
        std = np.std(x_test[col], ddof=1)
        mu = np.mean(x_test[col])
        if mu == 0 or mu is np.nan:
            cv = 0
        else:
            cv = std / mu * 100
        if cv < 5:
            print(f'Dropping {col} from X table...')
            x.drop(col, inplace=True, axis=1)
    return x, y


def main(path, datafile_name, col1, col2, param_size):
    output_name = f'{col1[0]}_{col1[1]}_{col2[0]}_{col2[1]}_{param_size}_{datafile_name}'

    x, y = get_data(path, datafile_name, col1, col2, output_name)
    x, y = check_minimum_presence_parameter(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=10)
    # Running model
    models = machines.run_classifiers(x_train, x_test, y_train, y_test)

    # # Generating random configuration data to test against optimal results
    r = generating_random_conf.compound(x_train, param_size)
    print('Generated expanded configuration dataset')

    # Predicting results using machine on generated set of random parameters
    results = dict()
    for key in models.keys():
        yr = machines.predict(models[key], r[x.columns.tolist()])
        print('Sum of ones {}: {}'.format(key, yr.sum()))
        yr = pd.DataFrame({key: yr.tolist()})
        results[key] = [r, yr]
    # Output basic descriptive stats
    # Sending over X and Y as lists in a dictionary for current and each model
    current = {'current': [pd.concat([x_train, x_test], axis=0),
                           pd.concat([pd.DataFrame(y_train), pd.DataFrame(y_test)], axis=0, ignore_index=True)]}
    current.update(results)

    with open(f'output/results_data_{output_name}', 'wb') as f:
        pickle.dump(current, f)
    descriptive_stats.print_conf_stats(current, output_name)


if __name__ == "__main__":

    # p is the original results of the ABM model provided by the original modeler. It is 67 GB, thus we only
    # provide the pre_processed_data. However, all the code used in 'preparing_data.py' is here.
    p = ''
    # f'temp_' + {stats', 'firms', 'banks', 'construction' and 'regional'} are always saved
    file = 'temp_stats'
    sample_size = 1000000

    # Choice of targets include: 'gdp_index', 'unemployment', 'families_median_wealth', 'firms_profit',
    # 'gini_index', 'inflation', 'average_qli', 'house_price'

    target1 = 'gdp_index', 75, operator.gt
    target2 = 'gini_index', 25, operator.lt
    main(p, file, target1, target2, sample_size)
