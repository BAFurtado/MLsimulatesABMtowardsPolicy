""" Prepare the simulation data for Machine Learning procedures 
"""

import json
import os

import pandas as pd

from utils import cols_specification as cols


def read_conf_results_files(general_path, config_name='temp_stats'):
    # Walks over directory collecting all files representing each simulation results
    return [os.path.join(dirpath, f)
            for dirpath, dirnames, files in os.walk(general_path)
            for f in files if f.startswith(config_name)
            if os.path.exists(os.path.join(os.path.dirname(dirpath), 'conf.json'))]


def associate_config_file(files):
    # Walks over directory collecting all configuration files associated with each simulation results
    return [os.path.join(os.path.dirname(os.path.dirname(f)), 'conf.json')
            for f in files
            if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(f)), 'conf.json'))]


def process_each_file(file_name, output_list, config_list):
    x = pd.DataFrame()
    # Provides names for the columns of results of simulation
    col_names = cols.OUTPUT_DATA_SPEC[file_name.replace('temp_', '')]['columns']
    y = pd.DataFrame(columns=col_names)
    # Extract both parameters from conf.JSON files and results of that given simulation from 'avg' folder
    for i, each in enumerate(output_list):
        y_test = pd.read_csv(each, sep=';', header=None, names=col_names)
        # Testing minimum length of simulation
        if len(y_test) >= 120:
            y_test = y_test.tail(1)
            y_test.drop('month', axis=1, inplace=True)
            y = y.append(y_test)
            x = x.append(json_to_dict(read_json(config_list[i])))
    return x, y


def read_json(path):
    # Interpret JSON file of configuration with simulation given parameters
    return json.load(open(path))


def json_to_dict(json_dict):
    # Transforms JSON data into DataFrame, removing unchanging columns, extracting data from list
    df = pd.DataFrame.from_dict(json_dict, orient='index').drop(labels='RUN', axis=0).dropna(axis=1)
    # Unpacking metropolis region info
    df['PROCESSING_ACPS'] = df['PROCESSING_ACPS'].apply(lambda x: x[0])
    return df


def last_month(df, month):
    return df[df.month == month].drop('month', axis=1)


def dummies(data):
    cat, num = list(), list()
    for i in data.columns:
        if data[i].dtype == object:
            cat.append(i)
        else:
            num.append(i)
    cat = data[cat]
    cat = pd.get_dummies(cat)
    num = data[num]
    return pd.concat([num, cat], axis=1)


def unique_cols(df):
    a = df.to_numpy()
    out = (a[0] == a).all(0)
    return out


def drop_columns(df):
    cols_to_drop = unique_cols(df)
    print(f'Dropping following cols...')
    for i, c in enumerate(df.columns):
        if cols_to_drop[i]:
            print(c)
            df.drop(c, inplace=True, axis=1)
    if 'SEED' in df.columns:
        df.drop('SEED', inplace=True, axis=1)
    return df


def save_xy(x, y, name):
    x.to_csv(f'pre_processed_data/x_{name}.csv', sep=';', index=False)
    y.to_csv(f'pre_processed_data/y_{name}.csv', sep=';', index=False)


def read_xy(output_name):
    return pd.read_csv(f'pre_processed_data/x_{output_name}.csv', sep=';'), \
           pd.read_csv(f'pre_processed_data/y_{output_name}.csv', sep=';')


def reading_saving_data(path, datafile_name, output_name):
    # Get list of files
    print('Reading configuration files...')
    list_of_files = read_conf_results_files(path, datafile_name)
    # Get associated conf.json files
    print('Reading data files...')
    list_of_conf_files = associate_config_file(list_of_files)
    print(f'Files of configuration are of the same size: {len(list_of_files) == len(list_of_conf_files)}')
    print('Processing files...')
    x, y = process_each_file(datafile_name, list_of_files, list_of_conf_files)
    # Drop columns if configuration is exactly the same for all runs
    x = drop_columns(x)
    x = dummies(x)
    print('Saving tables...')
    save_xy(x, y, output_name)
    return x, y


def main(path, datafile_name='temp_stats', output_name=''):
    x, y = reading_saving_data(path, datafile_name=datafile_name, output_name=output_name)
    return x, y
