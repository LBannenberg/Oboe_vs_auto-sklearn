from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import openml
import scipy.sparse as sps
from sklearn.preprocessing import scale
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer as Imputer

from experiment_config import Config

parser = argparse.ArgumentParser(description='Fetch one of the selected datasets')
parser.add_argument('-c', '--collection', type=str, default='OpenML_classification', help='Which dataset collection to put this in?')
method = parser.add_mutually_exclusive_group(required=True)  # Specify selection:
method.add_argument('-i', '--individual', type=int, default=0, help='Specify the id of a single dataset to preprocess.')
method.add_argument('-f75', '--first75', action='store_true', help='Use the first 75 (manually noted) qualifying sets.')
method.add_argument('-s', '--selection', type=str, default='', help='Use the selection found in the .csv file.')
args = parser.parse_args()

dataset_dir = Path(Config.DATASETS).resolve()
collection_dir = dataset_dir / args.collection
collection_dir.mkdir(parents=True, exist_ok=True)

if args.individual:
    selected_datasets = [args.individual]
elif args.first75:
    # Leaves out 4 of the 79 datasets selected by the other script because they're problematic.
    selected_datasets = [3, 11, 12, 14, 16, 18, 20, 22, 23, 28, 30, 31, 36, 37, 39, 40, 41, 43, 44, 46, 48, 50, 53, 54,
                         59, 60, 61, 181, 182, 183, 187, 285, 300, 307, 311, 313, 329, 333, 334, 335, 336,
                         337, 338, 375, 377, 383, 384, 385, 386, 387, 388, 389, 391, 392, 393, 394, 395,
                         396, 397, 398, 400, 401, 446, 450, 458, 463, 464, 469, 475, 679, 694, 715, 717, 718, 720]
    # Problematic, therefore left out: 312, 316, 373, 390
    # Numerical issues: 383, 384, 385, 386, 387, 388, 389, 391, 392, 393, 394, 395, 396, 397, 398, 400, 401
elif args.selection != '':
    selected_datasets = dataset_dir / args.selection
    selected_datasets = pd.read_csv(selected_datasets, index_col=None, header=None).values.T[0]
else:
    selected_datasets = []
    print('Error: no datasets were selected.')


def pre_process(raw_data, categorical, impute=True, standardize=True, one_hot_encode=True):
    """
    Pre-process one dataset.

    Args:
        raw_data (np.ndarray):    raw features of the n-by-d dataset, without indices and headings.
        categorical (list):       a boolean list of length d indicating whether each raw feature is categorical.
        impute (bool):            whether to impute missing entries or not.
        standardize (bool):       whether to standardize each feature or not.
        one_hot_encode (bool):    whether to use one hot encoding to pre-process categorical features or not.
    Returns:
        np.ndarray:               pre-processed dataset.
    """
    # list of pre-processed arrays (sub-portions of dataset)
    processed = []

    # whether to impute missing entries
    if impute:
        # if there are any categorical features
        if np.array(categorical).any():
            raw_categorical = raw_data[:, categorical]
            # impute missing entries in categorical features using the most frequent number
            imp_categorical = Imputer(missing_values='NaN', strategy='most_frequent', axis=0, copy=False)
            processed.append(imp_categorical.fit_transform(raw_categorical))

        # if there are any numeric features
        if np.invert(categorical).any():
            raw_numeric = raw_data[:, np.invert(categorical)]
            # impute missing entries in non-categorical features using mean
            imp_numeric = Imputer(missing_values='NaN', strategy='mean', axis=0, copy=False)
            processed.append(imp_numeric.fit_transform(raw_numeric))

        # data has now been re-ordered so all categorical features appear first
        categorical = np.array(sorted(categorical, reverse=True))
        processed_data = np.hstack(tuple(processed))

    else:
        processed_data = raw_data

    # one-hot encoding for categorical features (only if there exist any)
    if one_hot_encode and np.array(categorical).any():
        try:
            encoder = OneHotEncoder()  # LAU: TODO figure out how we can ensure that exactly the columns in "categorical" are targeted
            processed_data = encoder.fit_transform(processed_data)
        except:
            print(categorical)
            raise
        processed_data = processed_data.toarray()
        categorical = np.zeros(processed_data.shape[1], dtype=bool)

    # standardize all numeric and one-hot encoded categorical features
    if standardize:
        if isinstance(processed_data, pd.DataFrame):
            processed_data = processed_data.to_numpy()
        processed_data[:, np.invert(categorical)] = scale(processed_data[:, np.invert(categorical)])

    print('\tData pre-processing finished')
    return processed_data, categorical


for dataset_id in selected_datasets:
    try:
        dataset = openml.datasets.get_dataset(int(dataset_id))
        data_numeric, data_labels, categorical, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
        if sps.issparse(data_numeric):
            data_numeric = data_numeric.todense()

        # doing imputation and standardization and not doing one-hot-encoding achieves optimal empirical performances
        # (smallest classification error) on a bunch of OpenML datasets
        data_numeric, categorical = pre_process(raw_data=data_numeric, categorical=categorical,
                                                impute=False, standardize=True, one_hot_encode=True)

        # the output is a preprocessed dataset with all the columns except the last one being preprocessed features,
        # and the last column being labels
        data = np.append(data_numeric, np.array(data_labels, ndmin=2).T, axis=1)

        destination = collection_dir / f'dataset_{dataset_id}_features_and_labels.csv'
        pd.DataFrame(data, index=None, columns=None).to_csv(destination, header=False, index=False)
        print(f'dataset {dataset_id} finished')
    except Exception as e:
        print(f'Failure on dataset {dataset_id}')
        print(e)
