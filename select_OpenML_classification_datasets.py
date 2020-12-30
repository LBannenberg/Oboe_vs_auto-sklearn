import argparse
import pandas as pd
import openml
from experiment_config import Config
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--collection', type=str, required=True, help='name of your new collection')
parser.add_argument('-m', '--minimum', type=int, required=True, help='minimum number of instances')
parser.add_argument('-M', '--Maximum', type=int, required=True, help='Maximum of instances')
args = parser.parse_args()

# Fetch the table of options
openml_datasets = openml.datasets.list_datasets()
openml_datasets = pd.DataFrame.from_dict(openml_datasets, orient='index')

# Filter
openml_datasets = openml_datasets[openml_datasets['NumberOfInstances'] >= args.minimum]
openml_datasets = openml_datasets[openml_datasets['NumberOfInstances'] <= args.Maximum]
openml_datasets = openml_datasets[openml_datasets['NumberOfMissingValues'] == 0]  # no missing values please
openml_datasets = openml_datasets[openml_datasets['NumberOfClasses'] > 0]  # we want classification datasets

# Output as single-column csv
filename = Path(Config.DATASETS) / f'selected_{args.collection}_dataset_indices.csv'
pd.DataFrame(openml_datasets.index, columns=None, index=None).to_csv(filename, header=False, index=False)