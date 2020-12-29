import pandas as pd
import openml
from experiment_config import Config
from pathlib import Path

# Fetch the table of options
openml_datasets = openml.datasets.list_datasets()
openml_datasets = pd.DataFrame.from_dict(openml_datasets, orient='index')

# Filter
openml_datasets = openml_datasets[openml_datasets['NumberOfInstances'] >= 150]
openml_datasets = openml_datasets[openml_datasets['NumberOfInstances'] < 10000]
openml_datasets = openml_datasets[openml_datasets['NumberOfMissingValues'] == 0]  # no missing values please
openml_datasets = openml_datasets[openml_datasets['NumberOfClasses'] > 0]  # we want classification datasets

# Output as single-column csv
filename = Path(Config.DATASETS) / 'selected_OpenML_classification_dataset_indices.csv'
pd.DataFrame(openml_datasets.index, columns=None, index=None).to_csv(filename, header=False, index=False)