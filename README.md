# Oboe_vs_auto-sklearn
Experiment bench to compare performance of Oboe to auto-sklearn


## Installation
* Clone this repository into a directory of your choice.
* Clone the Oboe repository into a directory of your choice (such as at the same level as this one).
* Satisfy the package dependencies. You need at least Python 3.6. This is most easily done as follows:

    `pip install --no-cache-dir openml`
    
    `pip install --no-cache-dir auto-sklearn` (these two use slightly different versions of sklearn, so install them in this order)

* Prepare your `experiment_config.py` file. You can use the `config_example.py` file as a starting point. If you want to, you can define different paths, such as placing your datasets in a scratch directory for large files. **Paths should be absolute.** 

## Usage
### Basics
The basic script here is `run_one_test.py`, which runs one k-fold test. It **requires** the following parameters:
* `-c`, `--collection`: which collection of datasets to use (which must be a subdirectory of the `DATASETS` directory defined in `experiment_config.py`). For example `OpenML_classification`.
* `-d`, `--dataset`: which dataset to use, for example `3` to select `dataset_3_features_and_labels.csv` from the collection defined above.
* `-f`, `--framework`: choice of `Oboe` or `auto-sklearn`
* `-r`, `--runtime`: how many seconds of runtime per fold.

There are also optional parameters:
* `-k`, `--kfolds`: how many folds to use in cross-validation, default is 1.
* `-n`, `--njobs`: how many cores to task for this test, default is 1. For small datasets/low runtimes, multiple cores may actually produce more overhead than they're worth.
* `-q`, `--quiet` (flag): suppresses (some) warnings

A typical invocation would look like this:

`python run_one_test -c OpenML_classification -d 3 -f auto-sklearn -r 64`

The results will be placed in the `RESULTS/COLLECTION/FRAMEWORK/NJOBS/` directory for your test. All runtimes for the same (collection, framework, njobs, dataset) combination will be appended to the same CSV file. All experiments using the same (collection, framework, njobs) will appended to the same finished.txt and errors.txt file. The files are named as follows:
* `dataset_D_FRAMEWORK_njobs_N.csv`
* `finished.txt`
* `errors.txt`

### Selecting datasets for a collection
You can define a new OpenML classification dataset with `select_OpenML_classification_datasets.py`. It takes three arguments:
* `-c`, `--collection`: the name of the new collection you're going to make
* `-m`, `--minimum`: minimum number of instances for the datasets
* `-M`, `--Maximum`: maximum number of instances for the datasets

Only datasets with class labels and no missing values will be selected. The results will be placed in a file `selected_COLLECTION_classification_dataset_indices.csv`.

A typical invocation lookes like this:

`python select_OpenML_classification_datasets.py -c OpenML_midsize_classification -m 150 -M 9999`

### Fetching and preprocessing datasets for a collection

You can fetch one or more datasets from the OpenML API into a collection directory with `fetch_preprocess_datasets.py`. It takes one collection argument:

* `-c`, `--collection`: name of the collection to send the dataset(s) to.

Also it takes *one* of the following arguments to specify datasets:

* `-i`, `--individual`: the index of an individual dataset to fetch.
* `-f75`, `--first75`: a selection of 75 datasets based on the experiments in the original paper. All these datasets are 150 <= instances < 10000 in size, classification, no missing values. Of the 79 datasets in the experimental set in the original paper, we remove 312, 316, 373, 390 for being problematic. This leaves us with the following datasets: 3, 11, 12, 14, 16, 18, 20, 22, 23, 28, 30, 31, 36, 37, 39, 40, 41, 43, 44, 46, 48, 50, 53, 54, 59, 60, 61, 181, 182, 183, 187, 285, 300, 307, 311, 313, 329, 333, 334, 335, 336, 337, 338, 375, 377, 383, 384, 385, 386, 387, 388, 389, 391, 392, 393, 394, 395, 396, 397, 398, 400, 401, 446, 450, 458, 463, 464, 469, 475, 679, 694, 715, 717, 718, 720.
* `-s`, `--selection`: use a selection file. This must be a single-column CSV of dataset indices located inside the DATASETS directory as defined by `experiment_config.py`.

A typical invocation looks like this:

`python -c OpenML_midsize_classification -s selected_OpenML_midsize_classification_dataset_indices.csv`

#### Preprocessing
Datasets will be fetched, then preprocessed by one-hot encoding of categorical variables and normalizing around a zero mean, unit variance.

### Slurm jobs
We performed our experiments on the ALICE server of Leiden University, which uses Slurm as a scheduler. Use of the Slurm jobs is not required but makes large experiments easier. This code could be adapted to other high performance systems as needed without having to change the underlying scripts.

Basic usage is as follows: run `create_classification_slurm_jobs` specifying a collection to prepare jobs for, then run `enqueue_COLLECTION_njobs_N.sh` to submit the jobs to Slurm. The following parameters are available:

* `-c`, `--collection`: **required** Name of the collection to prepare a batch for. 
* `-n`, `--njobs`: number of cores to use per job, default is 1.
* `-m`, `--memory_per_core`: amount of RAM memory to reserve per core. Default is 10GB.
* `-k`, `--kfolds`: Number of folds for k-fold cross validation. Default is 5. 

One job will be generated for every combination of (framework, dataset, runtime), using all the datasets in the collection directory and all runtimes in the `experiment_config.py`.

Expected overall runtime for each dataset job is estimated as `k * r * 2`, where `k` is the number of folds and `r`1 is the runtime budget for the framework inside one fold. For example, with `k=5` the job for dataset 3 with runtime budget 32 is estimated to take 5 * 32 * 2 = 320 seconds.

Calling the *enqueue* bash script submits all these jobs to the Slurm scheduler at once, but each with a reasonable estimate of maximum runtime. By keeping all the jobs relatively small, we allow the scheduler to efficiently allocate resources.

### Plotting

[comment]: <> (TODO)