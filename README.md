# Oboe_vs_auto-sklearn
Experiment bench to compare performance of Oboe to auto-sklearn


### Installation
* Clone this repository into a directory of your choice.
* Clone the Oboe repository into a directory of your choice (such as at the same level as this one).
* Satisfy the package dependencies. You need at least Python 3.6. This is most easily done as follows:

    `pip install --no-cache-dir openml`
    
    `pip install --no-cache-dir auto-sklearn` (these two use slightly different versions of sklearn, so install them in this order)

* Prepare your `experiment_config.py` file. You can use the `config_example.py` file as a starting point. If you want to, you can define different paths, such as placing your datasets in a scratch directory for large files. **Paths should be absolute.** 

### Usage
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
