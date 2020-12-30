import argparse
import time
import csv
import sys
from pathlib import Path

import warnings

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

# Get our overall experiment configuration
from experiment_config import Config

# Load auto-sklearn
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import balanced_accuracy

# Load Oboe
sys.path.append(Config.OBOE)
from auto_learner import AutoLearner as OboeLearner


def error(y_true, y_predicted, p_type):
    """Compute error metric for the model; varies based on classification/regression and algorithm type.
    BER (Balanced Error Rate): For classification.
                              1/n * sum (0.5*(true positives/predicted positives + true negatives/predicted negatives))
    MSE (Mean Squared Error): For regression. 1/n * sum(||y_pred - y_obs||^2).

    Args:
        y_true (np.ndarray):      Observed labels.
        y_predicted (np.ndarray): Predicted labels.
        p_type (str):             Type of problem. One of {'classification', 'regression'}
    Returns:
        float: Error metric.
    """

    assert p_type in {'classification', 'regression'}, "Please specify a valid type."
    y_true = np.squeeze(y_true)
    y_predicted = np.squeeze(y_predicted)

    if p_type == 'classification':
        errors = []
        epsilon = 1e-15
        for i in np.unique(y_true):
            tp = ((y_true == i) & (y_predicted == i)).sum()
            tn = ((y_true != i) & (y_predicted != i)).sum()
            fp = ((y_true != i) & (y_predicted == i)).sum()
            fn = ((y_true == i) & (y_predicted != i)).sum()
            errors.append(1 - 0.5*(tp / np.maximum(tp + fn, epsilon)) - 0.5*(tn / np.maximum(tn + fp, epsilon)))
        return np.mean(errors)

    elif p_type == 'regression':
        return mean_squared_error(y_true, y_predicted)


if __name__ == "__main__":  # note that the multiprocessing doesn't work properly without this guard
    # Get run parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--framework', required=True, type=str, choices=['auto-sklearn', 'Oboe'])
    parser.add_argument('-c', '--collection', required=True, type=str, help='Collection of datasets')
    parser.add_argument('-d', '--dataset_id', required=True, type=int, help='Dataset ID')
    parser.add_argument('-r', '--runtime_limit', required=True, type=int, help='Runtime limit (s)')
    parser.add_argument('--RANDOMSTATE', type=int, default=0, help='random state')
    parser.add_argument('-n', '--njobs', type=int, default=1, help='number of parallel jobs for this task')
    parser.add_argument('-k', '--kfolds', type=int, default=5, help='k for k-fold cross validation')
    parser.add_argument('-q', '--quiet', action='store_true', help='run extra quiet')
    args = parser.parse_args()

    # Get input handles
    dataset_dir = Path(Config.DATASETS).expanduser() / args.collection
    dataset_filename = dataset_dir / f'dataset_{args.dataset_id}_features_and_labels.csv'

    # Read the dataset
    dataset = pd.read_csv(dataset_filename, header=None)
    x = dataset.iloc[:, :-1].values
    y = LabelEncoder().fit_transform(dataset.iloc[:, -1])

    # Get output handles
    results_dir = Path(Config.RESULTS).expanduser() / args.collection / args.framework / f'njobs_{args.njobs}'
    results_dir.mkdir(parents=True, exist_ok=True)
    column_names = ['set_runtime_limit_per_fold', 'average_training_error', 'average_test_error',
                    'actual_runtime_per_fold'] + \
                   ['training_error_fold_{}'.format(i + 1) for i in range(args.kfolds)] + \
                   ['test_error_fold_{}'.format(i + 1) for i in range(args.kfolds)]
    output_filename = results_dir / f'dataset_{args.dataset_id}_{args.framework}_njobs_{args.njobs}.csv'
    log_file = results_dir / 'finished.txt'

    # Track things that go wrong
    if args.quiet:
        warnings.simplefilter("ignore")
    error_file = results_dir / 'errors.txt'
    dataset_error_count = 0
    dataset_errors = []

    # Run the task
    training_error = []
    test_error = []
    time_elapsed = []
    kf = StratifiedKFold(args.kfolds, shuffle=True, random_state=args.RANDOMSTATE)
    folds = 0
    try:
        for train_idx, test_idx in kf.split(x, y):
            folds += 1
            print(f'Fold {folds}/{args.kfolds}')
            x_train = x[train_idx, :]
            y_train = y[train_idx]
            x_test = x[test_idx, :]
            y_test = y[test_idx]

            # START TIMED SECTION
            if args.framework == 'auto-sklearn':
                start = time.time()
                clf = AutoSklearnClassifier(
                    time_left_for_this_task=args.runtime_limit,
                    include_preprocessors=["no_preprocessing"],
                    include_estimators=["adaboost", "gaussian_nb", "extra_trees", "gradient_boosting", "liblinear_svc",
                                        "libsvm_svc", "random_forest", "k_nearest_neighbors", "decision_tree"],
                    metric=balanced_accuracy,
                    n_jobs=args.njobs
                )
                clf.fit(x_train, y_train)
                clf.refit(x_train, y_train)
                time_elapsed.append(time.time() - start)  # Lau: TODO how to treat the time spent on refitting?
            elif args.framework == 'Oboe':
                start = time.time()
                clf = OboeLearner(
                    p_type='classification',
                    runtime_limit=args.runtime_limit,
                    verbose=False,
                    algorithms=['AB', 'GNB', 'ExtraTrees', 'GBT', 'lSVM', 'kSVM', 'RF', 'KNN', 'DT'],
                    selection_method='min_variance',
                    stacking_alg='greedy',
                    n_cores=args.njobs
                )
                clf.fit(x_train, y_train)
                time_elapsed.append(time.time() - start)  # Lau: TODO how to treat the time spent on refitting?
            else:
                raise NotImplementedError
            # END TIMED SECTION

            # Evaluate quality of learning
            y_train_pred = clf.predict(x_train)
            y_test_pred = clf.predict(x_test)
            training_error.append(error(y_train, y_train_pred, 'classification'))
            test_error.append(error(y_test, y_test_pred, 'classification'))

        # Reformat the output
        print('Folds finished, writing results.')
        training_error = np.array(training_error)
        test_error = np.array(test_error)

        # Aggregate the folds output
        time_elapsed = np.array(time_elapsed).mean()
        average_training_error = training_error.mean()
        average_test_error = test_error.mean()
    except Exception as e:
        print(e)
        dataset_errors.append(f'Error in dataset {args.dataset_id} with runtime limit {args.runtime_limit}.')
        dataset_errors.append(str(e))
        training_error = np.full(args.kfolds, np.nan)
        test_error = np.full(args.kfolds, np.nan)
        time_elapsed = np.nan
        average_training_error = np.nan
        average_test_error = np.nan
        dataset_error_count += 1
    results = np.concatenate((
        np.array([args.runtime_limit, average_training_error, average_test_error, time_elapsed]),
        training_error,
        test_error
    ))

    # Write out the results
    new_csv = not output_filename.is_file()
    with open(output_filename, 'a') as f:
        writer = csv.writer(f)
        if new_csv:  # add header to new file
            writer.writerow(column_names)
        writer.writerow(results)  # write the actual results

    with open(log_file, 'a') as log:
        log.write(f'dataset No.\t{args.dataset_id}\t finished with {dataset_error_count} errors.\n')
    with open(error_file, 'a') as failure_output:
        for failure in dataset_errors:
            failure_output.write(failure)
            failure_output.write('\n\n')
