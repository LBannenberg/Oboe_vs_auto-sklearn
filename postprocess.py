import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from experiment_config import Config

percentiles = [75, 50, 25]
runs = len(Config.RUNTIMES)
x = range(runs)

frameworks = [
    'auto-sklearn',
    'Oboe'
]
colors = {
    'auto-sklearn': 'blue',
    'Oboe': 'red'
}


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--collection', type=str, required=True, help='Collection name')
parser.add_argument('-n', '--njobs', type=int, required=True, help='njobs')
args = parser.parse_args()


def extract_or_impute_average_test_error(frame):
    results = np.empty(len(Config.RUNTIMES))
    for i, r in enumerate(Config.RUNTIMES):
        temp = frame[frame['set_runtime_limit_per_fold'] == r]
        if temp.shape[0] != 1:
            results[i] = 0.5
        elif np.isnan(float(temp['average_test_error'])):
            results[i] = 0.5
        else:
            results[i] = temp['average_test_error']
    return results


results_dir = Path(Config.RESULTS).expanduser() / args.collection
overall_results = {}
for framework in frameworks:
    print(framework.upper())
    fdf = pd.DataFrame()
    fdf['runtime'] = Config.RUNTIMES
    framework_results_dir = results_dir / framework / f'njobs_{args.njobs}'
    files = framework_results_dir.glob(f'dataset_*_{framework}_njobs_{args.njobs}.csv')
    datasets = []
    dataset_results = []
    for f in files:
        dataset_id = int(f.name.split('_')[1])
        datasets.append(dataset_id)
        df = pd.read_csv(f)
        values = extract_or_impute_average_test_error(df)
        dataset_results.append(values)
    dataset_results = np.vstack(dataset_results)

    fdf['success'] = [np.sum(dataset_results[:, i] < 0.5) for i in range(dataset_results.shape[1])]
    fdf['mean'] = np.nanmean(dataset_results, axis=0)
    fdf['median'] = np.nanmedian(dataset_results, axis=0)

    for perc in percentiles:
        runtime_percs = []
        for i in range(dataset_results.shape[1]):
            column = dataset_results[:, i]
            column = column[~np.isnan(column)]
            if column.shape[0]:
                runtime_percs.append(np.percentile(column, q=perc, interpolation='nearest'))
            else:
                runtime_percs.append(np.nan)
        fdf[perc] = runtime_percs
    overall_results[framework] = fdf


fig, ax = plt.subplots()
for framework in overall_results:
    df = overall_results[framework]
    color = colors[framework]
    plt.plot(x, df[25], color=color, linestyle=':', label=f'{framework} {25}%')
    plt.scatter(x, df[25], color=color)
    plt.plot(x, df[50], color=color, linestyle='-', label=f'{framework} {50}%')
    plt.scatter(x, df[50], color=color)
    plt.plot(x, df[75], color=color, linestyle='--', label=f'{framework} {75}%')
    plt.scatter(x, df[75], color=color)
plt.xticks(x, Config.RUNTIMES)
ax.set_xlabel('Runtime (s)')
ax.set_ylabel('Balanced Error Rate')
plt.legend()
plt.title('Runtime vs. Error')
plt.grid()
plt.tight_layout()
plt.savefig('Runtime-vs-Error.pdf')
plt.savefig('Runtime-vs-Error.png')
plt.show()
plt.close()


fig, ax = plt.subplots()
for framework in overall_results:
    df = overall_results[framework]
    color = colors[framework]
    ax.plot(x, df['success'], color=color, label=f'{framework} successful jobs')
    ax.scatter(x, df['success'], color=color)
plt.xticks(x, Config.RUNTIMES)
plt.yticks(range(0, 76, 5))
ax.set_xlabel('Runtime (s)')
ax.set_ylabel('Successes')
ax.set_title('Runtime vs. successful runs (75 datasets)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('Runtime-vs-Success.pdf')
plt.savefig('Runtime-vs-Success.png')
plt.show()


fig, ax = plt.subplots()
for framework in overall_results:
    df = overall_results[framework]
    color = colors[framework]
    ax.plot(x, df['mean'], color=color, linestyle='--', label=f'{framework} mean')
    ax.scatter(x, df['mean'], color=color)
    ax.plot(x, df['median'], color=color, label=f'{framework} median')
    ax.scatter(x, df['median'], color=color)
plt.xticks(x, Config.RUNTIMES)
ax.set_xlabel('Runtime (s)')
ax.set_ylabel('Balanced Error Rate')
ax.set_title('Mean and median performance')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('Runtime-vs-Mids.pdf')
plt.savefig('Runtime-vs-Mids.png')
plt.show()

combined = pd.DataFrame()
combined['runtime'] = Config.RUNTIMES
for framework in frameworks:
    df = overall_results[framework]
    combined[f'success_{framework}'] = df['success']
    combined[f'25th_{framework}'] = df[25]
    combined[f'50th_{framework}'] = df[50]
    combined[f'75th_{framework}'] = df[75]
    combined[f'mean_{framework}'] = df['mean']
    combined[f'median_{framework}'] = df['median']

success_rates = combined[['runtime', 'success_auto-sklearn', 'success_Oboe']]
success_rates.columns = ['runtime', 'auto-sklearn', 'Oboe']
success_rates.to_latex('success.tex', index=False)


percentile_results = combined[['runtime', '25th_auto-sklearn',
       '50th_auto-sklearn', '75th_auto-sklearn', '25th_Oboe', '50th_Oboe', '75th_Oboe']].round(3)
percentile_results.columns = ['runtime', '25th', '50th', '75th', '25th', '50th', '75th']
percentile_results.to_latex('percentiles.tex', index=False)

mids = combined[['runtime', 'mean_auto-sklearn', 'median_auto-sklearn', 'mean_Oboe',  'median_Oboe']].round(3)
mids.columns = ['runtime', 'mean', 'median', 'mean', 'median']
mids.to_latex('mids.tex', index=False)