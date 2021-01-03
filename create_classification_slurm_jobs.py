from pathlib import Path
import argparse
import time
from experiment_config import Config

MAX_JOBS_PER_ENQUEUE = 50
FRAMEWORKS = ['auto-sklearn', 'Oboe']

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--collection', type=str, required=True, help='Collection of datasets')
parser.add_argument('-n', '--njobs', default=1, type=int, help='Number of cores to use')
parser.add_argument('-m', '--memory_per_core', default=10, type=int, help='Amount of memory to request per core in GB.')
parser.add_argument('-k', '--kfolds', type=int, default=5, help='Number of folds for k-fold cross validation')
args = parser.parse_args()

# Collect datasets
in_dir = Path(Config.DATASETS).expanduser() / args.collection
datasets = sorted([d for d in in_dir.iterdir() if d.is_file() and d.suffix == '.csv'])

# Define the target directory
slurm_jobs = Path(Config.SLURM_JOBS).expanduser()
slurm_output = Path(Config.SLURM_OUTPUT).expanduser()


batches = []
for d in datasets:
    i = d.stem.split('_')[1]
    dataset_id = int(i)
    if not i.isdigit():
        raise ValueError(f'Not a well-formed dataset name: {d}')
    for framework in FRAMEWORKS:
        runtimes = Config.RUNTIMES
        if framework == 'auto-sklearn':
            runtimes = [r for r in runtimes if r >= 30]  # This is a hardcoded lower limit in auto-sklearn
        total_script_time = sum(runtimes) * args.kfolds * 2
        formatted_time = time.strftime('%H:%M:%S', time.gmtime(total_script_time))
        batch = f'{args.collection}_{framework}_d{dataset_id}_n{args.njobs}.slurm'
        batches.append(batch)
        tasks_in_batch = []
        for runtime in runtimes:
            task = f'python run_one_test.py --collection {args.collection} --framework {framework} ' \
                   f'--dataset_id {dataset_id} --runtime_limit {runtime} --njobs {args.njobs} --quiet'
            tasks_in_batch.append(task)
        tasks_in_batch = '\n'.join(tasks_in_batch)
        with open(slurm_jobs / batch, 'w') as f:
            f.write(f'''#!/bin/bash

#SBATCH --job-name=bench_d{dataset_id}_{framework}_n{args.njobs}
#SBATCH --output={str(slurm_output)}/%x_%j.out
#SBATCH --error={str(slurm_output)}/%x_%j.err
#SBATCH --mail-user="{Config.SLURM_EMAIL}"
#SBATCH --mail-type="ALL"

#SBATCH --partition="cpu-short"
#SBATCH --time={formatted_time}
#SBATCH --ntasks={args.njobs}
#SBATCH --mem={args.memory_per_core}G


echo "### Starting Script"
cd {Config.PROJECT_ROOT}
module load Python/3.8.2-GCCcore-9.3.0
source venv/bin/activate
{tasks_in_batch}
echo "### Finished Script. Have a nice day."''')

for e, i in enumerate(range(0, len(batches), MAX_JOBS_PER_ENQUEUE)):
    with open(slurm_jobs / f'enqueue_{args.collection}_njobs_{args.njobs}_part_{e}.sh', 'w') as f:
        f.write('#!/bin/bash\n\n')
        for j in range(MAX_JOBS_PER_ENQUEUE):
            if i+j >= len(batches):
                break
            f.write(f'sbatch {str(slurm_jobs / batches[i+j])}\n')