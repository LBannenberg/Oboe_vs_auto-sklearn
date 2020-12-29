from pathlib import Path
import argparse
import time
from experiment_config import Config

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--njobs', default=1, type=int, help='Number of cores to use')
parser.add_argument('-m', '--memory_per_core', default=10, type=int, help='Amount of memory to request per core in GB.')
parser.add_argument('-k', '--kfolds', type=int, default=5, help='Number of folds for k-fold cross validation')
parser.add_argument('-c', '--collection', type=str, default='OpenML_classification', help='Collection of datasets')
args = parser.parse_args()

# Collect datasets
in_dir = Path(Config.DATASETS).expanduser() / args.collection
datasets = [d for d in in_dir.iterdir() if d.is_file() and d.suffix == '.csv']

# Define the target directory
out_dir = Path(Config.RESULTS).expanduser() / 'OpenML_classification' / 'autosklearn' / f'njobs_{args.njobs}'
slurm_jobs = Path(Config.SLURM_JOBS).expanduser()
slurm_output = Path(Config.SLURM_OUTPUT).expanduser()

batches = []
for d in datasets:
    for framework in ['auto-sklearn', 'Oboe']:
        for runtime in Config.RUNTIMES:
            total_script_time = runtime * args.kfolds * 2
            formatted_time = time.strftime('%H:%M:%S', time.gmtime(total_script_time))
            i = d.stem.split('_')[1]
            if not i.isdigit():
                raise ValueError(f'Not a well-formed dataset name: {d}')
            dataset_id = int(i)
            batch = f'OpenML_classification_{framework}_d{dataset_id}_n{args.njobs}.slurm'
            batches.append(batch)
            with open(slurm_jobs / batch, 'w') as f:
                f.write(f'''#!/bin/bash

#SBATCH --job-name=bench_{framework}_d{dataset_id}_n{args.njobs}
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
python run_one_test.py --collection {args.collection} --framework {framework} --dataset_id {dataset_id} --runtime {runtime} --njobs {args.njobs} --quiet
echo "### Finished Script. Have a nice day."''')


with open(slurm_jobs / f'enqueue_OpenML_classification_njobs_{args.njobs}.sh', 'w') as f:
    f.write('#!/bin/bash\n\n')
    for batch in batches:
        f.write(f'sbatch {str(slurm_jobs / batch)}\n')