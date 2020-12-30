#!/bin/bash

#SBATCH --job-name=fetch_datasets
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="your.email@example.com"
#SBATCH --mail-type="ALL"

#SBATCH --partition="cpu-short"
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=10G

echo "### Starting Script"
cd /home/username/PythonProjects/Oboe_vs_auto-sklearn
module load Python/3.8.2-GCCcore-9.3.0
source venv/bin/activate
python fetch_preprocess_datasets.py --first75
echo "### Finished Script. Have a nice day."
