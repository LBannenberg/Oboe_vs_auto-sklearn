class Config:
    # Oboe dependency
    OBOE = '/home/username/PythonProjects/oboe/automl/Oboe'

    # Experiments
    PROJECT_ROOT = '/home/username/PythonProjects/Oboe_vs_auto-sklearn'
    DATASETS = '/home/username/PythonProjects/Oboe_vs_auto-sklearn/datasets'
    RESULTS = '/home/username/PythonProjects/Oboe_vs_auto-sklearn/results'

    # Slurm
    SLURM_EMAIL = 'my.email@example.com'
    SLURM_JOBS = '/home/username/PythonProjects/Oboe_vs_auto-sklearn/jobs_slurm'
    SLURM_OUTPUT = '/home/username/PythonProjects/Oboe_vs_auto-sklearn/output_slurm'

    RUNTIMES = [1, 2, 4, 8, 16, 32, 64, 128]
