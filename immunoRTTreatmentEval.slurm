#!/bin/bash -l
#SBATCH --job-name=ImmunoRTTreatmentEval2Parallel
#SBATCH --partition=long
#SBATCH --mail-user=23303755@student.uwa.edu.au
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --time=80:00:00
#SBATCH --exclusive
# load modules
module load python/3.10.10
module load py-numpy/1.23.4
module load py-matplotlib/3.6.2
module load py-pandas/1.5.1
# launch serial python script
srun python3 ImmunoRTTreatmentEval.py
