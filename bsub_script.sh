#!/bin/sh
#BSUB -q gpua100
#BSUB -J "Adult Dataset Data"
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 23:59
#BSUB -R "rusage[mem=8GB]"
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo gpu-%J.out
#BSUB -eo gpu-%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/11.0
module load cudnn/v8.0.5.39-prod-cuda-11.0

source venv/bin/activate
cd notebooks
python HPC_Test.py
