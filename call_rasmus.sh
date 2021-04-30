#!/bin/sh
#BSUB -q gpuv100
#BSUB -J "gas_CP2"
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 23:59
#BSUB -R "rusage[mem=8GB]"
##BSUB -R "select[gpu32gb]"
#BSUB -u s173910@student.dtu.dk
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo gas_CP2-%J.out
# -- end of LSF options --

#nvidia-smi
# Load the cuda module
module load cuda/11.0
module load cudnn/v8.0.5.39-prod-cuda-11.0

source venv/bin/activate
cd notebooks
cd ffjord_data_runs
python gas.py