#!/bin/bash

#$ -cwd                      # Run the code from the current directory
#$ -j y                      # combine screen output and errors 
#$ -l h_rt=70:0:0            # Limit each task to 70 hr
# -l h_vmem= 1G   
#$ -t 1-1001                  # jobs from 1 to 100
#$ -tc 100                   # number of core requested
#$ -pe smp 1                 # one core per job
#$ -o /data/home/apx050/env/PyTransportEnv/PyTransport-master/Examples/MINE/StepModel/cluster/ # screeen output goes here
module load gcc
module load python   
source /data/home/apx050/env/PyTransportEnv/bin/activate
CMD0=${SGE_TASK_ID}

python test.py ${CMD0}