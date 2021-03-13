#!/bin/bash
 
#SBATCH --nodes=1 			#number of compute nodes
#SBATCH -n 12 			#number of CPU cores to reserve on this compute node

#SBATCH -p rcgpu6		#Use cidsegpu1 partition
#SBATCH -q wildfire		#Run job under wildfire QOS queue

#SBATCH --gres=gpu:1		#limit is 4, physicsgpu1:4, sulcgpu1:8, wzhengpu1:4

#SBATCH --time=55:00:00
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=akobtan@asu.edu # send-to address

##module purge    # Always purge modules to ensure a consistent environment

module load anaconda/py3
##pip3 install --user pytorch_transformers
##pip3 install --user transformers
##pip3.7 install --user tqdm


export OMP_NUM_THREADS=24
export OMP_DYNAMIC=TRUE

python3.7 BERT_NER_agave.py
