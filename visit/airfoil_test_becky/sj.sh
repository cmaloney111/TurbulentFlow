#!/bin/bash
#SBATCH --partition=shared
#SBATCH --account=slo102
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --output=job_output.txt

# Run the command
nekbmpi airfoil_run2 1
