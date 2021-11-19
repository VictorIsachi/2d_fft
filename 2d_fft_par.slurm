#!/bin/bash

#SBATCH --job-name 2d_fft_par_8
#SBATCH --error 2d_fft_par_8_err_%j.txt
#SBATCH --output 2d_fft_par_8_out_%j.txt
#SBATCH --partition allgroups
#SBATCH --ntasks 8
#SBATCH -c 1
#SBATCH --mem 1G
#SBATCH --time 01:00

module load intel-parallel-studio-professional.2019.4-gcc-8.2.1-fnvratt

mpirun -np 8 ./2d_fft_par "test_par_1.txt" 480 640
