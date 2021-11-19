#!/bin/bash

#SBATCH --job-name 2d_fft_seq
#SBATCH --error 2d_fft_seq_err_%j.txt
#SBATCH --output 2d_fft_seq_out_%j.txt
#SBATCH --partition allgroups
#SBATCH --ntasks 1
#SBATCH -c 1
#SBATCH --mem 1G
#SBATCH --time 01:00

module load intel-parallel-studio-professional.2019.4-gcc-8.2.1-fnvratt

./2d_fft_seq "test_seq.txt" 480 640
