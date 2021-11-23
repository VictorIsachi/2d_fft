# 2d_fft

In this repository you can find some of the code used to carry out the *parallel computing* project.

`2d_fft_seq_v2.0.cpp` and `2d_fft_par_v2.1.cpp` contain the source code of the sequential and parallel implementation respectively.

`compile_seq.txt` and `compile_par.txt` contain the commands used to compile the sequential and parallel implementation respectively. CAPRI did not allow me to compile with `spack load intel-parallel-studio@professional.2019.4` but, after consulting some online documentation, I was able to compile with `module load intel-parallel-studio-professional.2019.4-gcc-8.2.1-fnvratt` instead. CAPRI would also not compile with `mpicc` or `mpicxx` thus to compile I used the *Intel compiler* (`mpiicc`). To be more fair in the comparison I also compiled the sequential code with the *Intel compiler* (`icc`). 

The programs are executed with: `./2d_fft_{par|seq} "<filename>" <vertical_resolution> <horizontal_resolution>` \
e.g.: `./2d_fft_seq "input_file.txt" 1080 1920`

The files `.slurm` contain examples of tasks scheduled on CAPRI.

The files  `2d_fft_par_8_out_160865.txt` and `2d_fft_seq_out_160926.txt` contain example outputs of the program, for the parallel and sequential case respectively.

Details about the project can be found in the `.pdf` report (not in the repository).
