# PS-CUDA

*A prefix sum implementation leveraging CUDA*

*Coursework in Multi-Core Many-Core Programming 2016/2017*


## Build
To build the application, run:

  make clean all

It is possible to override the following parameters to customize the compilation:

* **BLOCK_DIM** set the dimension of blocks in the grid.
The value can be set to BLOCK_DIM_4, BLOCK_DIM_8, BLOCK_DIM_16, BLOCK_DIM_32, BLOCK_DIM_64, BLOCK_DIM_128, BLOCK_DIM_256, BLOCK_DIM_512, BLOCK_DIM_1024.
Default value is BLOCK_DIM_32.

* **GPU_ARCHITECTURE** set the real GPU architecture.
The value can be set to any Nvidia code for real GPU architectures.
Default value is sm_35.

* **OPT_LEVEL** set the optimization level for the C compiler.
The value can be set to any optimization level supported by the GNU compiler.
Default value is 0.

* **VERBOSITY** set the verbosity level.
The value can be set to VERBOSITY_OFF or VERBOSE.
Default value is VERBOSITY_OFF.

For example, to compile the program with block dimension 128, run:

  make clean all BLOCK_DIM=BLOCK_DIM_128


## Test of Correctness
To test the correctness of the application, run:

  make test

To test the correctness of the application, taking into account all the combinations of settings, run:

  make test_all

and see the output in `out/correctness/correctness-[PRECISION]-[BLOCK_DIMENSION].out`.


## Profiling
To evaluate the performance of the compiled application, run:

  make profile

and see the output in console`out/profile/profile-[DATE].csv`.

To evaluate the performance of the application, taking into account all the combinations of settings, run:

  make profile_all

and see the output in `out/profile/timing.csv`, which contains 5 replications of timing performances for each configuration.

Notice that the evaluation process profiles the application varying the following parameters:

* PRECISION: single and double.
* BLOCK_DIMENSION: 16, 32, 64, 128, 256, 512 and 1024.
* OPT_LEVEL: 0, 1, 2 and 3.

You can plot this last output by running the Matlab script `profile/profile_plot.m`.


## Authors
Giacomo Marciani, [gmarciani@acm.org](mailto:gmarciani@acm.org)

Gabriele Santi, [gabriele.santi@acm.org](mailto:gabriele.santi@acm.org)


## References
Giacomo Marciani and Gabriele Santi. 2017. *A HPC showcase*. Courseworks in Multi-Core Many-Core Programming. University of Rome Tor Vergata, Italy


## License
The project is released under the [MIT License](https://opensource.org/licenses/MIT).
