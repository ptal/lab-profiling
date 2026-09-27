# Profiling a CUDA Kernel

```
nvcc -O3 matrix.cu -Xcompiler -fopenmp -o matrix
./matrix 1000 gpu-opt1
ncu --set full -o gpu_opt1_1000_report -f ./matrix 1000 gpu-opt1
ncu-ui gpu_opt1_1000_report.ncu-repz
```

More info:
* Usage: ./matrix [n] [mode] [-test], where mode is "gpu" (default) or "cpu".
* Add the suffix "-opt1" or "-opt2" to "gpu" to use matrix_mul_opt1 or matrix_mul_opt2 instead of matrix_mul (e.g. "gpu-opt1").
* The option -test (disabled by default because it is slow) runs both the GPU and CPU versions and compares their results.
* The number of CPU threads is controlled by OMP_NUM_THREADS.
