#!/usr/bin/env bash

mpirun -x LD_PRELOAD=/usr/local/cuda/lib/libnccl.so -n 2 ./a.out "$@"
