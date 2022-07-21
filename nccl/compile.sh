#!/usr/bin/env bash

gcc nccl-hang.c -I/usr/local/cuda/include/ -L/usr/local/cuda/lib64/ -lcudart -L/usr/local/cuda/lib/ -lnccl -I/opt/amazon/openmpi/include -pthread -L/opt/amazon/openmpi/lib -Wl,-rpath -Wl,/opt/amazon/openmpi/lib -Wl,--enable-new-dtags -lmpi 
