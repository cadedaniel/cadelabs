nccl-hang.c is a simple program that verifies that NCCL hangs when communications are enqueued in the incorrect order.

#### Overview
There are two ranks which each send and receive one tensor, such that two tensors are communicated in total. Rank 0 sends tensor A and receives tensor B. Rank 1 sends tensor B and receives tensor A.

The order in which each rank sends/receives the tensor is configurable. Rank 0 can send A before receiving B, or vice versa. Same for rank 1.

This can be used to show that NCCL will hang when the ordering is incorrect; when rank 0 sends tensor A and rank 1 sends tensor B, there will be a hang since both NCCL kernels are awaiting a cooperative reception kernel on their peer device.


#### Correct ordering
```bash
$ ./run.sh --rank0-order=b,a --rank1-order=b,a
[MPI Rank 0] ncclUniqueId 2812674050
[MPI Rank 1] ncclUniqueId 2812674050
bidirectional NCCL test
A has size 2097152 bytes
B has size 2097152 bytes
  Rank 0: Recv B
  Rank 0: Send A
  Rank 1: Send B
  Rank 1: Recv A
[MPI Rank 0] Success
[MPI Rank 1] Success
```

#### Incorrect ordering
```bash
$ ./run.sh --rank0-order=b,a --rank1-order=a,b
[MPI Rank 0] ncclUniqueId 1773338626
[MPI Rank 1] ncclUniqueId 1773338626
bidirectional NCCL test
A has size 2097152 bytes
B has size 2097152 bytes
  Rank 0: Recv B
  Rank 0: Send A
  Rank 1: Recv A
  Rank 1: Send B
# Hang
```
