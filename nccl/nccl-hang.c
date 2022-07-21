//
// Example 2: One Device Per Process Or Thread
//

#include "cuda_runtime.h"
#include "mpi.h"
#include "nccl.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MPICHECK(cmd)                                                          \
  do {                                                                         \
    int e = cmd;                                                               \
    if (e != MPI_SUCCESS) {                                                    \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,            \
             cudaGetErrorString(e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t r = cmd;                                                      \
    if (r != ncclSuccess) {                                                    \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,            \
             ncclGetErrorString(r));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

typedef struct {
  uint16_t *device_buffer;
  size_t buffer_size;
  uint16_t repeating_value;
  char *label;
} comm_buffer_t;

typedef struct {
  int rank;
  int peer_rank;
  ncclComm_t nccl_comm;
  cudaStream_t cuda_stream;
} test_context_t;

typedef struct {
  int rank0_a_before_b;
  int rank1_a_before_b;
  int different_size_messages;
} comm_order_t;

void initialize(int *, int *, ncclUniqueId *);
void fill_gpu_buffer_with_value(void *device_buffer, size_t buffer_size,
                                uint16_t value);
uint16_t *copy_gpu_buffer_to_cpu(void *device_buffer, size_t buffer_size);
void load_and_check_device_buffer(comm_buffer_t message, int rank);

void bidirectional(test_context_t ctx, comm_buffer_t message_a,
                   comm_buffer_t message_b, comm_order_t ordering) {

  if (ctx.rank == 0) {
    printf("bidirectional NCCL test\n");
    printf("A has size %zu bytes\n", message_a.buffer_size * sizeof(uint16_t));
    printf("B has size %zu bytes\n", message_b.buffer_size * sizeof(uint16_t));
    if (ordering.rank0_a_before_b) {
      printf("  Rank 0: Send A\n");
      printf("  Rank 0: Recv B\n");
    } else {
      printf("  Rank 0: Recv B\n");
      printf("  Rank 0: Send A\n");
    }

    if (ordering.rank1_a_before_b) {
      printf("  Rank 1: Recv A\n");
      printf("  Rank 1: Send B\n");
    } else {
      printf("  Rank 1: Send B\n");
      printf("  Rank 1: Recv A\n");
    }
  }

  // 0 sends A to 1
  // 1 sends B to 0
  if (ctx.rank == 0) {
    fill_gpu_buffer_with_value(message_a.device_buffer, message_a.buffer_size,
                               message_a.repeating_value);
  }
  if (ctx.rank == 1) {
    fill_gpu_buffer_with_value(message_b.device_buffer, message_b.buffer_size,
                               message_b.repeating_value);
  }

  // 0 sends A to 1
  {
    NCCLCHECK(ncclGroupStart());
    if (ctx.rank == 0) {
      if (ordering.rank0_a_before_b) {
        NCCLCHECK(ncclSend(message_a.device_buffer,
                           message_a.buffer_size * sizeof(uint16_t), ncclUint8,
                           ctx.peer_rank, ctx.nccl_comm, ctx.cuda_stream));
      } else {
        NCCLCHECK(ncclRecv(message_b.device_buffer,
                           message_b.buffer_size * sizeof(uint16_t), ncclUint8,
                           ctx.peer_rank, ctx.nccl_comm, ctx.cuda_stream));
      }
    }

    if (ctx.rank == 1) {
      if (ordering.rank1_a_before_b) {
        NCCLCHECK(ncclRecv(message_a.device_buffer,
                           message_a.buffer_size * sizeof(uint16_t), ncclUint8,
                           ctx.peer_rank, ctx.nccl_comm, ctx.cuda_stream));
      } else {
        NCCLCHECK(ncclSend(message_b.device_buffer,
                           message_b.buffer_size * sizeof(uint16_t), ncclUint8,
                           ctx.peer_rank, ctx.nccl_comm, ctx.cuda_stream));
      }
    }
    NCCLCHECK(ncclGroupEnd());
  }

  // 1 sends B to 0
  {
    NCCLCHECK(ncclGroupStart());
    if (ctx.rank == 0) {
      if (ordering.rank0_a_before_b) {
        NCCLCHECK(ncclRecv(message_b.device_buffer,
                           message_b.buffer_size * sizeof(uint16_t), ncclUint8,
                           ctx.peer_rank, ctx.nccl_comm, ctx.cuda_stream));
      } else {
        NCCLCHECK(ncclSend(message_a.device_buffer,
                           message_a.buffer_size * sizeof(uint16_t), ncclUint8,
                           ctx.peer_rank, ctx.nccl_comm, ctx.cuda_stream));
      }
    }

    if (ctx.rank == 1) {
      if (ordering.rank1_a_before_b) {
        NCCLCHECK(ncclSend(message_b.device_buffer,
                           message_b.buffer_size * sizeof(uint16_t), ncclUint8,
                           ctx.peer_rank, ctx.nccl_comm, ctx.cuda_stream));
      } else {
        NCCLCHECK(ncclRecv(message_a.device_buffer,
                           message_a.buffer_size * sizeof(uint16_t), ncclUint8,
                           ctx.peer_rank, ctx.nccl_comm, ctx.cuda_stream));
      }
    }
    NCCLCHECK(ncclGroupEnd());
  }

  CUDACHECK(cudaDeviceSynchronize());

  if (ctx.rank == 1) {
    load_and_check_device_buffer(message_a, ctx.rank);
  }

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  if (ctx.rank == 0) {
    load_and_check_device_buffer(message_b, ctx.rank);
  }
}

int parse_args(int argc, char **argv, comm_order_t *ordering) {

  char *rank0_a_before_b = "--rank0-order=a,b";
  char *rank0_b_before_a = "--rank0-order=b,a";
  char *rank1_a_before_b = "--rank1-order=a,b";
  char *rank1_b_before_a = "--rank1-order=b,a";
  char *different_size_messages = "--unique-message-size";

  for (size_t i = 1; i < argc; i++) {
    if (!strncmp(argv[i], rank0_a_before_b, strlen(rank0_a_before_b))) {
      ordering->rank0_a_before_b = 1;
    }
    if (!strncmp(argv[i], rank0_b_before_a, strlen(rank0_b_before_a))) {
      ordering->rank0_a_before_b = 0;
    }
    if (!strncmp(argv[i], rank1_a_before_b, strlen(rank1_a_before_b))) {
      ordering->rank1_a_before_b = 1;
    }
    if (!strncmp(argv[i], rank1_b_before_a, strlen(rank1_b_before_a))) {
      ordering->rank1_a_before_b = 0;
    }
    if (!strncmp(argv[i], different_size_messages,
                 strlen(different_size_messages))) {
      ordering->different_size_messages = 1;
    }
  }
  return 0;
}

int main(int argc, char *argv[]) {

  comm_order_t ordering = {.rank0_a_before_b = 1, .rank1_a_before_b = 1, .different_size_messages=0};
  if (parse_args(argc, argv, &ordering)) {
    return 1;
  }

  int rank;
  int comm_size;
  ncclUniqueId nccl_id;

  MPICHECK(MPI_Init(&argc, &argv));

  initialize(&rank, &comm_size, &nccl_id);

  ncclComm_t comm;
  cudaStream_t cuda_stream = 0;

  comm_buffer_t message_a = {
      .buffer_size = 1 << 20, .repeating_value = 55, .label = "A"};
  comm_buffer_t message_b = {
      .buffer_size = 1 << 20, .repeating_value = 56, .label = "B"};

  if (ordering.different_size_messages) {
    message_b.buffer_size *= 2;
  }

  CUDACHECK(cudaMalloc((void **)&message_a.device_buffer,
                       message_a.buffer_size * sizeof(uint16_t)));
  CUDACHECK(cudaMalloc((void **)&message_b.device_buffer,
                       message_b.buffer_size * sizeof(uint16_t)));

  NCCLCHECK(ncclCommInitRank(&comm, comm_size, nccl_id, rank));

  test_context_t ctx = {.rank = rank,
                        .peer_rank = rank == 0 ? 1 : 0,
                        .nccl_comm = comm,
                        .cuda_stream = cuda_stream};

  bidirectional(ctx, message_a, message_b, ordering);

  CUDACHECK(cudaFree(message_a.device_buffer));
  CUDACHECK(cudaFree(message_b.device_buffer));
  NCCLCHECK(ncclCommDestroy(comm));
  MPICHECK(MPI_Finalize());

  printf("[MPI Rank %d] Success \n", rank);
  return 0;
}

void initialize(int *rank, int *comm_size, ncclUniqueId *id) {
  // initializing MPI
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, comm_size));

  // get NCCL unique ID at rank 0 and broadcast it to all others
  if (*rank == 0) {
    NCCLCHECK(ncclGetUniqueId(id));
  }
  MPICHECK(
      MPI_Bcast((void *)id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));
  CUDACHECK(cudaSetDevice(*rank));

  printf("[MPI Rank %d] ncclUniqueId %u\n", *rank, *(uint32_t *)id);
}

void fill_gpu_buffer_with_value(void *device_buffer, size_t buffer_size,
                                uint16_t value) {
  uint16_t *cpu_buffer = malloc(buffer_size * sizeof(uint16_t));
  for (size_t i = 0; i < buffer_size; i++) {
    cpu_buffer[i] = value;
  }

  CUDACHECK(cudaMemcpy(device_buffer, cpu_buffer,
                       buffer_size * sizeof(uint16_t), cudaMemcpyDefault));
  CUDACHECK(cudaStreamSynchronize(0));
  free(cpu_buffer);
}

uint16_t *copy_gpu_buffer_to_cpu(void *device_buffer, size_t buffer_size) {
  uint16_t *cpu_buffer = malloc(buffer_size * sizeof(uint16_t));
  CUDACHECK(cudaMemcpy(cpu_buffer, device_buffer,
                       buffer_size * sizeof(uint16_t), cudaMemcpyDefault));
  CUDACHECK(cudaStreamSynchronize(0));
  return cpu_buffer;
}

void load_and_check_device_buffer(comm_buffer_t message, int rank) {
  uint16_t *cpu_buffer =
      copy_gpu_buffer_to_cpu(message.device_buffer, message.buffer_size);

  // for (size_t i = 0; i < 5; i++) {
  //  printf("[MPI Rank %d] buffer %s value %zu: %hu\n", rank, message.label, i,
  //         cpu_buffer[i]);
  //}

  for (size_t i = 0; i < message.buffer_size; ++i) {
    if (cpu_buffer[i] != message.repeating_value) {
      printf("[MPI Rank %d] different buffer %s value %zu: %hu\n", rank,
             message.label, i, cpu_buffer[i]);
      break;
    }
  }
  free(cpu_buffer);
}
