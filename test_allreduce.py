import torch
import torch.distributed as dist
import os
from sglang.srt.distributed import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
import vllm
import time
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--elements", type=int, default=1024*5120)
parser.add_argument("--dtype", type=str, choices=["bf16", "fp32", "fp16"], default="bf16")
parser.add_argument("--task", type=str, choices=["allreduce", "allgather"], default="allreduce")
parser.add_argument("--nwarmup", type=int, default=20)
parser.add_argument("--niters", type=int, default=50)
parser.add_argument("--dim", type=int, default=0)

args = parser.parse_args()


if args.dtype=="bf16":
    dtype = torch.bfloat16
elif args.dtype=="fp32":
    dtype = torch.float32
elif args.dtype=="fp16":
    dtype = torch.float16


def setup(rank, world_size):
    """Initialize the distributed process group"""
    dist.init_process_group(
        backend="gloo",  # Use "nccl" for GPUs
        init_method="tcp://127.0.0.1:12355",
        rank=rank,
        world_size=world_size
    )

def cleanup():
    """Destroy the process group"""
    dist.destroy_process_group()


# Function to print timings
def print_head(world_size):
    print("#------------------------------------------------------------")
    print(f"# Benchmarking: {args.task}")
    print(f"# #processes: {world_size}")
    print("#------------------------------------------------------------")
    print()
    print("        #bytes  #repetitions   t_min[usec]   t_max[usec]   t_avg[usec]  stddev[%]")

def print_timings(local_timings, num_elements, dtype, num_iterations):
    # Compute min, max, average, and standard deviation
    # Calculate min, max, average, and standard deviation
    t_min = np.min(local_timings)
    t_max = np.max(local_timings)
    t_avg = np.mean(local_timings)
    stddev = np.std(local_timings) / t_avg * 100  # Percentage

    # Print results
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    total_bytes = num_elements * bytes_per_element
    print(f"{total_bytes:14d}{num_iterations:14d}{t_min:14.2f}{t_max:14.2f}{t_avg:14.2f}{stddev:12.2f}")

def init_all(rank, world_size):
    omp_cpuids = os.environ.get("SGLANG_CPU_OMP_THREADS_BIND", "all")
    if omp_cpuids == "all":
        local_omp_cpuid = "all"
    else:
        local_omp_cpuid = omp_cpuids.split("|")[rank]
    print("### local_omp_cpuid: ", local_omp_cpuid)
    torch.ops._C_utils.init_cpu_threads_env(local_omp_cpuid)
    init_distributed_environment(
        backend="gloo",
        world_size=world_size,
        rank=rank,
        distributed_init_method="tcp://127.0.0.1:12355",
        local_rank=rank,
    )
    # Initialization of shm all_reduce
    import sgl_kernel.common_ops

    shm_comm_op = sgl_kernel.common_ops
    os.environ["LOCAL_SIZE"] = str(world_size)
    shm_comm_op.initialize(world_size, rank)
    initialize_model_parallel(tensor_model_parallel_size=world_size, shm_comm_op=shm_comm_op)


def test_allreduce(rank, world_size):
    shm_comm_op = get_tp_group().shm_comm_op
    if rank == 0:
        print_head(world_size)
    num_elms = args.elements
    tensor = torch.rand(num_elms, dtype=dtype)
    warmup = args.nwarmup
    niters = args.niters
    time_list = []
    for i in range(niters + warmup):
        start = time.time()
        shm_comm_op.shm_allreduce(
            tensor, get_tp_group().device_group, torch.distributed.ReduceOp.SUM
        )
        end = time.time()
        if i >= warmup:
            time_list.append((end - start) * 1e6)  # Convert to microseconds

    if rank == 0:
        print_timings(time_list, num_elms, dtype, niters)



def test_allgather(rank, world_size):
    if rank == 0:
        print_head(world_size)
    num_elms = args.elements
    tensor = torch.rand(num_elms, dtype=dtype)
    warmup = args.nwarmup
    niters = args.niters
    dim = args.dim
    time_list = []
    for i in range(niters + warmup):
        start = time.time()
        get_tp_group().all_gather(tensor, dim=dim)
        end = time.time()
        if i >= warmup:
            time_list.append((end - start) * 1e6)  # Convert to microseconds

    if rank == 0:
        print_timings(time_list, num_elms, dtype, niters)

def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default

if __name__ == "__main__":
    rank = get_int_from_env(["LOCAL_RANK", "MPI_LOCALRANKID"], "0")
    world_size = get_int_from_env(["WORLD_SIZE", "PMI_SIZE"], "1")
    init_all(rank, world_size)
    if args.task == "allgather":
        test_allgather(rank, world_size)
    else:
        test_allreduce(rank, world_size)
