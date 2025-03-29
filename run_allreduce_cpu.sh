
SGLANG_CPU_OMP_THREADS_BIND="0-41|43-84|86-127|128-169|171-212|214-255" torchrun \
    --nproc_per_node=2 --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12355  \
    test_allreduce.py \
    --task allreduce \
    --elements 5242880 \
    --dtype bf16 \
    --nwarmup 100 \
    --niters 1000
