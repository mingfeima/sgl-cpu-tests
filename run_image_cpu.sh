
# jemalloc:
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1";
export LD_PRELOAD=/home/mingfeima/packages/jemalloc-5.3.0/lib/libjemalloc.so


# change this according to local machine setting
# using 3rd numa node with 40 cores
CORES=40
FIRST_CORE=120
LAST_CORE=`expr $FIRST_CORE + $CORES - 1`
PREFIX="numactl --physcpubind=$FIRST_CORE-$LAST_CORE --membind=3"
echo -e "### using $PREFIX\n"


# exaples to run:

# enable profile and validate
#OMP_NUM_THREADS=$CORES $PREFIX python3 ./bench_image_processing.py --profile --validate

# enable validate
OMP_NUM_THREADS=$CORES $PREFIX python3 ./bench_image_processing.py --validate

# enable validate and fuse bfloat16 conversion
#OMP_NUM_THREADS=$CORES $PREFIX python3 ./bench_image_processing.py --validate --bfloat16
