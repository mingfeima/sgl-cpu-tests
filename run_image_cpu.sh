
# jemalloc:
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1";
export LD_PRELOAD=/home/mingfeim/jemalloc-5.3.0/lib/libjemalloc.so

CORES=32
LAST_CORE=`expr 60 + $CORES - 1`
PREFIX="numactl --physcpubind=60-$LAST_CORE --membind=1"
echo -e "### using $PREFIX\n"

#OMP_NUM_THREADS=$CORES $PREFIX python3 ./bench_image_processing.py

export ATEN_CPU_CAPABILITY=avx2

# enable profile and validate
#OMP_NUM_THREADS=$CORES $PREFIX python3 ./bench_image_processing.py --profile --validate
OMP_NUM_THREADS=$CORES $PREFIX python3 ./bench_image_processing.py --validate
