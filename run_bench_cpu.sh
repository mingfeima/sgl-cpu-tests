
# jemalloc:
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1";
export LD_PRELOAD=/home/mingfeim/jemalloc-5.3.0/lib/libjemalloc.so

CORES=42
LAST_CORE=`expr 0 + $CORES - 1`
PREFIX="numactl --physcpubind=0-$LAST_CORE --membind=0"
echo -e "### using $PREFIX\n"

OMP_NUM_THREADS=$CORES $PREFIX python3 $1
