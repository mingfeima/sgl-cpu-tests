
export KMP_BLOCKTIME=1
export KMP_TPAUSE=0
#export KMP_SETTINGS=1
export KMP_FORKJOIN_BARRIER_PATTERN="dist,dist"
export KMP_PLAIN_BARRIER_PATTERN="dist,dist"
export KMP_REDUCTION_BARRIER_PATTERN="dist,dist"
export KMP_AFFINITY="granularity=fine,compact,1,0"

# jemalloc:
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1";
export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}/home/mingfeima/.venv/lib/libiomp5.so:/home/mingfeima/packages/jemalloc-5.3.0/lib/libjemalloc.so"

CORES=40
FIRST_CORE=80
LAST_CORE=`expr $FIRST_CORE + $CORES - 1`
PREFIX="numactl --physcpubind=$FIRST_CORE-$LAST_CORE --membind=2"
echo -e "### using $PREFIX\n"

OMP_NUM_THREADS=$CORES $PREFIX python3 $1
