#!/bin/bash

# Configuration
NVCC=nvcc
CFLAGS="-O2"
SRC_FILE="gpu_dataset.cu"
EXEC_FILE="gpu_dataset.exe"
LOG_FILE="benchmark_results.log"

# Dataset sizes matching your experiment
DATASET_SIZES=(1000000 5000000 10000000 20000000 50000000)
ITERATIONS=5

echo "=========================================="
echo " Starting CUDA Dataset Processing Benchmark"
echo "=========================================="

# Clear previous log file
> "$LOG_FILE"

for size in "${DATASET_SIZES[@]}"; do
    echo "Processing Dataset Size: $size elements..." | tee -a "$LOG_FILE"
    
    # Compile with updated DATA_SIZE macro
    $NVCC $CFLAGS -DDATA_SIZE=$size "$SRC_FILE" -o "$EXEC_FILE"
    if [ $? -ne 0 ]; then
        echo "Compilation failed for DATA_SIZE=$size"
        exit 1
    fi

    for iter in $(seq 1 $ITERATIONS); do
        echo "--- Iteration $iter (N=$size) ---" >> "$LOG_FILE"
        ./"$EXEC_FILE" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"
    done
done

echo "=========================================="
echo " Benchmark completed. Output saved to $LOG_FILE"
echo "=========================================="
