#!/bin/bash

# CUDA compiler
NVCC=nvcc

# Compiler flags
CFLAGS="-O2"

# CUDA dataset source files
DATASET_FILES=(
    "dataset_1M.cu"
    "dataset_5M.cu"
    "dataset_10M.cu"
    "dataset_20M.cu"
    "dataset_50M.cu"
)

# Dataset names
DATASET_NAMES=(
    "1,000,000"
    "5,000,000"
    "10,000,000"
    "20,000,000"
    "50,000,000"
)

# Number of recordings for each dataset
ITERATIONS=5

LOG_FILE="benchmark_results.log"

echo "=========================================="
echo " Starting CUDA Dataset Processing Benchmark"
echo "=========================================="

# Clear previous log
> "$LOG_FILE"

for i in "${!DATASET_FILES[@]}"; do

    SRC_FILE="${DATASET_FILES[$i]}"
    DATASET_SIZE="${DATASET_NAMES[$i]}"
    EXEC_FILE="gpu_dataset.exe"

    echo ""
    echo "Processing Dataset Size: $DATASET_SIZE elements..."
    echo "Source File: $SRC_FILE"

    echo "==========================================" >> "$LOG_FILE"
    echo "Dataset Size: $DATASET_SIZE elements" >> "$LOG_FILE"
    echo "Source File: $SRC_FILE" >> "$LOG_FILE"
    echo "==========================================" >> "$LOG_FILE"

    # Compile the CUDA source file
    $NVCC $CFLAGS "$SRC_FILE" -o "$EXEC_FILE"

    if [ $? -ne 0 ]; then
        echo "Compilation failed for $SRC_FILE"
        exit 1
    fi

    # Run 5 recordings
    for iter in $(seq 1 $ITERATIONS); do

        echo "Run $iter/5 for $DATASET_SIZE..."

        echo "--- Run $iter ---" >> "$LOG_FILE"

        ./"$EXEC_FILE" >> "$LOG_FILE"

        if [ $? -ne 0 ]; then
            echo "Execution failed for $SRC_FILE, Run $iter"
            exit 1
        fi

        echo "" >> "$LOG_FILE"
    done

done

echo "=========================================="
echo " Benchmark completed."
echo " Results saved to: $LOG_FILE"
echo "=========================================="
