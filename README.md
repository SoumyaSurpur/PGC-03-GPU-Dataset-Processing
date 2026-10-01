
# GPU Dataset Processing using CUDA

A comprehensive experiment to analyze execution times, speedup, and memory overhead when processing large numeric datasets sequentially on the CPU versus in parallel using CUDA on an NVIDIA GPU.

---

## Table of Contents

- [Overview](#overview)
- [Dataset Plan](#dataset-plan)
- [System Requirements](#system-requirements) 
- [Quick Start Guide](#quick-start-guide)
- [Benchmark Execution & Results Table](#benchmark-execution--results-table)
  - [Iteration Runs (1 to 5)](#iteration-runs-1-to-5)
  - [Final Average Summary Table](#final-average-summary-table)
- [Calculating Speedup](#calculating-speedup)
- [Performance Graphs & Visualizations](#performance-graphs--visualizations)
  - [1. Compute-Only Kernel Speedup Scaling](#1-compute-only-kernel-speedup-scaling)
  - [2. Execution Time Breakdown](#2-execution-time-breakdown)
- [Deep-Dive Technical Analysis](#deep-dive-technical-analysis)
  - [1. Why Total CUDA Time Exceeds CPU Time (Small Datasets)](#1-why-total-cuda-time-exceeds-cpu-time-small-datasets)
  - [2. Why Compare CUDA Kernel Time vs. CPU Time?](#2-why-compare-cuda-kernel-time-vs-cpu-time)
- [Key Takeaways](#key-takeaways)
- [How to Run](#how-to-run)

---

## Overview

- **Theme:** GPU Dataset Processing
- **Parallel Mode:** CUDA
- **Main Task:** Perform an element-wise numeric operation ($\text{Output}[i] = \text{Input}[i] \times 2$) across 5 different dataset sizes, measure processing times, verify data correctness, and calculate speedup.

---

## Dataset Plan

The experiment tests 5 separate dataset sizes using single-precision floating-point numbers:

| Dataset Run | Number of Elements | Approx. Float Data Size |
| :--- | :--- | :--- |
| **Dataset 1** | 1,000,000 (1M) | ~4 MB |
| **Dataset 2** | 5,000,000 (5M) | ~20 MB |
| **Dataset 3** | 10,000,000 (10M) | ~40 MB |
| **Dataset 4** | 20,000,000 (20M) | ~80 MB |
| **Dataset 5** | 50,000,000 (50M) | ~200 MB |

> **Note:** Data is programmatically generated in host memory using `h_input[i] = (float)(i % 1000)`—no external datasets are required.

---

## System Requirements

### Hardware & Driver
- NVIDIA CUDA-capable GPU
- Installed NVIDIA GPU Driver
- Sufficient System RAM and GPU Memory

### Software & Environment
- **OS:** Windows 10/11 or Linux/WSL
- **Compiler:** NVIDIA CUDA Toolkit (`nvcc`) with host C++ compiler (`cl.exe` or `g++`)
- **Python Environment:** Python 3.8+ with `matplotlib`, `numpy`, and `pandas`
- **IDE/Terminal:** VS Code, PowerShell, Bash, or Command Prompt

---

## Quick Start Guide

### 1. Environment Verification
Verify that your GPU and CUDA compiler are properly configured in your path:

```bash
nvidia-smi
```
<img width="933" height="798" alt="nvidia-smi" src="https://github.com/user-attachments/assets/e32b91c7-18bb-43e2-869e-5585adf5517d" />

``` bash
nvcc --version
```
<img width="790" height="135" alt="nvcc--version" src="https://github.com/user-attachments/assets/49ba9ba9-5ddf-4b1b-beec-aea5b5c868f2" />


### 2. Build and Run Single Benchmark
Compile the CUDA source file using the `-O2` optimization flag and execute the binary:

```bash
nvcc -O2 dataset_1M.cu -o dataset_1M.exe
.\dataset_1M.exe
```

---

## Benchmark Execution & Results Table

### Iteration Runs (1 to 5)

#### Iteration 1
| Dataset Size | CPU Time (ms) | CUDA Kernel (ms) | Total CUDA (ms) | Speedup (Total) | Speedup (Kernel) | Verification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1,000,000** | 0.808400 | 0.426722 | 2.037024 | 0.40x | **1.89x** | PASSED |
| **5,000,000** | 5.190000 | 0.567008 | 8.397856 | 0.62x | **9.15x** | PASSED |
| **10,000,000** | 8.697700 | 0.657568 | 16.986401 | 0.51x | **13.23x** | PASSED |
| **20,000,000** | 17.936500 | 0.959392 | 32.492737 | 0.55x | **18.70x** | PASSED |
| **50,000,000** | 44.009000 | 1.587072 | 77.753311 | 0.57x | **27.73x** | PASSED |

#### Iteration 2
| Dataset Size | CPU Time (ms) | CUDA Kernel (ms) | Total CUDA (ms) | Speedup (Total) | Speedup (Kernel) | Verification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1,000,000** | 0.856800 | 0.405312 | 2.056160 | 0.42x | **2.11x** | PASSED |
| **5,000,000** | 4.079500 | 0.612544 | 8.349984 | 0.49x | **6.66x** | PASSED |
| **10,000,000** | 8.895000 | 0.567648 | 15.294400 | 0.58x | **15.67x** | PASSED |
| **20,000,000** | 17.531900 | 0.927744 | 32.451553 | 0.54x | **18.90x** | PASSED |
| **50,000,000** | 45.869300 | 1.557568 | 77.176125 | 0.59x | **29.45x** | PASSED |

#### Iteration 3
| Dataset Size | CPU Time (ms) | CUDA Kernel (ms) | Total CUDA (ms) | Speedup (Total) | Speedup (Kernel) | Verification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1,000,000** | 0.930900 | 0.568320 | 2.369536 | 0.39x | **1.64x** | PASSED |
| **5,000,000** | 4.289200 | 0.558944 | 7.999680 | 0.54x | **7.67x** | PASSED |
| **10,000,000** | 8.336500 | 0.623392 | 17.220928 | 0.48x | **13.37x** | PASSED |
| **20,000,000** | 16.712100 | 0.905088 | 29.339487 | 0.57x | **18.46x** | PASSED |
| **50,000,000** | 46.652800 | 1.591744 | 85.198433 | 0.55x | **29.31x** | PASSED |

#### Iteration 4
| Dataset Size | CPU Time (ms) | CUDA Kernel (ms) | Total CUDA (ms) | Speedup (Total) | Speedup (Kernel) | Verification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1,000,000** | 0.742500 | 0.370912 | 2.126944 | 0.35x | **2.00x** | PASSED |
| **5,000,000** | 4.736700 | 0.446112 | 8.466944 | 0.56x | **10.62x** | PASSED |
| **10,000,000** | 8.192100 | 0.710240 | 16.504736 | 0.50x | **11.53x** | PASSED |
| **20,000,000** | 17.698500 | 0.898880 | 29.836704 | 0.59x | **19.69x** | PASSED |
| **50,000,000** | 49.854700 | 1.532576 | 79.208031 | 0.63x | **32.53x** | PASSED |

#### Iteration 5
| Dataset Size | CPU Time (ms) | CUDA Kernel (ms) | Total CUDA (ms) | Speedup (Total) | Speedup (Kernel) | Verification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1,000,000** | 1.497300 | 0.596096 | 2.499104 | 0.60x | **2.51x** | PASSED |
| **5,000,000** | 5.322900 | 0.777504 | 8.387168 | 0.63x | **6.85x** | PASSED |
| **10,000,000** | 8.124600 | 0.929984 | 15.336000 | 0.53x | **8.74x** | PASSED |
| **20,000,000** | 17.256000 | 0.959104 | 29.139072 | 0.59x | **17.99x** | PASSED |
| **50,000,000** | 44.498500 | 1.702016 | 80.912544 | 0.55x | **26.14x** | PASSED |

---

### Final Average Summary Table

$$\text{Average Time} = \frac{\text{Run}_1 + \text{Run}_2 + \text{Run}_3 + \text{Run}_4 + \text{Run}_5}{5}$$

| Dataset Size | Avg CPU Time (ms) | Avg CUDA Kernel (ms) | Avg Total CUDA (ms) | Avg Speedup (Total) | Avg Speedup (Kernel) | Verification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1,000,000** | 0.967180 | 0.473472 | 2.217754 | 0.43x | **2.04x** | PASSED |
| **5,000,000** | 4.723660 | 0.592422 | 8.320326 | 0.57x | **7.97x** | PASSED |
| **10,000,000** | 8.449180 | 0.697766 | 16.268493 | 0.52x | **12.11x** | PASSED |
| **20,000,000** | 17.427000 | 0.930042 | 30.651911 | 0.57x | **18.74x** | PASSED |
| **50,000,000** | 46.176860 | 1.594195 | 80.049689 | 0.58x | **28.97x** | PASSED |

---

## Calculating Speedup

Speedup quantifies how many times faster the parallel CUDA implementation performs compared to the sequential CPU baseline.

1. **Total System Speedup (End-to-End):** Includes data transfer over the PCIe bus ($\text{Host} \to \text{Device}$ and $\text{Device} \to \text{Host}$).
   $$\text{Speedup}_{\text{Total}} = \frac{\text{CPU Execution Time}}{\text{Total CUDA Execution Time}}$$

2. **Compute-Only Speedup (Kernel Performance):** Isolates pure GPU parallel processing performance from bus transfer latency.
   $$\text{Speedup}_{\text{Kernel}} = \frac{\text{CPU Execution Time}}{\text{CUDA Kernel Time}}$$

---

## Performance Graphs & Visualizations

### 1. Compute-Only Kernel Speedup Scaling

Shows how GPU compute core performance scales relative to sequential CPU execution as dataset size increases:
<img width="2700" height="1500" alt="cuda_speedup_analysis" src="https://github.com/user-attachments/assets/ad0d4070-3c64-4ee2-bf62-782c81a49e1d" />


### 2. Execution Time Breakdown

Visualizing where execution time is spent during CUDA processing vs. CPU processing:

<img width="3000" height="1800" alt="cuda_time_comparison" src="https://github.com/user-attachments/assets/4565fac7-4d47-4d5c-be0f-14deba1e00cf" />


---

## Deep-Dive Technical Analysis

### 1. Why Total CUDA Time Exceeds CPU Time (Small Datasets)
For lighter or simple math workloads, Total CUDA Time often exceeds CPU time, yielding an end-to-end speedup below $1.0\times$:

- **PCIe Data Transfer Overhead:** Copying data from Host RAM to Device VRAM via `cudaMemcpy` introduces high fixed transfer latency across the PCIe bus.
- **Driver and Context Initialization:** Allocating GPU memory (`cudaMalloc`) and initializing CUDA context/kernel launches adds fixed time overhead.
- **Low Arithmetic Density:** For an $\mathcal{O}(N)$ element-wise operation ($\times 2.0$), the algorithm performs only 1 floating-point operation per load/store. The time spent moving data across the PCIe bus dwarfs the computation time.

### 2. Why Compare CUDA Kernel Time vs. CPU Time?
Comparing CUDA Kernel Time directly to CPU Time isolates raw processing throughput from bus bottlenecks:

- **Raw Parallel Capability:** Demonstrates the true compute potential of thousands of GPU threads executing simultaneously without bus transfer noise.
- **Architecture Potential:** Highlights how massively parallel hardware executes element-wise workloads once data resides in high-bandwidth VRAM.
- **Optimization Guidance:** Helps engineers identify whether an application is constrained by algorithm compute density or PCIe memory bandwidth, directing future optimizations.

---

## Key Takeaways

- **Massive Compute Potential:** Raw GPU compute core performance scales exceptionally well with data size—achieving **~28.97x** speedup on the kernel alone for 50 million elements.
- **Increasing GPU Efficiency:** As the dataset size increases, the GPU is able to utilize its parallel processing resources more effectively, with kernel speedup increasing from **2.04x for 1 million elements to 28.97x for 50 million elements**.
- **PCIe Transfer Bottleneck:** For simple element-wise array operations with low arithmetic intensity ($1 \text{ FLOP}/\text{element}$), data transfers account for **~98%** of total CUDA execution time.
- **Future Optimizations:** To achieve end-to-end speedups ($> 1.0\times$) on simple math operations, consider:
  - **Pinned Memory (`cudaHostAlloc`):** Unlocks higher transfer bandwidth across the PCIe bus.
  - **Asynchronous Streams (`cudaMemcpyAsync`):** Overlaps memory transfers with active GPU kernel computation.
  - **In-Place Pipelines:** Keeps data on the GPU across sequential processing kernels rather than copying back to host RAM between steps.

---

## How to Run
- Running the CUDA Benchmark Suite
  Automate compilation and execution across all 5 dataset sizes and iterations:
  
  ```bash
  # Make script executable
  chmod +x benchmark.sh
  
  # Run benchmark script (generates benchmark_results.log)
  ./benchmark.sh
  ```

- Parsing Log Files & Generating Reports
  Process `benchmark_results.log` to aggregate averages and generate summary charts:
  
  ```bash
  python parse-sysbench.py
  ```

- Standalone Graph Generation (`generate_plots.py`)
   1. Prerequisites
    Install required Python visualization packages:
    ```bash
    pip install matplotlib numpy
    ```
   2. Execution
    Run the standalone plotting script:
    ```bash
    python generate_plots.py
    ```
