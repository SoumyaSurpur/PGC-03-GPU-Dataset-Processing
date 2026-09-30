#!/usr/bin/env python3
import re
import pandas as pd
import matplotlib.pyplot as plt

LOG_FILE = "benchmark_results.log"

def parse_benchmark_log(file_path):
    data = []
    
    current_size = None
    cpu_time = None
    kernel_time = None
    total_time = None
    verification = None

    with open(file_path, 'r') as f:
        for line in f:
            size_match = re.search(r'Dataset Size\s*=\s*(\d+)', line)
            cpu_match = re.search(r'CPU Execution Time\s*=\s*([\d\.]+)\s*ms', line)
            kernel_match = re.search(r'CUDA Kernel Time\s*=\s*([\d\.]+)\s*ms', line)
            total_match = re.search(r'Total CUDA Time\s*=\s*([\d\.]+)\s*ms', line)
            verif_match = re.search(r'Verification\s*=\s*(\w+)', line)

            if size_match:
                current_size = int(size_match.group(1))
            if cpu_match:
                cpu_time = float(cpu_match.group(1))
            if kernel_match:
                kernel_time = float(kernel_match.group(1))
            if total_match:
                total_time = float(total_match.group(1))
            if verif_match:
                verification = verif_match.group(1)
                
                # Append collected entry
                data.append({
                    'Dataset Size': current_size,
                    'CPU Time (ms)': cpu_time,
                    'CUDA Kernel (ms)': kernel_time,
                    'Total CUDA (ms)': total_time,
                    'Verification': verification
                })

    return pd.DataFrame(data)

def generate_graphs(df_avg):
    sizes = [f"{s // 1000000}M" for s in df_avg['Dataset Size']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Graph 1: Execution Times comparison
    ax1.plot(sizes, df_avg['CPU Time (ms)'], marker='o', label='CPU Time (ms)', color='red')
    ax1.plot(sizes, df_avg['Total CUDA (ms)'], marker='s', label='Total CUDA (ms)', color='orange')
    ax1.plot(sizes, df_avg['CUDA Kernel (ms)'], marker='^', label='CUDA Kernel (ms)', color='green')
    ax1.set_title('Execution Time Comparison')
    ax1.set_xlabel('Dataset Size')
    ax1.set_ylabel('Time (ms)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Graph 2: Speedup comparison
    ax2.plot(sizes, df_avg['Speedup (Kernel)'], marker='^', label='Kernel Speedup (Compute Only)', color='green')
    ax2.plot(sizes, df_avg['Speedup (Total)'], marker='s', label='Total System Speedup', color='orange')
    ax2.axhline(y=1.0, color='gray', linestyle=':', label='Baseline (1.0x)')
    ax2.set_title('Speedup Metrics Analysis')
    ax2.set_xlabel('Dataset Size')
    ax2.set_ylabel('Speedup Factor (x)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('cuda_performance_analysis.png', dpi=300)
    print("Graph saved as 'cuda_performance_analysis.png'")

def main():
    df = parse_benchmark_log(LOG_FILE)
    if df.empty:
        print("No log data found. Run benchmark.sh first.")
        return

    # Calculate averages per Dataset Size
    df_avg = df.groupby('Dataset Size', as_index=False).mean()
    
    # Calculate Speedups
    df_avg['Speedup (Kernel)'] = df_avg['CPU Time (ms)'] / df_avg['CUDA Kernel (ms)']
    df_avg['Speedup (Total)'] = df_avg['CPU Time (ms)'] / df_avg['Total CUDA (ms)']

    # Display Parsed Summary
    print("\n================ FINAL AVERAGED SUMMARY ================")
    print(df_avg[['Dataset Size', 'CPU Time (ms)', 'CUDA Kernel (ms)', 'Total CUDA (ms)', 'Speedup (Kernel)', 'Speedup (Total)']])
    
    # Generate visualization plots
    generate_graphs(df_avg)

if __name__ == "__main__":
    main()
