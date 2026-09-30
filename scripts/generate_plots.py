#!/usr/bin/env python3
"""
generate_plots.py
Generates performance analysis charts for CUDA vs. CPU benchmark dataset results.
"""

import matplotlib.pyplot as plt
import numpy as np

# Averaged Benchmark Results
datasets = ["1M", "5M", "10M", "20M", "50M"]
dataset_sizes = [1_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000]

avg_cpu_time = [0.967180, 4.723660, 8.449180, 17.427000, 46.176860]
avg_kernel_time = [0.473472, 0.592422, 0.697766, 0.930042, 1.594195]
avg_total_cuda_time = [2.217754, 8.320326, 16.268493, 30.651911, 80.049689]

speedup_total = [0.43, 0.57, 0.52, 0.57, 0.58]
speedup_kernel = [2.04, 7.97, 12.11, 18.74, 28.97]


def plot_speedups():
    """Generates the Speedup Metrics comparison graph."""
    plt.figure(figsize=(9, 5), dpi=300)

    # Plot lines
    plt.plot(
        datasets,
        speedup_kernel,
        marker="o",
        color="#2ca02c",
        linewidth=2.5,
        markersize=8,
        label="Kernel Speedup (Compute Only)",
    )
    plt.plot(
        datasets,
        speedup_total,
        marker="s",
        color="#ff7f0e",
        linewidth=2.5,
        markersize=8,
        label="Total System Speedup (End-to-End)",
    )

    # Baseline parity line (1.0x)
    plt.axhline(
        y=1.0,
        color="#d62728",
        linestyle="--",
        alpha=0.8,
        linewidth=1.5,
        label="Parity Baseline (1.0x)",
    )

    # Add data labels
    for i, (k_val, t_val) in enumerate(zip(speedup_kernel, speedup_total)):
        plt.annotate(
            f"{k_val:.2f}x",
            (datasets[i], k_val),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontweight="bold",
            color="#2ca02c",
        )
        plt.annotate(
            f"{t_val:.2f}x",
            (datasets[i], t_val),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontweight="bold",
            color="#ff7f0e",
        )

    plt.title(
        "CUDA vs CPU Speedup Analysis Across Dataset Sizes",
        fontsize=14,
        pad=15,
        fontweight="bold",
    )
    plt.xlabel("Dataset Size (Elements)", fontsize=11, fontweight="bold")
    plt.ylabel("Speedup Factor (x)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    filename = "cuda_speedup_analysis.png"
    plt.savefig(filename)
    plt.close()
    print(f"Successfully generated: {filename}")


def plot_execution_times():
    """Generates the Execution Time comparison bar chart."""
    x = np.arange(len(datasets))
    width = 0.25

    plt.figure(figsize=(10, 6), dpi=300)

    plt.bar(x - width, avg_cpu_time, width, label="CPU Time", color="#1f77b4")
    plt.bar(
        x,
        avg_total_cuda_time,
        width,
        label="Total CUDA Time (Transfer + Kernel)",
        color="#ff7f0e",
    )
    plt.bar(
        x + width,
        avg_kernel_time,
        width,
        label="CUDA Kernel Time (Compute Only)",
        color="#2ca02c",
    )

    plt.title(
        "Execution Time Comparison (CPU vs CUDA)",
        fontsize=14,
        pad=15,
        fontweight="bold",
    )
    plt.xlabel("Dataset Size (Elements)", fontsize=11, fontweight="bold")
    plt.ylabel("Time (milliseconds)", fontsize=11, fontweight="bold")
    plt.xticks(x, datasets)
    plt.grid(True, linestyle=":", alpha=0.5, axis="y")
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    filename = "cuda_time_comparison.png"
    plt.savefig(filename)
    plt.close()
    print(f"Successfully generated: {filename}")


if __name__ == "__main__":
    plot_speedups()
    plot_execution_times()
