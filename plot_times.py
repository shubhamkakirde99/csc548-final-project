import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the JSON data
with open("times.json", "r") as file:
    data = json.load(file)

# Extract matrix sizes and implementations
sizes = []
serial_times = []
mpi_times = []
cuda_times = []

for experiment, times in data.items():
    # Extract size from experiment name (e.g., "experiment1_1000" -> 1000)
    size = int(experiment.split("_")[1])
    sizes.append(size)

    # Extract times, replacing -1 with NaN for plotting (to skip invalid data)
    serial_times.append(times["serial"] if times["serial"] != -1 else np.nan)
    mpi_times.append(times["mpi"] if times["mpi"] != -1 else np.nan)
    cuda_times.append(times["cuda"] if times["cuda"] != -1 else np.nan)

# Sort all lists by size
sorted_data = sorted(zip(sizes, serial_times, mpi_times, cuda_times))
sizes = [item[0] for item in sorted_data]
serial_times = [item[1] for item in sorted_data]
mpi_times = [item[2] for item in sorted_data]
cuda_times = [item[3] for item in sorted_data]

# Create plot
plt.figure(figsize=(12, 7))

# Set width of bars
bar_width = 0.25
x = np.arange(len(sizes))

# Create bars
plt.bar(x - bar_width, serial_times, width=bar_width, label="Serial", color="blue")
plt.bar(x, mpi_times, width=bar_width, label="MPI", color="green")
plt.bar(x + bar_width, cuda_times, width=bar_width, label="CUDA", color="red")

# Add labels and title
plt.xlabel("Matrix Size")
plt.ylabel("Execution Time (seconds)")
plt.title("Matrix Multiplication Performance Comparison")
plt.xticks(x, sizes)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Add a secondary plot with a better view of CUDA times
plt.figure(figsize=(12, 7))
plt.bar(x, cuda_times, label="CUDA", color="red")
plt.xlabel("Matrix Size")
plt.ylabel("Execution Time (seconds)")
plt.title("CUDA Implementation Performance")
plt.xticks(x, sizes)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Calculate and display speedups
if not all(np.isnan(serial_times)):
    plt.figure(figsize=(12, 7))

    # Calculate speedups (avoiding division by zero or NaN)
    mpi_speedup = [
        s / m if not np.isnan(s) and not np.isnan(m) and m > 0 else np.nan
        for s, m in zip(serial_times, mpi_times)
    ]
    cuda_speedup = [
        s / c if not np.isnan(s) and not np.isnan(c) and c > 0 else np.nan
        for s, c in zip(serial_times, cuda_times)
    ]

    # Create bar chart for speedups
    plt.bar(
        x - bar_width / 2,
        mpi_speedup,
        width=bar_width,
        label="MPI Speedup",
        color="green",
    )
    plt.bar(
        x + bar_width / 2,
        cuda_speedup,
        width=bar_width,
        label="CUDA Speedup",
        color="red",
    )

    plt.xlabel("Matrix Size")
    plt.ylabel("Speedup (relative to Serial)")
    plt.title("Performance Speedup Comparison")
    plt.xticks(x, sizes)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
plt.figure(1)
plt.savefig(os.path.join(output_dir, "execution_times.png"))
plt.figure(2)
plt.savefig(os.path.join(output_dir, "cuda_times.png"))
if not all(np.isnan(serial_times)):
    plt.figure(3)
    plt.savefig(os.path.join(output_dir, "speedup_comparison.png"))

# Show plots
plt.show()
