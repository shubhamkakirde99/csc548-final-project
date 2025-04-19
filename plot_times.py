import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)


def plot_size_data():
    # Load the JSON data
    with open("times_size.json", "r") as file:
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
    # Add blur radius text
    plt.figtext(0.02, 0.02, "blur_radius=100px", fontsize=10, ha="left")
    # Add image size text
    plt.figtext(0.98, 0.02, "image_size=3500px×3500px", fontsize=10, ha="right")

    # Add a secondary plot with a better view of CUDA times
    plt.figure(figsize=(12, 7))
    plt.bar(x, cuda_times, label="CUDA", color="red")
    plt.xlabel("Matrix Size")
    plt.ylabel("Execution Time (seconds)")
    plt.title("CUDA Implementation Performance")
    plt.xticks(x, sizes)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # Add blur radius text
    plt.figtext(0.02, 0.02, "blur_radius=100px", fontsize=10, ha="left")
    # Add image size text
    plt.figtext(0.98, 0.02, "image_size=3500px×3500px", fontsize=10, ha="right")

    # Add a plot for MPI times
    plt.figure(figsize=(12, 7))
    plt.bar(x, mpi_times, label="MPI", color="green")
    plt.xlabel("Matrix Size")
    plt.ylabel("Execution Time (seconds)")
    plt.title("MPI Implementation Performance")
    plt.xticks(x, sizes)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # Add blur radius text
    plt.figtext(0.02, 0.02, "blur_radius=100px", fontsize=10, ha="left")
    # Add image size text
    plt.figtext(0.98, 0.02, "image_size=3500px×3500px", fontsize=10, ha="right")

    # Add a plot for Serial times
    plt.figure(figsize=(12, 7))
    plt.bar(x, serial_times, label="Serial", color="blue")
    plt.xlabel("Matrix Size")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Serial Implementation Performance")
    plt.xticks(x, sizes)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # Add blur radius text
    plt.figtext(0.02, 0.02, "blur_radius=100px", fontsize=10, ha="left")
    # Add image size text
    plt.figtext(0.98, 0.02, "image_size=3500px×3500px", fontsize=10, ha="right")

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
        # Add blur radius text
        plt.figtext(0.02, 0.02, "blur_radius=100px", fontsize=10, ha="left")
        # Add image size text
        plt.figtext(0.98, 0.02, "image_size=3500px×3500px", fontsize=10, ha="right")

    # Save plots
    plt.figure(1)
    plt.savefig(os.path.join(output_dir, "size_execution_times.png"))
    plt.figure(2)
    plt.savefig(os.path.join(output_dir, "size_cuda_times.png"))
    plt.figure(3)
    plt.savefig(os.path.join(output_dir, "size_mpi_times.png"))
    plt.figure(4)
    plt.savefig(os.path.join(output_dir, "size_serial_times.png"))
    if not all(np.isnan(serial_times)):
        plt.figure(5)
        plt.savefig(os.path.join(output_dir, "size_speedup_comparison.png"))


def plot_radius_data():
    # Load the JSON data
    with open("times_radius.json", "r") as file:
        data = json.load(file)

    # Extract radius values and implementation times
    radii = []
    serial_times = []
    mpi_times = []
    cuda_times = []

    for radius, times in data.items():
        radii.append(int(radius))
        serial_times.append(times["serial"])
        mpi_times.append(times["mpi"])
        cuda_times.append(times["cuda"])

    # Sort all lists by radius
    sorted_data = sorted(zip(radii, serial_times, mpi_times, cuda_times))
    radii = [item[0] for item in sorted_data]
    serial_times = [item[1] for item in sorted_data]
    mpi_times = [item[2] for item in sorted_data]
    cuda_times = [item[3] for item in sorted_data]

    # Plot serial implementation
    plt.figure(figsize=(12, 7))
    plt.plot(radii, serial_times, marker="o", color="blue", linewidth=2)
    plt.xlabel("Blur Radius")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Serial Implementation Performance by Blur Radius")
    plt.grid(True)
    plt.figtext(0.98, 0.02, "image_size=3500px×3500px", fontsize=10, ha="right")
    plt.savefig(os.path.join(output_dir, "radius_serial_times.png"))

    # Plot MPI implementation
    plt.figure(figsize=(12, 7))
    plt.plot(radii, mpi_times, marker="o", color="green", linewidth=2)
    plt.xlabel("Blur Radius")
    plt.ylabel("Execution Time (seconds)")
    plt.title("MPI Implementation Performance by Blur Radius")
    plt.grid(True)
    plt.figtext(0.98, 0.02, "image_size=3500px×3500px", fontsize=10, ha="right")
    plt.savefig(os.path.join(output_dir, "radius_mpi_times.png"))

    # Plot CUDA implementation
    plt.figure(figsize=(12, 7))
    plt.plot(radii, cuda_times, marker="o", color="red", linewidth=2)
    plt.xlabel("Blur Radius")
    plt.ylabel("Execution Time (seconds)")
    plt.title("CUDA Implementation Performance by Blur Radius")
    plt.grid(True)
    plt.figtext(0.98, 0.02, "image_size=3500px×3500px", fontsize=10, ha="right")
    plt.savefig(os.path.join(output_dir, "radius_cuda_times.png"))


# Run the plotting functions
plot_size_data()
plot_radius_data()

# Show plots
plt.show()
