# Team Members:

Shubham Umesh Kakirde (skakird@ncsu.edu)

# Gaussian Blur Implementation

This project implements a Gaussian blur filter for PNG images using three different approaches:

- Serial implementation (C)
- Parallel implementation using MPI
- GPU-accelerated implementation using CUDA

## Project Structure

```
.
├── include/                 # Header files and utility functions
│   ├── util.h               # Declarations for PNG I/O utilities
│   └── util.c               # Implementation of PNG I/O utilities
├── serial.c                 # Serial implementation of Gaussian blur
├── mpi.c                    # MPI-based parallel implementation
├── cuda.cu                  # CUDA implementation for GPU acceleration
├── Makefile                 # Build automation
├── plot_times.py            # Python script to plot performance data
├── times_radius.json        # Performance data with varying blur radius
├── times_size.json          # Performance data with varying image sizes
└── .gitignore               # Git ignore file
```

## Requirements

- GCC compiler
- CUDA toolkit (for GPU implementation)
- MPI library (for parallel implementation)
- libpng library (for all implementations)
- Python with matplotlib (for plotting results)

## Building and Running the Project

The project includes a Makefile with targets for all three implementations.

### Serial Implementation

```bash
# Compile
make compileserial

# Run (with default parameters)
make serial

# Run (with custom parameters)
./serial <blur_radius> <image_path>
```

### MPI Implementation

```bash
# Compile
make compilempi

# Run (with default parameters)
make mpi

# Run (with custom parameters)
mpirun -np <num_processes> ./mpi <blur_radius> <image_path>
```

### CUDA Implementation

```bash
# Compile
make compilecuda

# Run (with default parameters)
make cuda

# Run (with custom parameters)
./cuda <blur_radius> <image_path>
```

### Cleaning Build Files

```bash
make clean
```

## Makefile Explanation

The Makefile provides several targets:

- `compileserial`, `compilempi`, `compilecuda`: Compile the respective implementations
- `serial`, `mpi`, `cuda`: Compile and run the respective implementations
- `clean`: Remove all compiled binaries and output images

Configuration variables at the top of the Makefile:

- `IMAGE`: Default input image file (e.g., `experiment1_1000.png`)
- `RADIUS`: Default blur radius (e.g., `100`)
- `PROCS`: Number of MPI processes to use (e.g., `16`)

## Performance Analysis

The project includes two JSON files with performance data:

- `times_radius.json`: Execution times with varying blur radius
- `times_size.json`: Execution times with varying image sizes

To visualize this data, run:

```bash
python plot_times.py
```

This generates performance comparison plots in the `plots` directory.

## Implementation Details

### Serial Implementation

The serial implementation processes the Gaussian blur filter in two passes (horizontal and vertical). It uses a separable Gaussian kernel for efficiency.

### MPI Implementation

The MPI version distributes image rows among processes. Each process handles a subset of rows and applies both horizontal and vertical blur passes. Results are gathered at the root process.

### CUDA Implementation

The CUDA implementation offloads computation to the GPU. It uses CUDA kernels for the horizontal and vertical passes, leveraging massive parallelism for significant speedup.

## Results

The implementations show significant performance differences:

- Serial implementation: Baseline performance
- MPI implementation: 7-8x speedup (with 16 processes)
- CUDA implementation: 100-250x speedup (dependent on image size and blur radius)

Detailed performance charts can be generated using the plotting script.
