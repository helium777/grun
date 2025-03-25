# grun - GPU Resource Manager

A simple command-line tool to run scripts when GPU resources are available.

## Features

- Monitor GPU memory and utilization using NVIDIA Management Library (NVML)
- Run commands when sufficient GPU resources are available
- Smart GPU selection strategies:
  - Default: Select GPUs with lowest utilization
  - Exclusive: Only use GPUs with no other running processes
- Detailed GPU status reporting (memory, utilization, active processes)
- Configurable check interval
- Option to occupy GPU memory and keep it busy with computation

## Requirements

- Python 3.9+
- NVIDIA GPU with NVIDIA drivers installed
- CUDA toolkit (for GPU occupation feature)

## Installation

Since this package is currently in demo stage and not available on PyPI, you can install it directly from GitHub using `uv`:

```bash
# Install using uv
uv tool install git+https://github.com/helium777/grun.git

# If you want to use the GPU occupation feature, install with the 'occupy' extra
uv tool install "grun[occupy] @ git+https://github.com/helium777/grun.git"
```

Alternatively, you can install it using pip:

```bash
# Install using pip
pip install git+https://github.com/helium7/grun.git

# If you want to use the GPU occupation feature, install with the 'occupy' extra
pip install "grun[occupy] @ git+https://github.com/helium7/grun.git"
```

## Usage

Basic syntax:
```bash
grun --mem <memory_in_gb> [--strategy <strategy>] <command>
```

Available strategies:
- `utilization` (default): Prefer GPUs with lower utilization
- `exclusive`: Only use GPUs with no other processes

### Examples

1. Run a training script using default strategy (lowest utilization):
```bash
grun --mem 40 python train.py
```

2. Run a script on exclusive GPUs:
```bash
grun --mem 32 --strategy exclusive python train.py
```

3. Run a script with custom check interval (e.g., check every 5 seconds):
```bash
grun --mem 32 --interval 5 python my_script.py
```

4. Run a command with arguments:
```bash
grun --mem 16 python train.py --batch-size 32 --epochs 100
```

5. Occupy GPU memory and keep it busy:
```bash
grun --mem 16 --occupy
```

## How it works

1. The tool uses NVIDIA Management Library (NVML) to monitor:
   - Available GPU memory
   - GPU utilization
   - Running processes
2. Based on the selected strategy:
   - Default (`utilization`): Selects GPUs with lowest utilization and sufficient memory
   - `exclusive`: Only selects GPUs with no other processes running
3. When suitable GPUs are found:
   - Sets the `CUDA_VISIBLE_DEVICES` environment variable
   - Either runs the specified command or occupies the GPU (if --occupy is used)
4. The process continues until the command completes or is interrupted

## Notes

- Memory requirement is specified in gigabytes (GB)
- The tool will wait indefinitely until suitable GPUs are found
- Use Ctrl+C to stop waiting for resources or to stop GPU occupation
- The tool properly initializes and cleans up NVML resources
- When using --occupy, the tool will:
  - Allocate the specified amount of GPU memory
  - Keep the GPU busy with continuous computation
  - Clean up resources when interrupted with Ctrl+C

## Development

To contribute to the development of `grun`:

1. Clone the repository
2. Install development dependencies:
```bash
pip install -e ".[occupy]"
```
