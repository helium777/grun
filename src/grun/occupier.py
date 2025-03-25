import sys
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
import setproctitle
from typing import List

console = Console()


def check_numba_available():
    """Check if numba is available and CUDA is accessible."""
    try:
        from numba import cuda

        if not cuda.is_available():
            console.print("[red]Error: No CUDA device detected[/red]")
            sys.exit(1)
        return cuda
    except ImportError:
        console.print(
            Panel.fit(
                "[red]Error: numba is required for GPU occupation\n"
                "Please install dependencies using one of these commands:\n\n"
                "[yellow]pip install 'grun[occupy]'[/yellow]\n"
                "# Or directly install dependencies:\n"
                "[yellow]pip install numba numpy[/yellow]",
                title="Missing Dependencies",
            )
        )
        sys.exit(1)


def occupy_gpu_memory_and_sm(
    memory_gb: float,
    num_gpus: int = 1,
    gpu_indices: List[int] = None,
    process_name: str = "python train.py",
):
    """
    Occupy specified amount of GPU memory and compute resources on multiple GPUs.

    Args:
        memory_gb: Amount of GPU memory to occupy in GB per GPU
        num_gpus: Number of GPUs to occupy
        process_name: Name to display in process list
        gpu_indices: List of specific GPU indices to occupy. If None, uses first N GPUs
    """
    cuda = check_numba_available()

    # Get available GPU count
    device_count = len(cuda.gpus)

    # Validate GPU indices
    invalid_gpus = [idx for idx in gpu_indices if idx >= device_count]
    if invalid_gpus:
        console.print(f"[red]Error: Invalid GPU indices: {invalid_gpus}[/red]")
        sys.exit(1)

    # Set process name
    setproctitle.setproctitle(process_name)

    # Convert GB to bytes
    memory_bytes = int(memory_gb * 1024 * 1024 * 1024)
    n_floats = memory_bytes // 4  # float32 uses 4 bytes

    try:
        # Define compute kernel with more stable computation
        @cuda.jit
        def busy_kernel(data):
            idx = cuda.grid(1)
            if idx < data.size:
                x = float(idx)  # Use float to avoid integer overflow
                for i in range(10):
                    x = (x * x + 1.0) % 1000.0  # More stable computation
                data[idx] = x

        # Show occupation status
        console.print(
            Panel.fit(
                f"[green]Successfully allocated {memory_gb}GB GPU memory on {len(gpu_indices)} GPU(s)\n"
                f"GPU indices: {gpu_indices}\n"
                "[yellow]Press Ctrl+C to stop[/yellow]",
                title="GPU Occupier Status",
            )
        )

        with console.status("Occupying GPU compute resources..."):
            # Allocate memory for each GPU
            memory_occupiers = {}
            for gpu_idx in gpu_indices:
                try:
                    with cuda.gpus[gpu_idx]:
                        memory_occupiers[gpu_idx] = cuda.device_array(
                            n_floats, dtype=np.float32
                        )
                except Exception as e:
                    console.print(f"[red]Error initializing GPU {gpu_idx}: {e}[/red]")
                    sys.exit(1)

            while True:
                try:
                    # Run kernel on each GPU
                    for gpu_idx in gpu_indices:
                        with cuda.gpus[gpu_idx]:
                            # Get optimal configuration for this device
                            threads_per_block = 256
                            max_blocks_per_grid = cuda.gpus[gpu_idx].MAX_GRID_DIM_X

                            # Calculate optimal blocks per grid
                            blocks_per_grid = min(
                                (n_floats + threads_per_block - 1) // threads_per_block,
                                max_blocks_per_grid,
                            )

                            # Run kernel with error handling
                            try:
                                busy_kernel[blocks_per_grid, threads_per_block](
                                    memory_occupiers[gpu_idx]
                                )
                                cuda.synchronize()
                            except Exception as e:
                                console.print(f"[red]Error on GPU {gpu_idx}: {e}[/red]")
                                continue

                    time.sleep(0.5)  # Reduce CPU usage
                except KeyboardInterrupt:
                    console.print("\n[yellow]Received interrupt signal, cleaning up...[/yellow]")
                    break
                except Exception as e:
                    console.print(f"[red]Error during GPU occupation: {e}[/red]")
                    break

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)
