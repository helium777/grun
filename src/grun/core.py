import argparse
import subprocess
import time
import sys
import os
from typing import List, Optional
import pynvml
import signal
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from enum import Enum
from .config import settings
from .notify import get_notifier


class GpuSelectionStrategy(Enum):
    UTILIZATION = "utilization"  # Default strategy - prefer GPUs with lower utilization
    EXCLUSIVE = "exclusive"  # Only use GPUs with no other processes


console = Console()


def signal_handler(sig, frame):
    console.print("\n[red]Received interrupt signal, shutting down...[/red]")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def run_command_on_gpu(command: List[str], gpu_indices: List[int]) -> None:
    """Run the command on the specified GPUs."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_indices))

    subprocess.run(command, env=env, check=True)


class GpuMonitor:
    def __init__(self):
        pynvml.nvmlInit()

    def __del__(self):
        pynvml.nvmlShutdown()

    def get_gpu_info(self) -> List[dict]:
        """Get comprehensive GPU information using NVML."""
        device_count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # Get GPU utilization
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
            except pynvml.NVMLError:
                gpu_util = 0

            # Get process count
            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                process_count = len(processes)
            except pynvml.NVMLError:
                process_count = 0

            gpus.append(
                {
                    "index": i,
                    "free_memory": info.free,
                    "total_memory": info.total,
                    "used_memory": info.used,
                    "utilization": gpu_util,
                    "process_count": process_count,
                }
            )
        return gpus

    def get_gpu_memory_info(self) -> List[dict]:
        """Get GPU memory information using NVML."""
        gpus = self.get_gpu_info()
        return [
            {"index": gpu["index"], "free_memory": gpu["free_memory"]} for gpu in gpus
        ]

    def find_available_gpu(
        self,
        required_memory_gb: float,
        num_gpus: int = 1,
        strategy: GpuSelectionStrategy = GpuSelectionStrategy.UTILIZATION,
    ) -> Optional[tuple[List[int], List[dict]]]:
        """Find GPUs with sufficient free memory using the specified strategy.

        Args:
            required_memory_gb: Required memory in GB for each GPU
            num_gpus: Number of GPUs needed
            strategy: GPU selection strategy to use

        Returns:
            Tuple of (list of GPU indices, list of GPU info dicts) if found, None otherwise
        """
        required_bytes = int(required_memory_gb * 1024**3)  # Convert GB to bytes
        gpus = self.get_gpu_info()

        if not gpus:
            return None

        # Filter GPUs based on memory requirement
        eligible_gpus = [gpu for gpu in gpus if gpu["free_memory"] >= required_bytes]

        if not eligible_gpus:
            return None

        # Apply different selection strategies
        if strategy == GpuSelectionStrategy.EXCLUSIVE:
            eligible_gpus = [gpu for gpu in eligible_gpus if gpu["process_count"] == 0]
            eligible_gpus.sort(key=lambda x: -x["free_memory"])
        else:  # UTILIZATION strategy (default)
            eligible_gpus.sort(key=lambda x: (x["utilization"], -x["free_memory"]))

        if len(eligible_gpus) >= num_gpus:
            selected_gpus = eligible_gpus[:num_gpus]
            return ([gpu["index"] for gpu in selected_gpus], selected_gpus)
        return None

    def wait_for_gpu(
        self,
        required_memory_gb: float,
        num_gpus: int = 1,
        check_interval: float = 1.0,
        strategy: GpuSelectionStrategy = GpuSelectionStrategy.UTILIZATION,
    ) -> Optional[List[int]]:
        """Wait for available GPUs with sufficient memory."""
        welcome_text = Text()
        welcome_text.append("GPU Resource Manager", style="bold blue")
        welcome_text.append("\nWaiting for ", style="dim")
        welcome_text.append(f"{num_gpus}", style="bold green")
        welcome_text.append(" GPU(s) with ", style="dim")
        welcome_text.append(f"{required_memory_gb}GB", style="bold green")
        welcome_text.append(" free memory each...\n", style="dim")
        welcome_text.append(f"Strategy: {strategy.value}", style="bold yellow")
        console.print(Panel(welcome_text))

        with console.status("[yellow]Searching for available GPUs..."):
            while True:
                result = self.find_available_gpu(required_memory_gb, num_gpus, strategy)
                if result is not None:
                    gpu_indices, selected_gpus = result

                    # Print detailed information about selected GPUs
                    console.print("[green]✓ Found available GPUs:[/green]")
                    for gpu in selected_gpus:
                        # Create a styled GPU info string
                        gpu_info = Text()
                        gpu_info.append(f"  GPU {gpu['index']}", style="bold cyan")
                        gpu_info.append(" | ", style="dim")
                        gpu_info.append(
                            f"{gpu['free_memory'] / 1024**3:.1f}GB free", style="green"
                        )
                        gpu_info.append(" | ", style="dim")

                        # Color utilization based on level
                        util_value = gpu["utilization"]
                        util_style = (
                            "green"
                            if util_value < 30
                            else "yellow"
                            if util_value < 70
                            else "red"
                        )
                        gpu_info.append(f"Utilization: {util_value}%", style=util_style)
                        gpu_info.append(" | ", style="dim")

                        # Color process count based on number
                        proc_count = gpu["process_count"]
                        proc_style = (
                            "green"
                            if proc_count == 0
                            else "yellow"
                            if proc_count < 3
                            else "red"
                        )
                        gpu_info.append(f"Processes: {proc_count}", style=proc_style)

                        console.print(gpu_info)
                    return gpu_indices
                time.sleep(check_interval)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run commands when GPU resources are available"
    )
    parser.add_argument(
        "--mem", type=float, required=True, help="Required GPU memory in GB"
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs needed")
    parser.add_argument(
        "--interval", type=float, default=1.0, help="Check interval in seconds"
    )
    parser.add_argument(
        "--occupy",
        action="store_true",
        help="Only occupy GPU resources without running command",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=[s.value for s in GpuSelectionStrategy],
        default=GpuSelectionStrategy.UTILIZATION.value,
        help="GPU selection strategy (default: utilization)",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")

    args = parser.parse_args()

    if args.occupy and args.command:
        console.print(
            "[red]Error: Cannot specify command when using --occupy option[/red]"
        )
        sys.exit(1)
    elif not args.occupy and not args.command:
        console.print("[red]Error: Must specify a command to run[/red]")
        sys.exit(1)

    return args


def main():
    args = parse_args()

    # Get notification service instance
    notifier = get_notifier()

    gpu_monitor = GpuMonitor()
    strategy = GpuSelectionStrategy(args.strategy)
    gpu_indices = gpu_monitor.wait_for_gpu(
        args.mem,
        args.gpus,
        args.interval,
        strategy,
    )

    if settings.notification.notify_on_gpu_found:
        task_name = " ".join(args.command) if not args.occupy else "<GPU Occupation>"
        notifier.send(
            "GPUs Found",
            f"Task `{task_name}` started on GPUs {gpu_indices}",
        )

    try:
        if args.occupy:
            console.print("\n[yellow]Starting GPU occupier...[/yellow]")
            from .occupier import occupy_gpu_memory_and_sm

            occupy_gpu_memory_and_sm(
                memory_gb=args.mem,
                num_gpus=args.gpus,
                gpu_indices=gpu_indices,
            )
            if settings.notification.notify_on_task_complete:
                notifier.send("Occupation Complete", "")
        else:
            console.print("\n[yellow]Running command...[/yellow]")
            run_command_on_gpu(args.command, gpu_indices)
            if settings.notification.notify_on_task_complete:
                notifier.send("Task Complete", f"Command: {' '.join(args.command)}")
    except Exception as e:
        if settings.notification.notify_on_task_complete:
            task_name = (
                " ".join(args.command) if not args.occupy else "<GPU Occupation>"
            )
            msg = f"Error: {e}, Task: {task_name}"
            notifier.send("Task Failed", msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
