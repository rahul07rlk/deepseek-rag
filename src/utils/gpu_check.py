"""GPU diagnostics + VRAM probes."""
import torch
from rich.table import Table

from src.utils.logger import console, get_logger

logger = get_logger("gpu_check", "indexer.log")


def run_gpu_diagnostics() -> None:
    table = Table(title="GPU Diagnostic Report", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    table.add_row("CUDA Version", torch.version.cuda or "N/A")
    table.add_row("PyTorch Version", torch.__version__)
    table.add_row("Device Count", str(torch.cuda.device_count()))
    console.print(table)

    if not torch.cuda.is_available():
        logger.warning("No CUDA GPU found. System will run on CPU.")
        return

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpu_table = Table(title=f"cuda:{i} - {props.name}")
        gpu_table.add_column("Spec", style="cyan")
        gpu_table.add_column("Value", style="yellow")
        gpu_table.add_row("Total VRAM", f"{props.total_memory / 1024**3:.2f} GB")
        gpu_table.add_row("SMs", str(props.multi_processor_count))
        gpu_table.add_row("Compute Capability", f"{props.major}.{props.minor}")
        console.print(gpu_table)


def check_vram_after_model_load(device: str) -> None:
    if "cuda" not in device:
        return
    idx = int(device.split(":")[-1]) if ":" in device else 0
    allocated = torch.cuda.memory_allocated(idx) / 1024**2
    reserved = torch.cuda.memory_reserved(idx) / 1024**2
    total = torch.cuda.get_device_properties(idx).total_memory / 1024**2
    free = total - reserved
    logger.info(
        f"VRAM - Used: {allocated:.0f}MB | Reserved: {reserved:.0f}MB | "
        f"Free: {free:.0f}MB | Total: {total:.0f}MB"
    )


if __name__ == "__main__":
    run_gpu_diagnostics()
