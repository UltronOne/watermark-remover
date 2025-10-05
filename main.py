"""
Modern GPU-optimized watermark remover CLI
"""
import typer
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from loguru import logger
import time
import sys

from config import settings, ProcessingConfig, ModelConfig
from processor import ModernProcessor, ProcessingResult


# Initialize Typer app
app = typer.Typer(
    name="watermark-remover",
    help="Modern GPU-optimized watermark removal tool",
    rich_markup_mode="rich"
)

# Initialize console
console = Console()


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Setup logging configuration"""
    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days"
        )


@app.command()
def process(
    input_path: Path = typer.Argument(..., help="Input file or directory"),
    output_path: Path = typer.Argument(..., help="Output file or directory"),
    transparent: bool = typer.Option(
        False, "--transparent", "-t", help="Make watermarks transparent"),
    overwrite: bool = typer.Option(
        False, "--overwrite", "-f", help="Overwrite existing files"),
    max_bbox_percent: float = typer.Option(
        10.0, "--max-bbox", "-b", help="Max bounding box percentage"),
    batch_size: int = typer.Option(
        4, "--batch-size", "-bs", help="Batch size for processing"),
    num_workers: int = typer.Option(
        4, "--workers", "-w", help="Number of worker processes"),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Device to use (auto/cuda/cpu)"),
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l", help="Logging level"),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", help="Log file path")
):
    """Process images/videos to remove watermarks"""

    # Setup logging
    setup_logging(log_level, log_file)

    # Update settings
    settings.processing.max_bbox_percent = max_bbox_percent
    settings.processing.batch_size = batch_size
    settings.processing.num_workers = num_workers
    settings.processing.device = device
    settings.transparent = transparent
    settings.overwrite = overwrite

    # Validate paths
    if not input_path.exists():
        console.print(
            f"[red]Error: Input path does not exist: {input_path}[/red]")
        raise typer.Exit(1)

    # Create output directory if needed
    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)

    # Initialize processor
    console.print(
        Panel.fit("🚀 Initializing Modern Watermark Remover", style="bold blue"))

    try:
        processor = ModernProcessor()
    except Exception as e:
        console.print(f"[red]Error initializing processor: {e}[/red]")
        raise typer.Exit(1)

    # Process files
    start_time = time.time()

    if input_path.is_file():
        # Single file processing
        console.print(f"\n📁 Processing single file: {input_path.name}")

        if processor._is_image(input_path):
            result = processor.process_image(
                input_path, output_path, transparent)
        elif processor._is_video(input_path):
            result = processor.process_video(input_path, output_path)
        else:
            console.print(
                f"[red]Error: Unsupported file type: {input_path.suffix}[/red]")
            raise typer.Exit(1)

        results = [result]

    else:
        # Batch processing
        console.print(f"\n📁 Processing directory: {input_path}")

        # Find all supported files
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg")) + \
            list(input_path.glob("*.png")) + list(input_path.glob("*.webp"))
        video_files = list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi")) + \
            list(input_path.glob("*.mov")) + list(input_path.glob("*.mkv"))

        all_files = image_files + video_files

        if not all_files:
            console.print(
                "[yellow]No supported files found in directory[/yellow]")
            raise typer.Exit(0)

        console.print(
            f"Found {len(image_files)} images and {len(video_files)} videos")

        # Process batch
        results = processor.process_batch(all_files, output_path)

    # Display results
    total_time = time.time() - start_time
    display_results(results, total_time)


def display_results(results: List[ProcessingResult], total_time: float):
    """Display processing results"""

    # Create results table
    table = Table(title="Processing Results")
    table.add_column("File", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Time", style="yellow")
    table.add_column("Detections", style="blue")
    table.add_column("Error", style="red")

    successful = 0
    failed = 0
    total_detections = 0

    for result in results:
        status = "✅ Success" if result.success else "❌ Failed"
        detections = len(result.detections) if result.detections else 0
        error = result.error_message[:50] + "..." if result.error_message and len(
            result.error_message) > 50 else result.error_message or ""

        table.add_row(
            result.input_path.name,
            status,
            f"{result.processing_time:.2f}s",
            str(detections),
            error
        )

        if result.success:
            successful += 1
            total_detections += detections
        else:
            failed += 1

    console.print(table)

    # Summary
    summary_table = Table(title="Summary")
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Total Files", str(len(results)))
    summary_table.add_row("Successful", str(successful))
    summary_table.add_row("Failed", str(failed))
    summary_table.add_row("Total Detections", str(total_detections))
    summary_table.add_row("Total Time", f"{total_time:.2f}s")
    summary_table.add_row("Average Time/File",
                          f"{total_time/len(results):.2f}s")

    console.print(summary_table)

    if failed > 0:
        console.print(
            f"\n[red]⚠️  {failed} files failed to process. Check logs for details.[/red]")
    else:
        console.print(
            f"\n[green]🎉 All {successful} files processed successfully![/green]")


@app.command()
def info():
    """Display system information"""

    # System info table
    table = Table(title="System Information")
    table.add_column("Component", style="bold")
    table.add_column("Status", style="green")
    table.add_column("Details", style="cyan")

    # Python version
    import sys
    table.add_row("Python", "✅", f"{sys.version}")

    # PyTorch
    try:
        import torch
        table.add_row("PyTorch", "✅", f"{torch.__version__}")

        # CUDA
        if torch.cuda.is_available():
            table.add_row(
                "CUDA", "✅", f"{torch.version.cuda} - {torch.cuda.get_device_name()}")
            table.add_row(
                "GPU Memory", "✅", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            table.add_row("CUDA", "❌", "Not available")

        # MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            table.add_row("MPS", "✅", "Apple Silicon GPU available")
        else:
            table.add_row("MPS", "❌", "Not available")

    except ImportError:
        table.add_row("PyTorch", "❌", "Not installed")

    # Other dependencies
    dependencies = [
        ("transformers", "Transformers"),
        ("diffusers", "Diffusers"),
        ("opencv-python", "OpenCV"),
        ("PIL", "Pillow"),
        ("ffmpeg-python", "FFmpeg"),
    ]

    for module, name in dependencies:
        try:
            __import__(module)
            table.add_row(name, "✅", "Available")
        except ImportError:
            table.add_row(name, "❌", "Not installed")

    console.print(table)


@app.command()
def download_models():
    """Download required AI models"""
    console.print(Panel.fit("📥 Downloading AI Models", style="bold blue"))

    try:
        from models import ModernWatermarkDetector, ModernInpainter

        console.print("Downloading detection models...")
        detector = ModernWatermarkDetector()

        console.print("Downloading inpainting models...")
        inpainter = ModernInpainter()

        console.print("[green]✅ All models downloaded successfully![/green]")

    except Exception as e:
        console.print(f"[red]❌ Error downloading models: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
