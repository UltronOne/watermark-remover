"""
Modern high-performance image and video processor
"""
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import ffmpeg
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
from loguru import logger
import time
from dataclasses import dataclass

from config import settings
from models import ModernWatermarkDetector, ModernInpainter


@dataclass
class ProcessingResult:
    """Result of processing operation"""
    success: bool
    input_path: Path
    output_path: Path
    processing_time: float
    error_message: Optional[str] = None
    detections: Optional[List[Dict[str, Any]]] = None


class ModernProcessor:
    """High-performance watermark removal processor"""

    def __init__(self):
        self.device = self._get_device()
        self.detector = ModernWatermarkDetector(device=self.device)
        self.inpainter = ModernInpainter(device=self.device)

        # Performance settings
        self.batch_size = settings.processing.batch_size
        self.num_workers = min(settings.processing.num_workers, mp.cpu_count())

        logger.info(f"Processor initialized on {self.device}")
        logger.info(
            f"Batch size: {self.batch_size}, Workers: {self.num_workers}")

    def _get_device(self) -> str:
        """Get optimal device for processing"""
        if settings.processing.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                # Set optimal CUDA settings
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
                logger.info(
                    f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                logger.info("MPS (Apple Silicon) available")
            else:
                device = "cpu"
                logger.info("Using CPU")
        else:
            device = settings.processing.device

        return device

    def process_image(self, input_path: Path, output_path: Path,
                      transparent: bool = False) -> ProcessingResult:
        """Process single image"""
        start_time = time.time()

        try:
            # Load image
            image = Image.open(input_path).convert("RGB")
            logger.info(f"Processing image: {input_path.name}")

            # Detect watermarks
            detections = self.detector.detect_watermarks(
                image,
                threshold=settings.models.detection_threshold
            )

            if not detections:
                logger.info("No watermarks detected")
                # Copy original if no watermarks found
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_path)
                return ProcessingResult(
                    success=True,
                    input_path=input_path,
                    output_path=output_path,
                    processing_time=time.time() - start_time,
                    detections=[]
                )

            logger.info(f"Found {len(detections)} watermarks")

            # Create mask
            mask = self.detector.create_mask(
                image,
                detections,
                max_area_percent=settings.processing.max_bbox_percent
            )

            if transparent:
                # Make watermark regions transparent
                result = self._make_transparent(image, mask)
            else:
                # Inpaint watermarks
                result = self.inpainter.inpaint(
                    image=image,
                    mask=mask,
                    num_inference_steps=settings.processing.inpaint_steps,
                    guidance_scale=settings.processing.guidance_scale
                )

            # Save result
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(output_path)

            processing_time = time.time() - start_time
            logger.info(
                f"Processed {input_path.name} in {processing_time:.2f}s")

            return ProcessingResult(
                success=True,
                input_path=input_path,
                output_path=output_path,
                processing_time=processing_time,
                detections=detections
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing {input_path}: {e}")

            return ProcessingResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                processing_time=processing_time,
                error_message=str(e)
            )

    def process_video(self, input_path: Path, output_path: Path) -> ProcessingResult:
        """Process video file"""
        start_time = time.time()

        try:
            logger.info(f"Processing video: {input_path.name}")

            # Get video info
            probe = ffmpeg.probe(str(input_path))
            video_info = next(
                s for s in probe['streams'] if s['codec_type'] == 'video')

            width = int(video_info['width'])
            height = int(video_info['height'])
            fps = eval(video_info['r_frame_rate'])
            total_frames = int(float(video_info['nb_frames']))

            logger.info(
                f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")

            # Create temporary directory for frames
            temp_dir = Path("temp_frames") / input_path.stem
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Extract frames
            logger.info("Extracting frames...")
            (
                ffmpeg
                .input(str(input_path))
                .output(str(temp_dir / "frame_%06d.png"))
                .overwrite_output()
                .run(quiet=True)
            )

            # Process frames
            frame_files = sorted(temp_dir.glob("frame_*.png"))
            processed_frames = []

            logger.info(f"Processing {len(frame_files)} frames...")

            # Process frames in batches for better GPU utilization
            for i in tqdm(range(0, len(frame_files), self.batch_size), desc="Processing frames"):
                batch = frame_files[i:i + self.batch_size]

                # Process batch
                batch_results = []
                for frame_file in batch:
                    result = self.process_image(
                        frame_file,
                        frame_file.parent / f"processed_{frame_file.name}"
                    )
                    batch_results.append(result)

                processed_frames.extend(batch_results)

            # Reconstruct video
            logger.info("Reconstructing video...")
            processed_frame_pattern = str(
                temp_dir / "processed_frame_%06d.png")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if settings.processing.preserve_audio:
                # Combine video and audio
                (
                    ffmpeg
                    .input(processed_frame_pattern, framerate=fps)
                    .input(str(input_path))
                    .output(str(output_path), vcodec='libx264', acodec='aac')
                    .overwrite_output()
                    .run(quiet=True)
                )
            else:
                # Video only
                (
                    ffmpeg
                    .input(processed_frame_pattern, framerate=fps)
                    .output(str(output_path), vcodec='libx264')
                    .overwrite_output()
                    .run(quiet=True)
                )

            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)

            processing_time = time.time() - start_time
            logger.info(f"Processed video in {processing_time:.2f}s")

            return ProcessingResult(
                success=True,
                input_path=input_path,
                output_path=output_path,
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing video {input_path}: {e}")

            return ProcessingResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                processing_time=processing_time,
                error_message=str(e)
            )

    def process_batch(self, input_paths: List[Path], output_dir: Path) -> List[ProcessingResult]:
        """Process multiple files in parallel"""
        results = []

        # Separate images and videos
        image_files = [p for p in input_paths if self._is_image(p)]
        video_files = [p for p in input_paths if self._is_video(p)]

        logger.info(
            f"Processing {len(image_files)} images and {len(video_files)} videos")

        # Process images in parallel
        if image_files:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []

                for img_path in image_files:
                    output_path = output_dir / img_path.name
                    future = executor.submit(
                        self.process_image, img_path, output_path)
                    futures.append(future)

                # Collect results
                for future in tqdm(futures, desc="Processing images"):
                    results.append(future.result())

        # Process videos sequentially (they're already parallelized internally)
        for video_path in tqdm(video_files, desc="Processing videos"):
            output_path = output_dir / \
                f"{video_path.stem}_processed{video_path.suffix}"
            result = self.process_video(video_path, output_path)
            results.append(result)

        return results

    def _is_image(self, path: Path) -> bool:
        """Check if file is an image"""
        return path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']

    def _is_video(self, path: Path) -> bool:
        """Check if file is a video"""
        return path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']

    def _make_transparent(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Make masked regions transparent"""
        image = image.convert("RGBA")
        mask = mask.convert("L")

        # Create transparent image
        transparent = Image.new("RGBA", image.size, (0, 0, 0, 0))

        # Copy non-masked regions
        for x in range(image.width):
            for y in range(image.height):
                if mask.getpixel((x, y)) == 0:  # Not masked
                    transparent.putpixel((x, y), image.getpixel((x, y)))

        return transparent
