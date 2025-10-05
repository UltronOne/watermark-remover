"""
Modern AI models for watermark detection and removal
"""
import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import supervision as sv
from transformers import AutoProcessor, AutoModelForCausalLM
from diffusers import StableDiffusionInpaintPipeline
import torch.nn.functional as F
from loguru import logger


class ModernWatermarkDetector:
    """Modern watermark detection using GroundingDINO + SAM"""

    def __init__(self, device: str = "auto", model_cache_dir: Optional[Path] = None):
        self.device = self._get_device(device)
        self.model_cache_dir = model_cache_dir or Path.home() / ".cache" / \
            "watermark_remover"

        # Initialize models
        self._load_detection_model()
        self._load_segmentation_model()

    def _get_device(self, device: str) -> str:
        """Auto-detect best available device"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _load_detection_model(self):
        """Load GroundingDINO for object detection"""
        try:
            from groundingdino.util.inference import load_model, load_image, predict, annotate

            model_config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            model_checkpoint_path = self.model_cache_dir / "groundingdino_swint_ogc.pth"

            # Download model if not exists
            if not model_checkpoint_path.exists():
                self._download_groundingdino(model_checkpoint_path)

            self.detection_model = load_model(
                model_config_path, str(model_checkpoint_path))
            self.detection_model.to(self.device)
            self.detection_model.eval()

            logger.info(f"GroundingDINO loaded on {self.device}")

        except ImportError:
            logger.warning(
                "GroundingDINO not available, falling back to CLIP-based detection")
            self._load_clip_detector()

    def _load_segmentation_model(self):
        """Load SAM for segmentation"""
        try:
            from segment_anything import sam_model_registry, SamPredictor

            sam_checkpoint = self.model_cache_dir / "sam_vit_h_4b8939.pth"
            model_type = "vit_h"

            # Download SAM if not exists
            if not sam_checkpoint.exists():
                self._download_sam(sam_checkpoint)

            sam = sam_model_registry[model_type](
                checkpoint=str(sam_checkpoint))
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)

            logger.info(f"SAM loaded on {self.device}")

        except ImportError:
            logger.warning(
                "SAM not available, using bounding box detection only")
            self.sam_predictor = None

    def _load_clip_detector(self):
        """Fallback CLIP-based detector"""
        from transformers import CLIPProcessor, CLIPModel

        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14")
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14")
        self.clip_model.to(self.device)
        self.clip_model.eval()

        logger.info(f"CLIP detector loaded on {self.device}")

    def detect_watermarks(self, image: Image.Image, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Detect watermarks in image"""
        if hasattr(self, 'detection_model'):
            return self._detect_with_groundingdino(image, threshold)
        else:
            return self._detect_with_clip(image, threshold)

    def _detect_with_groundingdino(self, image: Image.Image, threshold: float) -> List[Dict[str, Any]]:
        """Detect using GroundingDINO"""
        try:
            from groundingdino.util.inference import load_image, predict

            # Convert PIL to numpy
            image_np = np.array(image)

            # Load image for GroundingDINO
            image_gd, _ = load_image(image_np)

            # Detect watermarks
            boxes, logits, phrases = predict(
                model=self.detection_model,
                image=image_gd,
                caption="watermark logo text",
                box_threshold=threshold,
                text_threshold=0.25
            )

            detections = []
            for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
                if logit > threshold and "watermark" in phrase.lower():
                    x1, y1, x2, y2 = box
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(logit),
                        'label': phrase,
                        'area': (x2 - x1) * (y2 - y1)
                    })

            return detections

        except Exception as e:
            logger.error(f"GroundingDINO detection failed: {e}")
            return self._detect_with_clip(image, threshold)

    def _detect_with_clip(self, image: Image.Image, threshold: float) -> List[Dict[str, Any]]:
        """Fallback CLIP-based detection"""
        # Simple sliding window approach
        detections = []

        # Convert to numpy
        img_np = np.array(image)
        h, w = img_np.shape[:2]

        # Sliding window parameters
        window_size = min(w, h) // 4
        stride = window_size // 2

        for y in range(0, h - window_size, stride):
            for x in range(0, w - window_size, stride):
                # Extract patch
                patch = img_np[y:y+window_size, x:x+window_size]
                patch_pil = Image.fromarray(patch)

                # CLIP classification
                inputs = self.clip_processor(
                    text=["watermark", "logo", "text overlay", "normal image"],
                    images=patch_pil,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    probs = F.softmax(outputs.logits_per_image, dim=-1)

                    # Check if watermark probability is high
                    # First text is "watermark"
                    watermark_prob = probs[0][0].item()

                    if watermark_prob > threshold:
                        detections.append({
                            'bbox': [x, y, x + window_size, y + window_size],
                            'confidence': watermark_prob,
                            'label': 'watermark',
                            'area': window_size * window_size
                        })

        return detections

    def create_mask(self, image: Image.Image, detections: List[Dict[str, Any]],
                    max_area_percent: float = 10.0) -> Image.Image:
        """Create mask from detections"""
        mask = Image.new("L", image.size, 0)

        if not detections:
            return mask

        image_area = image.width * image.height

        for detection in detections:
            bbox = detection['bbox']
            area = detection['area']

            # Filter by area
            if (area / image_area) * 100 <= max_area_percent:
                # Create mask for this detection
                from PIL import ImageDraw
                draw = ImageDraw.Draw(mask)
                draw.rectangle(bbox, fill=255)

        return mask

    def _download_groundingdino(self, path: Path):
        """Download GroundingDINO model"""
        import requests

        url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading GroundingDINO model...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"GroundingDINO model saved to {path}")

    def _download_sam(self, path: Path):
        """Download SAM model"""
        import requests

        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading SAM model...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"SAM model saved to {path}")


class ModernInpainter:
    """Modern inpainting using Stable Diffusion"""

    def __init__(self, device: str = "auto", model_cache_dir: Optional[Path] = None):
        self.device = self._get_device(device)
        self.model_cache_dir = model_cache_dir or Path.home() / ".cache" / \
            "watermark_remover"

        self._load_inpaint_model()

    def _get_device(self, device: str) -> str:
        """Auto-detect best available device"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _load_inpaint_model(self):
        """Load Stable Diffusion Inpainting model"""
        model_id = "runwayml/stable-diffusion-inpainting"

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            cache_dir=str(self.model_cache_dir)
        )

        self.pipe = self.pipe.to(self.device)

        # Enable memory efficient attention
        if hasattr(self.pipe, 'enable_attention_slicing'):
            self.pipe.enable_attention_slicing()

        if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except:
                pass

        logger.info(f"Stable Diffusion Inpainting loaded on {self.device}")

    def inpaint(self, image: Image.Image, mask: Image.Image,
                prompt: str = "", negative_prompt: str = "watermark, logo, text, blurry, low quality",
                num_inference_steps: int = 50, guidance_scale: float = 7.5) -> Image.Image:
        """Inpaint masked regions"""

        # Convert mask to binary
        mask_np = np.array(mask)
        mask_np = (mask_np > 128).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_np)

        # Generate inpainted image
        result = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_pil,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=1.0
        ).images[0]

        return result
