"""
Modern configuration management using Pydantic Settings
"""
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ProcessingConfig(BaseModel):
    """Configuration for watermark processing"""
    max_bbox_percent: float = Field(default=10.0, ge=1.0, le=100.0)
    batch_size: int = Field(default=4, ge=1, le=32)
    num_workers: int = Field(default=4, ge=1, le=16)
    device: str = Field(default="auto")
    precision: str = Field(default="fp16")  # fp16, fp32, bf16
    use_cache: bool = Field(default=True)

    # Video processing
    video_batch_size: int = Field(default=8, ge=1, le=64)
    video_fps_limit: Optional[int] = Field(default=None, ge=1, le=120)
    preserve_audio: bool = Field(default=True)

    # Quality settings
    inpaint_steps: int = Field(default=50, ge=10, le=200)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)


class ModelConfig(BaseModel):
    """Configuration for AI models"""
    detection_model: str = Field(default="groundingdino")
    inpaint_model: str = Field(default="stable-diffusion-inpainting")
    model_cache_dir: Path = Field(
        default=Path.home() / ".cache" / "watermark_remover")

    # Model-specific settings
    detection_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    nms_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class Settings(BaseSettings):
    """Main application settings"""
    processing: ProcessingConfig = ProcessingConfig()
    models: ModelConfig = ModelConfig()

    # Paths
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None

    # Processing options
    overwrite: bool = False
    transparent: bool = False
    force_format: Optional[str] = None

    # Logging
    log_level: str = Field(default="INFO")
    log_file: Optional[Path] = None

    class Config:
        env_prefix = "WATERMARK_"
        case_sensitive = False


# Global settings instance
settings = Settings()
