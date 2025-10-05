# Modern Watermark Remover

A high-performance, GPU-optimized watermark removal tool using state-of-the-art AI models.

## Features

- **Modern AI Models**: GroundingDINO + SAM for detection, Stable Diffusion for inpainting
- **GPU Optimization**: CUDA acceleration with memory-efficient processing
- **Batch Processing**: Parallel processing for maximum throughput
- **Video Support**: Frame-by-frame processing with audio preservation
- **Modern CLI**: Rich terminal interface with progress tracking
- **Docker Support**: Easy deployment on GPU servers

## Quick Start

### Using Docker (Recommended)

```bash
# Build and run
docker-compose up --build

# Process files
docker-compose run watermark-remover python main.py process input/ output/

# With API server
docker-compose --profile api up --build
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download models
python main.py download-models

# Process files
python main.py process input/ output/ --batch-size 8 --workers 8
```

## Usage

### Basic Commands

```bash
# Process single file
python main.py process image.jpg output.jpg

# Process directory
python main.py process input/ output/ --batch-size 8

# Make watermarks transparent
python main.py process input/ output/ --transparent

# Check system info
python main.py info
```

### Advanced Options

```bash
# GPU optimization
python main.py process input/ output/ \
    --batch-size 16 \
    --workers 8 \
    --device cuda \
    --max-bbox 15.0

# Video processing
python main.py process video.mp4 output.mp4

# Logging
python main.py process input/ output/ \
    --log-level DEBUG \
    --log-file logs/processing.log
```

## Performance Optimization

### GPU Settings

- **Batch Size**: Increase for better GPU utilization (8-16 recommended)
- **Workers**: Match CPU cores (4-8 recommended)
- **Device**: Use `cuda` for NVIDIA GPUs, `mps` for Apple Silicon

### Memory Management

- Automatic memory optimization enabled
- FP16 precision for reduced memory usage
- Attention slicing for large images

## System Requirements

### Minimum

- Python 3.11+
- 8GB RAM
- NVIDIA GPU with 6GB VRAM (recommended)

### Recommended

- Python 3.11+
- 16GB+ RAM
- NVIDIA RTX 3080/4080 or better
- 12GB+ VRAM

## Supported Formats

### Images

- JPEG, PNG, WEBP, BMP, TIFF

### Videos

- MP4, AVI, MOV, MKV, FLV, WMV, WEBM

## Configuration

Create a `.env` file for persistent settings:

```env
WATERMARK_PROCESSING_BATCH_SIZE=8
WATERMARK_PROCESSING_NUM_WORKERS=8
WATERMARK_PROCESSING_DEVICE=cuda
WATERMARK_PROCESSING_MAX_BBOX_PERCENT=10.0
WATERMARK_MODELS_DETECTION_THRESHOLD=0.3
```

## API Usage

Start the API server:

```bash
docker-compose --profile api up
```

Then use the REST API:

```bash
# Process image
curl -X POST "http://localhost:8000/process" \
     -H "Content-Type: application/json" \
     -d '{"input_path": "input.jpg", "output_path": "output.jpg"}'

# Batch process
curl -X POST "http://localhost:8000/batch" \
     -H "Content-Type: application/json" \
     -d '{"input_dir": "input/", "output_dir": "output/"}'
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   - Reduce batch size: `--batch-size 4`
   - Use FP16: `--precision fp16`

2. **Slow Processing**

   - Increase batch size: `--batch-size 16`
   - Use more workers: `--workers 8`
   - Ensure GPU is being used: `--device cuda`

3. **Model Download Issues**
   - Check internet connection
   - Manually download models: `python main.py download-models`

### Performance Tips

- Use SSD storage for input/output
- Ensure adequate cooling for GPU
- Monitor GPU memory usage
- Use appropriate batch sizes for your hardware

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

- GitHub Issues: [Report bugs and request features](https://github.com/your-repo/issues)
- Documentation: [Full documentation](https://github.com/your-repo/wiki)
