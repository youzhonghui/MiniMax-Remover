# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiniMax-Remover is a fast and effective video object removal system based on minimax optimization. It uses a two-stage approach with a simplified DiT (Diffusion Transformer) architecture for video inpainting and object removal.

## Key Architecture Components

### Core Pipeline (`pipeline_minimax_remover.py`)
- `Minimax_Remover_Pipeline`: Main diffusion pipeline inheriting from DiffusionPipeline
- Integrates VAE (AutoencoderKLWan), Transformer3DModel, and FlowMatchEulerDiscreteScheduler
- Key parameters: 6 inference steps, no CFG (Classifier-Free Guidance)
- VAE scale factors: temporal=4, spatial=8

### Transformer Model (`transformer_minimax_remover.py`)
- `Transformer3DModel`: Custom 3D transformer for video processing
- Uses AttnProcessor2_0 with PyTorch 2.0's scaled_dot_product_attention
- Includes rotary positional embeddings and FP32LayerNorm
- Self-attention based architecture with custom attention processors

### Video Processing
- Uses decord for video loading (VideoReader)
- Input videos: 81 frames, 480x832 resolution
- Image preprocessing: normalize to [-1, 1] range
- Mask preprocessing: binary masks [0, 1] with dilation support

## Development Commands

### Installation
```bash
pip install -r requirements.txt
```

### Model Download
```bash
huggingface-cli download zibojia/minimax-remover --include vae transformer scheduler --local-dir .
```

### Running Tests
```bash
python test_minimax_remover.py
```

### Gradio Demo
```bash
cd gradio_demo
python3 test.py
```

## Key Dependencies
- PyTorch 2.7.1 (required for AttnProcessor2_0)
- diffusers 0.33.1
- decord 0.6 (video processing)
- gradio 3.40.0 (demo interface)

## Model Structure
- Input: Video frames + binary masks
- Processing: VAE encoding → Transformer inference → VAE decoding
- Output: Inpainted video with objects removed
- Mask dilation: Controlled by `iterations` parameter (default: 6)

## Important Implementation Details
- Uses UniPCMultistepScheduler with 12 inference steps
- Supports CUDA acceleration (device="cuda:0")
- Video length fixed at 81 frames
- Manual seed control for reproducible results
- Temporal and spatial downsampling via VAE

## File Organization
- Root: Main pipeline and transformer implementations
- `gradio_demo/`: Interactive demo with SAM2 integration
- `gradio_demo/sam2/`: SAM2 model for automatic mask generation
- Test videos in `gradio_demo/normal_videos/` and `gradio_demo/cartoon/`