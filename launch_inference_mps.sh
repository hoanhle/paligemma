#!/bin/bash

# MPS-optimized launch script for PaliGemma inference

# Model configuration
MODEL_PATH="$HOME/workspace/model_weights/paligemma-3b-pt-224"
PROMPT="what's in this image?"
IMAGE_FILE_PATH="./cat-dog.jpeg"
MAX_TOKENS_TO_GENERATE=100
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"

# MPS-specific environment variables
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Optional: Enable MPS profiling for debugging (set to 1 to enable)
export PYTORCH_MPS_PROFILING=0

# Check if running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Running on macOS - MPS backend may be available"
    
    # Check PyTorch MPS availability
    python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
    python -c "import torch; print(f'MPS built: {torch.backends.mps.is_built()}')"
else
    echo "Not running on macOS - MPS backend will not be available"
fi

echo "Starting inference with MPS support..."

# Run inference
python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE