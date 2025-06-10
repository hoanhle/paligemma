from PIL import Image
from paligemma_processor import PaliGemmaProcessor
from gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig, KVCache
import torch
import json
import os
from safetensors import safe_open
from typing import Tuple
import os
from transformers import AutoTokenizer
import glob
import fire
import platform


def get_optimal_device():
    """Get the optimal device with proper MPS support."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Additional check for MPS availability on macOS
        if platform.system() == "Darwin":
            return "mps"
    return "cpu"


def configure_mps_memory():
    """Configure MPS memory settings for optimal performance."""
    if torch.backends.mps.is_available():
        # Set memory fraction for MPS to prevent memory issues
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"
        # Enable MPS fallback for unsupported operations
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def load_hf_model(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    safetensors_file = glob.glob(os.path.join(model_path, "*.safetensors"))

    tensors = {}
    # For MPS, it's better to load on CPU first then move to device
    load_device = "cpu" if device == "mps" else device
    
    for file in safetensors_file:
        with safe_open(file, framework="pt", device=load_device) as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                # Move to target device if loaded on CPU
                if load_device == "cpu" and device != "cpu":
                    tensor = tensor.to(device)
                tensors[key] = tensor

    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    model = PaliGemmaForConditionalGeneration(config)
    
    # For MPS, ensure all parameters are on the correct device before loading state dict
    if device == "mps":
        model = model.to(device)
    
    model.load_state_dict(tensors, strict=False)
    model.tie_weights()

    return model, tokenizer


def move_inputs_to_device(model_inputs: dict, device: str):
    # Ensure proper dtype for MPS
    moved_inputs = {}
    for k, v in model_inputs.items():
        if device == "mps" and v.dtype == torch.float64:
            # MPS doesn't support float64, convert to float32
            v = v.to(torch.float32)
        moved_inputs[k] = v.to(device)
    return moved_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor,
    prompt: str,
    image_file_path: str,
    device: str
):
   image = Image.open(image_file_path)
   images = [image]
   prompt = [prompt]

   model_inputs = processor(text=prompt, images=images)
   model_inputs = move_inputs_to_device(model_inputs, device)
   return model_inputs


def _sample_top_p(prob: torch.Tensor, top_p: float):
   probs_sort, probs_idx = torch.sort(prob, dim=-1, descending=True)
   probs_sum = torch.cumsum(probs_sort, dim=-1)
   mask = probs_sum - probs_sort > top_p # subtracting probs_sort shift the cummulative sum by 1 position to the right
   probs_sort[mask] = 0
   probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
   next_token = torch.multinomial(probs_sort, num_samples=1)
   next_token = torch.gather(probs_idx, -1, next_token)

   return next_token


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()

    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    # Use autocast for better performance on supported devices
    autocast_device_type = "cuda" if device == "cuda" else "cpu"
    autocast_enabled = device == "cuda"  # Only enable autocast for CUDA
    
    # For MPS, we'll handle mixed precision differently
    if device == "mps":
        # Ensure float32 precision for MPS
        model = model.float()

    for _ in range(max_tokens_to_generate):
        with torch.autocast(device_type=autocast_device_type, enabled=autocast_enabled):
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                kv_cache=kv_cache
            )
        
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]

        # sample
        if do_sample:
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p) 
        else:
           # greedy
           next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0) # remove batch dimension
        generated_tokens.append(next_token)

        if next_token.item() == stop_token:
            break

        # attend the next token to the input
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    decoded_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print(prompt + " " + decoded_text)


def main(
    model_path: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
):
    # Configure MPS if available
    configure_mps_memory()
    
    # Get optimal device
    device = get_optimal_device()
    print(f"Using device: {device}")
    
    # Print additional device info for MPS
    if device == "mps":
        print(f"MPS backend built: {torch.backends.mps.is_built()}")
        print("Note: Some operations may fall back to CPU for MPS")

    print(f"Loading model from {model_path}")
    model, tokenizer = load_hf_model(model_path, device)
    
    # Ensure model is on correct device and in eval mode
    if device != "mps":  # For MPS, we already moved it in load_hf_model
        model = model.to(device)
    model = model.eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens=num_image_tokens, image_size=image_size)

    print(f"Running inference")

    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )


if __name__ == "__main__":
   fire.Fire(main)