from PIL import Image
from paligemma_processor import PaliGemmaProcessor
from gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig, KVCache
import torch
from torchao.quantization.quant_api import quantize_, Int4WeightOnlyConfig
import json
import os
from safetensors import safe_open
from typing import Tuple
import os
from transformers import AutoTokenizer
import glob
import fire


def load_hf_model(model_path: str, device: str):
   tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
   safetensors_file = glob.glob(os.path.join(model_path, "*.safetensors"))

   tensors = {}
   for file in safetensors_file:
      with safe_open(file, framework="pt", device=device) as f:
         for key in f.keys():
            tensors[key] = f.get_tensor(key)

   with open(os.path.join(model_path, "config.json"), "r") as f:
      model_config_file = json.load(f)
      config = PaliGemmaConfig(**model_config_file)

   model = PaliGemmaForConditionalGeneration(config)
   model.load_state_dict(tensors, strict=False)
   model.tie_weights()

   return model, tokenizer


def move_inputs_to_device(model_inputs: dict, device: str):
   model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
   return model_inputs


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

    for _ in range(max_tokens_to_generate):
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
            next_token_logits = _sample_top_p(next_token_logits, top_p) 
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
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model from {model_path} onto CPU for quantization...")
    # Load model onto CPU first
    model, tokenizer = load_hf_model(model_path, "cpu")
    model = model.eval() # Set to eval mode before quantization

    print("Saving original model state_dict to disk...")
    original_model_path = "original_model_state_dict.pth"
    torch.save(model.state_dict(), original_model_path)
    print(f"Original model state_dict saved to: {original_model_path}")
    original_size = os.path.getsize(original_model_path)
    print(f"Original model state_dict size: {original_size / (1024*1024):.2f} MB")

    print("Original model structure:")
    print(model)

    # Define quantization configuration
    # Using group_size=128 and hqq=True as per torchao examples for potentially better quality.
    # If hqq or specific group_size causes issues (e.g. not available in CPU-only torchao build),
    # a simpler config like Int4WeightOnlyConfig() can be tried.
    quantization_config = Int4WeightOnlyConfig(group_size=128, use_hqq=True)

    print(f"Applying INT4 weight-only quantization with config: {quantization_config}...")
    # Apply quantization
    # Ensure the model is on CPU before calling quantize_ if the backend expects it.
    # Some backends might require specific model states or data types.
    quantized_model = quantize_(model, quantization_config)
    print("Quantization complete.")

    print("Quantized model structure:")
    print(quantized_model)

    print("Saving quantized model state_dict to disk...")
    quantized_model_path = "quantized_model_state_dict.pth"
    # Ensure quantized_model is on CPU before saving for fair comparison if it's not already
    torch.save(quantized_model.state_dict(), quantized_model_path)
    print(f"Quantized model state_dict saved to: {quantized_model_path}")
    quantized_size = os.path.getsize(quantized_model_path)
    print(f"Quantized model state_dict size: {quantized_size / (1024*1024):.2f} MB")

    if original_size > 0:
        reduction_percentage = (original_size - quantized_size) / original_size * 100
        print(f"Model size reduction: {reduction_percentage:.2f}%")

    # Move the quantized model to the target device
    model = quantized_model.to(device)
    print(f"Moved quantized model to device: {device}")

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