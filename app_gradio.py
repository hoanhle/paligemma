import gradio as gr
from PIL import Image
from paligemma_processor import PaliGemmaProcessor
from gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig, KVCache
import torch
import json
import os
from safetensors import safe_open
from typing import Tuple
from transformers import AutoTokenizer
import glob

# Global variables to store the model
model = None
processor = None
device = None

def load_hf_model(model_path: str, device: str):
    """Load the HuggingFace model from the given path."""
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
    """Move model inputs to the specified device."""
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs

def get_model_inputs(processor, prompt: str, image: Image.Image, device: str):
    """Prepare model inputs from prompt and image."""
    images = [image]
    prompts = [prompt]
    
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs

def _sample_top_p(prob: torch.Tensor, top_p: float):
    """Sample from the probability distribution using top-p sampling."""
    probs_sort, probs_idx = torch.sort(prob, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def generate_response(
    image,
    prompt,
    max_tokens,
    temperature,
    top_p,
    do_sample
):
    """Generate response using the PaliGemma model."""
    global model, processor, device
    
    if model is None:
        return "Please load a model first!"
    
    if image is None:
        return "Please upload an image!"
    
    model_inputs = get_model_inputs(processor, prompt, image, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                kv_cache=kv_cache
            )
            kv_cache = outputs["kv_cache"]
            next_token_logits = outputs["logits"][:, -1, :]

            # Sample or greedy decode
            if do_sample:
                next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = _sample_top_p(next_token_logits, top_p)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            assert next_token.size() == (1, 1)
            next_token = next_token.squeeze(0)
            generated_tokens.append(next_token)

            if next_token.item() == stop_token:
                break

            # Prepare for next iteration
            input_ids = next_token.unsqueeze(-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
            )

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    decoded_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return prompt + " " + decoded_text

def load_model(model_path):
    """Load the model and update the global variables."""
    global model, processor, device
    
    try:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Load model
        model, tokenizer = load_hf_model(model_path, device)
        model = model.to(device).eval()
        
        # Initialize processor
        num_image_tokens = model.config.vision_config.num_image_tokens
        image_size = model.config.vision_config.image_size
        processor = PaliGemmaProcessor(tokenizer, num_image_tokens=num_image_tokens, image_size=image_size)
        
        return f"Model loaded successfully! Using device: {device}"
    except Exception as e:
        return f"Error loading model: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="PaliGemma Vision-Language Model") as demo:
    gr.Markdown("# PaliGemma Vision-Language Model Interface")
    gr.Markdown("Upload an image and provide a prompt to generate text based on the image content.")
    
    with gr.Row():
        with gr.Column():
            model_path_input = gr.Textbox(
                label="Model Path",
                placeholder="Enter the path to your HuggingFace model",
                value=""
            )
            load_button = gr.Button("Load Model")
            model_status = gr.Textbox(label="Model Status", interactive=False)
            
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Image",
                type="pil"
            )
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                value="Describe this image:"
            )
            
        with gr.Column():
            output_text = gr.Textbox(
                label="Generated Text",
                lines=5,
                interactive=False
            )
    
    with gr.Accordion("Advanced Settings", open=False):
        max_tokens_slider = gr.Slider(
            minimum=1,
            maximum=500,
            value=100,
            step=1,
            label="Max Tokens to Generate"
        )
        temperature_slider = gr.Slider(
            minimum=0.1,
            maximum=2.0,
            value=0.8,
            step=0.1,
            label="Temperature"
        )
        top_p_slider = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.9,
            step=0.1,
            label="Top-p"
        )
        do_sample_checkbox = gr.Checkbox(
            label="Do Sampling (unchecked = greedy decoding)",
            value=False
        )
    
    generate_button = gr.Button("Generate", variant="primary")
    
    # Set up event handlers
    load_button.click(
        fn=load_model,
        inputs=[model_path_input],
        outputs=[model_status]
    )
    
    generate_button.click(
        fn=generate_response,
        inputs=[
            image_input,
            prompt_input,
            max_tokens_slider,
            temperature_slider,
            top_p_slider,
            do_sample_checkbox
        ],
        outputs=[output_text]
    )
    
    # Example inputs
    gr.Examples(
        examples=[
            ["Describe this image:", 100, 0.8, 0.9, False],
            ["What objects can you see in this image?", 100, 0.8, 0.9, False],
            ["Generate a caption for this image:", 50, 0.7, 0.9, True],
        ],
        inputs=[prompt_input, max_tokens_slider, temperature_slider, top_p_slider, do_sample_checkbox],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)