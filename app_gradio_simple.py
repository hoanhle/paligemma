import gradio as gr
from PIL import Image
from paligemma_processor import PaliGemmaProcessor
from gemma import PaliGemmaForConditionalGeneration
from inference import load_hf_model, get_model_inputs, _sample_top_p
import torch
import argparse

# Global variables
model = None
processor = None
device = None

def generate_response(image, prompt, max_tokens=100, temperature=0.8, top_p=0.9, do_sample=False):
    """Generate response using the PaliGemma model."""
    global model, processor, device
    
    if model is None:
        return "Error: Model not loaded!"
    
    if image is None:
        return "Please upload an image!"
    
    # Convert prompt and image to model inputs
    model_inputs = get_model_inputs(processor, prompt, image.name if hasattr(image, 'name') else "temp.jpg", device)
    
    # If image is already a PIL Image, use it directly
    if isinstance(image, Image.Image):
        images = [image]
        prompts = [prompt]
        model_inputs = processor(text=prompts, images=images)
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"] 
    pixel_values = model_inputs["pixel_values"]
    
    # Import KVCache here to avoid circular imports
    from gemma import KVCache
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

def create_demo():
    """Create the Gradio demo interface."""
    with gr.Blocks(title="PaliGemma Vision-Language Model") as demo:
        gr.Markdown("# 🖼️ PaliGemma Vision-Language Model")
        gr.Markdown("Upload an image and ask questions about it!")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Image",
                    type="pil"
                )
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="What do you want to know about this image?",
                    value="Describe this image:"
                )
                
                with gr.Accordion("⚙️ Generation Settings", open=False):
                    max_tokens = gr.Slider(
                        minimum=10, maximum=200, value=100, step=10,
                        label="Max Tokens"
                    )
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.8, step=0.1,
                        label="Temperature"
                    )
                    top_p = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                        label="Top-p"
                    )
                    do_sample = gr.Checkbox(
                        label="Use Sampling", value=False
                    )
                
                generate_btn = gr.Button("🚀 Generate", variant="primary")
                
            with gr.Column(scale=1):
                output = gr.Textbox(
                    label="Model Response",
                    lines=8
                )
        
        # Examples
        gr.Examples(
            examples=[
                ["cat-dog.jpeg", "What animals are in this image?"],
                ["cat-dog.jpeg", "Describe the scene in detail:"],
                ["cat-dog.jpeg", "What is happening in this picture?"],
            ],
            inputs=[image_input, prompt_input],
        )
        
        # Event handler
        generate_btn.click(
            fn=generate_response,
            inputs=[image_input, prompt_input, max_tokens, temperature, top_p, do_sample],
            outputs=output
        )
    
    return demo

def main():
    parser = argparse.ArgumentParser(description="PaliGemma Gradio Interface")
    parser.add_argument("--model_path", type=str, help="Path to the model", default=None)
    parser.add_argument("--share", action="store_true", help="Create a public sharing link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    args = parser.parse_args()
    
    global model, processor, device
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model if path provided
    if args.model_path:
        print(f"Loading model from {args.model_path}...")
        try:
            model, tokenizer = load_hf_model(args.model_path, device)
            model = model.to(device).eval()
            
            # Initialize processor
            num_image_tokens = model.config.vision_config.num_image_tokens
            image_size = model.config.vision_config.image_size
            processor = PaliGemmaProcessor(tokenizer, num_image_tokens=num_image_tokens, image_size=image_size)
            
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("You'll need to load a model through the interface.")
    
    # Create and launch demo
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )

if __name__ == "__main__":
    main()