#!/bin/bash

# Launch the Gradio web interface for PaliGemma

echo "Starting PaliGemma Gradio Interface..."
echo "The interface will be available at http://localhost:7860"
echo ""
echo "To use the interface:"
echo "1. Enter your model path in the 'Model Path' field"
echo "2. Click 'Load Model' and wait for confirmation"
echo "3. Upload an image"
echo "4. Enter a prompt (e.g., 'Describe this image:')"
echo "5. Click 'Generate' to get the model's response"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app_gradio.py