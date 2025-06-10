## Implementation of Paligemma

Implementation of Vision Language Model (paligemma). 

![paligemma](./canvas.jpeg)


## Getting Started

### Environment setup

Use **conda** or **uv** to install dependencies:

```bash
# with conda
conda env create -f environment.yml
conda activate paligemma

# with uv
pip install uv
uv venv
uv pip install -r requirements.txt
```

### Download model weights

The weights are hosted on [Hugging Face](https://huggingface.co/google/paligemma-3b-pt-224). Download them with:

```bash
git lfs install
git clone https://huggingface.co/google/paligemma-3b-pt-224 ~/workspace/model_weights/paligemma-3b-pt-224
```

Set `MODEL_PATH` in `launch_inference.sh` to the downloaded directory.



## Citations

```bibtex
@misc{beyer2024paligemmaversatile3bvlm,
      title={PaliGemma: A versatile 3B VLM for transfer}, 
      author={Lucas Beyer and Andreas Steiner and André Susano Pinto and Alexander Kolesnikov and Xiao Wang and Daniel Salz and Maxim Neumann and Ibrahim Alabdulmohsin and Michael Tschannen and Emanuele Bugliarello and Thomas Unterthiner and Daniel Keysers and Skanda Koppula and Fangyu Liu and Adam Grycner and Alexey Gritsenko and Neil Houlsby and Manoj Kumar and Keran Rong and Julian Eisenschlos and Rishabh Kabra and Matthias Bauer and Matko Bošnjak and Xi Chen and Matthias Minderer and Paul Voigtlaender and Ioana Bica and Ivana Balazevic and Joan Puigcerver and Pinelopi Papalampidi and Olivier Henaff and Xi Xiong and Radu Soricut and Jeremiah Harmsen and Xiaohua Zhai},
      year={2024},
      eprint={2407.07726},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.07726}, 
}
```
```bibtex
@misc{zhai2023sigmoidlosslanguageimage,
      title={Sigmoid Loss for Language Image Pre-Training}, 
      author={Xiaohua Zhai and Basil Mustafa and Alexander Kolesnikov and Lucas Beyer},
      year={2023},
      eprint={2303.15343},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2303.15343}, 
}
```
```bibtex
@misc{vaswani2023attentionneed,
      title={Attention Is All You Need}, 
      author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
      year={2023},
      eprint={1706.03762},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1706.03762}, 
}
```
```bibtex
@misc{dosovitskiy2021imageworth16x16words,
      title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale}, 
      author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
      year={2021},
      eprint={2010.11929},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2010.11929}, 
}
```

```bibtex
@misc{zhang2019rootmeansquarelayer,
      title={Root Mean Square Layer Normalization}, 
      author={Biao Zhang and Rico Sennrich},
      year={2019},
      eprint={1910.07467},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1910.07467}, 
}
```

```bibtex
@misc{ba2016layernormalization,
      title={Layer Normalization}, 
      author={Jimmy Lei Ba and Jamie Ryan Kiros and Geoffrey E. Hinton},
      year={2016},
      eprint={1607.06450},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/1607.06450}, 
}
```

```bibtex
@misc{shazeer2019fasttransformerdecodingwritehead,
      title={Fast Transformer Decoding: One Write-Head is All You Need}, 
      author={Noam Shazeer},
      year={2019},
      eprint={1911.02150},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/1911.02150}, 
}
```


```bibtex
@misc{ainslie2023gqatraininggeneralizedmultiquery,
      title={GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints}, 
      author={Joshua Ainslie and James Lee-Thorp and Michiel de Jong and Yury Zemlyanskiy and Federico Lebrón and Sumit Sanghai},
      year={2023},
      eprint={2305.13245},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2305.13245}, 
}
```

```bibtex
@misc{su2023roformerenhancedtransformerrotary,
      title={RoFormer: Enhanced Transformer with Rotary Position Embedding}, 
      author={Jianlin Su and Yu Lu and Shengfeng Pan and Ahmed Murtadha and Bo Wen and Yunfeng Liu},
      year={2023},
      eprint={2104.09864},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2104.09864}, 
}
```

## Web Interface with Gradio

We've added a web interface using [Gradio](https://github.com/gradio-app/gradio) for easier interaction with the PaliGemma model.

### Installation

The Gradio dependency has been added to `requirements.txt`. Install it along with other dependencies:

```bash
pip install -r requirements.txt
```

### Running the Web Interface

There are two ways to run the Gradio interface:

#### 1. Full Interface (with model loading UI)

This interface allows you to load models dynamically through the web UI:

```bash
./launch_gradio.sh
# or
python app_gradio.py
```

The interface will be available at http://localhost:7860

#### 2. Simple Interface (with pre-loaded model)

This interface can load a model on startup:

```bash
# Without pre-loading a model
python app_gradio_simple.py

# With a pre-loaded model
python app_gradio_simple.py --model_path ~/workspace/model_weights/paligemma-3b-pt-224

# With sharing enabled (creates a public URL)
python app_gradio_simple.py --model_path ~/workspace/model_weights/paligemma-3b-pt-224 --share

# On a different port
python app_gradio_simple.py --port 8080
```

### Using the Interface

1. **Upload an image**: Click or drag-and-drop an image into the image upload area
2. **Enter a prompt**: Type your question or instruction about the image (e.g., "Describe this image:", "What objects are in this photo?")
3. **Adjust settings** (optional): Expand the "Generation Settings" panel to modify:
   - Max tokens: Number of tokens to generate
   - Temperature: Controls randomness (higher = more random)
   - Top-p: Nucleus sampling parameter
   - Use Sampling: Toggle between sampling and greedy decoding
4. **Generate**: Click the "Generate" button to get the model's response

### Features

- Clean, modern UI with emoji icons
- Real-time image preview
- Adjustable generation parameters
- Example prompts for quick testing
- Support for various image formats (JPEG, PNG, etc.)
- Responsive design that works on different screen sizes
