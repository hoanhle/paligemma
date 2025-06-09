import numpy as np
from PIL import Image
from typing import List, Tuple
import torch


# Hugging Face's default image mean and std for ImageNet (which is kind of wrong)
IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_length, image_token):
    return f"{image_token * image_seq_length} {bos_token}{prefix_prompt}\n"


def process_image(
    images: List[Image.Image],
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: List[float] = None,
    image_std: List[float] = None,
) -> List[np.ndarray]:
    height, width = size

    images = [image.resize((width, height), resample=resample) for image in images]
    images = [np.array(image) for image in images]

    # rescale the image to 0-1
    images = [image.astype(np.float32) * rescale_factor for image in images]

    # normalize the image to have mean 0 and std 1
    images = [(image - np.array(image_mean)) / np.array(image_std)for image in images]

    # move the channel to first dimension h, w, c -> c, h, w
    images = [image.transpose(2, 0, 1) for image in images]
    
    return images


class PaliGemmaProcessor:

    IMAGE_TOKEN = "<image>" # placeholder for the image token, later replaced by image embeddings produced by Vision Encoder

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # Tokenizer described here: 
        # https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/paligemma
        # https://huggingface.co/blog/paligemma

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ] # these tokens are used for object detection
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ] # these tokens are used for segmentation

        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        # we will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
            self,
            text: List[str],
            images: List[Image.Image],
            padding: str = "longest",
            truncation: bool = True,
    ):
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images and {len(text)} texts, but expected 1 image and 1 text"

        pixel_values = process_image(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1.0/255.0,
            image_mean=IMAGENET_MEAN,
            image_std=IMAGENET_STD,
        )

        # convert a list of numpy arrays to a single numpy array with shape (batch_size, channel, height, width)
        pixel_values = np.stack(pixel_values, axis=0)

        # convert the pixel values to a tensor
        pixel_values = torch.from_numpy(pixel_values)

        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_length=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # return input_ids, attention_mask
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data