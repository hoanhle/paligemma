from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

class SiglipVisionConfig:
    def __init__(
            self,
            hidden_size=768, # embedding size
            intermediate_size=3072, # linear layer size in FeedForward
            num_hidden_layers=12, # number of layers in VisionTransformer
            num_attention_heads=12,
            num_channels=3,
            image_size=224,
            patch_size=16,
            layer_norm_eps=1e-12,
            attention_dropout=0.0,
            num_image_tokens: int = None,
            **kwargs,
            ):
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # no padding is needed
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embeddings = nn.Embedding(
            self.num_positions,
            self.embed_dim,
        )

        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # b, c, h, w

        # b, c, h, w -> b, embed_dim, num_patches_h, num_patches_w
        patch_embeds = self.patch_embeddings(pixel_values)

        # b, embed_dim, num_patches_h, num_patches_w -> b, embed_dim, num_patches (= num_patches_h * num_patches_w)
        embeddings = patch_embeds.flatten(2)

        # b, embed_dim, num_patches -> b, num_patches, embed_dim
        embeddings = embeddings.transpose(1, 2)

        # add position embeddings to each patch embedding
        embeddings = embeddings + self.position_embeddings(self.position_ids)

        return embeddings
    
class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # b, n, c -> b, n, num_heads, head_dim
        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # b, n, embed_dim -> b, n, num_heads, head_dim -> b, num_heads, n, head_dim
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # calculate attention weights using the formula: QK^T / sqrt(d_k). attention weights: b, num_heads, n, n
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attn_weights.shape != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(f"Attention weights should be of shape (batch_size, num_heads, seq_len, seq_len), but is {attn_weights.shape}")

        # applying softmax to the attention weights row-wise b, num_heads, n, n
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training) # dropout is not really used in the codebase

        # attention output: b, num_head, n, head_dim
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2)

        if attn_output.shape != (batch_size, seq_len, self.num_heads, self.head_dim):
            raise ValueError(f"Attention output should be of shape (batch_size, seq_len, num_heads, head_dim), but is {attn_output.shape}")

        # b, num_heads, n, head_dim -> b, n, num_heads, head_dim
        attn_output = attn_output.transpose(1, 2).contiguous()

        # b, num_heads, n, head_dim -> b, n, embed_dim
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output
    

class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.encoder_layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = input_embeds
        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps) 
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

        
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoderLayer(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
    

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # b, c, h, w -> b, num_patches, embed_dim
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(hidden_states)

        return self.post_layernorm(last_hidden_state)



class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
    
    def forward(self, pixel_values) -> Tuple:
        # b, c, h, w -> b, num_patches, embed_dim

        return self.vision_model(pixel_values)


