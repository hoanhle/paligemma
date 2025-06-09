import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from siglip import SiglipVisionConfig, SiglipVisionModel


class KVCache():
    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self):
        if len(self.key_cache) == 0:
            return 0
        # key_cache shape: batch_size, num_heads, seq_len, head_dim
        return self.key_cache[0].shape[-2]
    
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    

class GemmaRotaryEmbedding(nn.Module):
    """
    Read: https://arxiv.org/abs/2104.09864 for more details on the rotary embedding
    """
    def __init__(self, head_dim: int, max_position_embeddings: int, base: float = 10000.0, device: torch.device = None):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.device = device

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=self.device) / self.head_dim))
        self.register_buffer("inv_freq", tensors=inv_freq, persistent=False)
    
    def forward(self, x, position_ids, seq_len=None):
        # x: batch_size, num_heads, seq_len, head_dim
        self.inv_freq = self.inv_freq.to(x.device)
        # copy inv_freq for batch in the sequence
        # inv_freq_expanded: batch_size, head_dim // 2, 1
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

            # Hugging Face's implementation https://github.com/huggingface/transformers/issues/25199
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos().to(x.dtype)
            sin = emb.sin().to(x.dtype)
            
        return cos, sin


class GemmaConfig():
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size, # this is the size of the intermediate layer in the feed forward network
        num_hidden_layers,
        num_attention_heads, # number of heads for query
        num_key_value_heads, # number of heads for key and value
        head_dim = 256,
        max_position_embeddings = 8192,
        rms_norm_eps = 1e-6,
        rope_theta = 10000.0,
        attention_bias = False,
        attention_dropout = 0.0,
        pad_token_id = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class PaliGemmaConfig():
    def __init__(
        self,
        vision_config: SiglipVisionConfig = None,
        text_config: GemmaConfig = None,
        ignore_index: int = -100,  
        image_token_index: int = 256000, # this is the index of the image token in the vocabulary
        vocab_size: int = 257152,
        projection_dim: int = 2048, 
        hidden_size: int = 2048,
        pad_token_id: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.is_encoder_decoder = False # we will not be using this but Hugging Face requires it
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config)

        self.vision_config.num_image_tokens = (self.vision_config.image_size //  self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim


class PaliGemmaMultiModelProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=True)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        # batch_size, num_patches, embed_dim -> batch_size, num_patches, projection_dim
        return self.linear(image_embeds)


class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = GemmaModel(config)

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        
        # batch_size, seq_len, hidden_size -> batch_size, seq_len, vocab_size
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        # TODO: shouldn't this be the last hidden state?
        hidden_states = outputs
        logits = self.lm_head(hidden_states).float()

        return_data = {
            "logits": logits,
        }
        
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data
    

class GemmaRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(torch.pow(x, 2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x):
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    
    hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch_size, num_key_value_heads * n_rep, seq_len, head_dim)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=-1):
    """
    Using Hugging Face's implementation
    """
    cos = cos.unsqueeze(unsqueeze_dim) # add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim) 
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx # since its layer has its own kv cache, we need to know which layer it is
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_casual = True
    
        assert self.hidden_size % self.num_heads == 0, f"hidden_size must be divisible by num_heads (got {self.hidden_size} and {self.num_heads})"

        # Read: https://arxiv.org/abs/2305.13245 for more details on the grouped query attention
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta
        )
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states, key_states, value_states = self.rotary_emb(query_states, key_states, value_states, position_ids)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2) # batch_size, num_heads, seq_len, head_dim
        key_states = key_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # batch_size, num_key_value_heads, seq_len, head_dim
        value_states = value_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # batch_size, num_key_value_heads, seq_len, head_dim

        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        
        # repeat the key and value states for each head since we are working with native attention implementation
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # perform the attention
        attention_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # TODO: recheck attention_mask or causal mask
        assert attention_mask is not None, "attention mask is required for causal attention"
        attention_weights = attention_weights + attention_mask

        attention_weights = nn.functional.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attention_weights = nn.functional.dropout(attention_weights, p=self.attention_dropout, training=self.training)
        attention_output = torch.matmul(attention_weights, value_states)

        if attention_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"Attention output size is not correct: {attention_output.size()} != ({batch_size}, {self.num_heads}, {q_len}, {self.head_dim})")
        
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, q_len, -1)

        attention_output = self.o_proj(attention_output)

        return attention_output, attention_weights
        


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = GemmaAttention(config, layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # b, seq_len, hidden_size
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
        
        
class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        hidden_states = inputs_embeds
        normalizer = torch.tensor(self.config.hidden_size**0.5, device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask,
                position_ids,
                kv_cache,
            )
        
        hidden_states = self.norm(hidden_states)

        return hidden_states

class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_model_projector = PaliGemmaMultiModelProjector(config)
        self.language_model = GemmaForCausalLM(config.text_config)
        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1
    

    def tie_weights(self):
        return self.language_model.tie_weights()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "I will add padding later"

        # shape: batch_size, seq_len, hidden_size
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # batch_size, channels, height, width -> batch_size, num_patches, embed_dim
        image_embeds = self.vision_tower(pixel_values.to(self.input_embeds.dtype))

        # batch_size, num_patches, embed_dim -> batch_size, num_patches, hidden_size
        # basically convert the image embeddings to the same dimension as the text embeddings
        image_embeds = self.multi_model_projector(image_embeds)

        input_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_embeds(image_embeds, input_embeds, input_ids, attention_mask, kv_cache)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            kv_cache=kv_cache,
        )

        return outputs
    
    def _merge_input_ids_with_image_embeds(
        self,
        image_embeds: torch.Tensor,
        input_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ):
        _, _, embed_dim = input_embeds.shape
        batch_size, seq_length = image_embeds.shape
        dtype, device = input_embeds.dtype, input_embeds.device

        # TODO: understand why they scale this way?
        scaled_image_embeds = image_embeds / math.sqrt(self.config.hidden_size**0.5)
        
        final_embeds = torch.zeros(batch_size, seq_length, embed_dim, device=device, dtype=dtype )

        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        image_mask = (input_ids == self.config.image_token_index)
        padding_mask = (input_ids == self.pad_token_id)

        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, seq_length)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, seq_length)
        padding_mask_expanded = padding_mask.unsqueeze(-1).expand(-1, -1, seq_length)

        # insert the text embeddings to the final embeddings
        final_embeds = torch.where(text_mask_expanded, input_embeds, final_embeds)
        
        # insert the image embeddings to the final embeddings. can't use torch.where because sequence length of 
        # scaled_image_embeds is different from final_embeds
        final_embeds = torch.masked_scatter(image_mask_expanded, scaled_image_embeds)

        # insert the padding embeddings to the final embeddings
        final_embeds = torch.where(padding_mask_expanded, torch.zeros_like(final_embeds), final_embeds)

    
        # create the attention mask
        dtype, device = input_embeds.dtype, input_embeds.device
        min_dtype = torch.finfo(dtype=dtype).min
        q_len = input_embeds.shape[1]
        if kv_cache is None or kv_cache.num_items() == 0:
            causal_mask = torch.full((batch_size, q_len, q_len), fill_value=0, device=device, dtype=dtype)
        else:
            assert q_len == 1 # query length must be 1
            kv_len = kv_cache.num_items() + q_len
            causal_mask = torch.zeros((batch_size, q_len, kv_len), device=device, dtype=dtype)

        # add the head dimension
        # b, q_len, kv_len -> b, num_heads_q, q_len, kv_len

        if kv_cache is not None and kv_cache.num_items() > 0:
            # the position of the query is just the last position
            position_ids = attention_mask.cumsum(dim=-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = (attention_mask.cumsum(dim=-1)).masked_fill((attention_mask == 0), 1).to(device)
        
        return final_embeds, causal_mask, position_ids