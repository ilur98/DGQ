import torch
from torch import nn
import math
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaAttention,
    LlamaDecoderLayer,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaRMSNorm,
    BaseModelOutputWithPast
)
from typing import Optional, Tuple, Union, List
from dgq.models.linear import W4A8BF32OF32Linear
from dgq.models.fused import RMSNormQ
from transformers.utils import logging
from dgq.models.bmm import BMM_S8T_S8N_F32T
logger = logging.get_logger(__name__)


class W4A8LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: LlamaConfig
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.k_proj = W4A8BF32OF32Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.v_proj = W4A8BF32OF32Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.q_proj = W4A8BF32OF32Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = W4A8BF32OF32Linear(self.hidden_size, self.hidden_size)
        self.register_buffer("input_scale", torch.tensor([0.1], dtype=torch.float))
        self.register_buffer("q_proj_scale", torch.tensor([0.1], dtype=torch.float))
        self.register_buffer("k_proj_scale", torch.tensor([0.1], dtype=torch.float))
        self.register_buffer("v_proj_scale", torch.tensor([0.1], dtype=torch.float))
        self.register_buffer("out_input_scale", torch.tensor([0.1], dtype=torch.float))
        self._init_rope()

    _init_rope = LlamaAttention._init_rope
    _shape = LlamaAttention._shape
    @staticmethod
    @torch.no_grad()
    def from_float(module: LlamaAttention,
                   config: LlamaConfig,
                   input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   out_input_scale: float):
        a8w4_module = W4A8LlamaAttention(config)
        a8w4_module.q_proj = W4A8BF32OF32Linear.from_float(
            module.q_proj, input_scale)
        a8w4_module.k_proj = W4A8BF32OF32Linear.from_float(
            module.k_proj, input_scale)
        a8w4_module.v_proj = W4A8BF32OF32Linear.from_float(
            module.v_proj, input_scale)
        a8w4_module.input_scale = input_scale
        a8w4_module.q_proj_scale = q_output_scale
        a8w4_module.k_proj_scale = k_output_scale
        a8w4_module.v_proj_scale = v_output_scale
        a8w4_module.out_input_scale = out_input_scale
        a8w4_module.o_proj = W4A8BF32OF32Linear.from_float(
            module.o_proj, out_input_scale)

        return a8w4_module

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        query_states = torch.round(query_states / self.q_proj_scale).clamp(-128, 127).to(torch.int8)
        key_states = torch.round(key_states / self.k_proj_scale).clamp(-128, 127).to(torch.int8)
        value_states = torch.round(value_states / self.v_proj_scale).clamp(-128, 127).to(torch.int8)
        
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        query_states = query_states * self.q_proj_scale
        key_states = key_states * self.k_proj_scale
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights.view(bsz, self.num_heads, q_len, kv_seq_len)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        value_states = value_states * self.v_proj_scale
        attn_output = torch.matmul(attn_weights, value_states)
        
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = torch.round(attn_output / self.out_input_scale).clamp(-127, 127).to(torch.int8)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value


class A8W4LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = W4A8LlamaAttention(
            config
        )

        self.input_layernorm = RMSNormQ(
            config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = A8W4LlamaMLP(config)
        self.post_attention_layernorm = RMSNormQ(self.hidden_size)

    @staticmethod
    def from_float(module: LlamaDecoderLayer,
                   config: LlamaConfig,
                   attn_input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   out_input_scale: float,
                   mlp_input_scale: float,
                   down_input_scale: float):
        a8w4_module = A8W4LlamaDecoderLayer(
            config
        )
        a8w4_module.input_layernorm = RMSNormQ.from_float(
            module.input_layernorm, attn_input_scale)
        a8w4_module.self_attn = W4A8LlamaAttention.from_float(
            module.self_attn, config, attn_input_scale, q_output_scale, k_output_scale, v_output_scale, out_input_scale)
        a8w4_module.post_attention_layernorm = RMSNormQ.from_float(
            module.post_attention_layernorm, mlp_input_scale)
        a8w4_module.mlp = A8W4LlamaMLP.from_float(module.mlp, mlp_input_scale, down_input_scale)
        return a8w4_module

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
 
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        residual.add_(hidden_states.to(residual.dtype))

        # Fully Connected
        # residual = hidden_states
        hidden_states = self.post_attention_layernorm(residual)
        hidden_states = self.mlp(hidden_states)

        residual.add_(hidden_states.to(residual.dtype))

        outputs = (residual,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class A8W4LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = W4A8BF32OF32Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = W4A8BF32OF32Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = W4A8BF32OF32Linear(self.intermediate_size, self.hidden_size)
        self.register_buffer("down_input_scale", torch.tensor([0.1], dtype=torch.float))

    @staticmethod
    def from_float(module: LlamaDecoderLayer,
                   mlp_input_scale: float,
                   down_input_scale: float):
        a8w4_module = A8W4LlamaMLP(
            module.config
        )
        a8w4_module.act_fn = module.act_fn
        a8w4_module.gate_proj = W4A8BF32OF32Linear.from_float(module.gate_proj, mlp_input_scale)
        a8w4_module.up_proj = W4A8BF32OF32Linear.from_float(module.up_proj, mlp_input_scale)
        a8w4_module.down_proj = W4A8BF32OF32Linear.from_float(module.down_proj, down_input_scale)
        a8w4_module.down_input_scale = down_input_scale
        return a8w4_module

    def forward(self, x):
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        x = torch.round(x / self.down_input_scale).clamp(-128, 127).to(torch.int8)
        down_proj = self.down_proj(x)

        return down_proj

class A8W4LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([A8W4LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
    get_input_embeddings = LlamaModel.get_input_embeddings
    set_input_embeddings = LlamaModel.set_input_embeddings
    _prepare_decoder_attention_mask = LlamaModel._prepare_decoder_attention_mask
    forward = LlamaModel.forward

    @staticmethod
    def from_float(module, decoder_layer_scales):
        a8w4_module = A8W4LlamaModel(module.config)
        a8w4_module.embed_tokens = module.embed_tokens
        a8w4_module.norm = module.norm
        for i, layer in enumerate(module.layers):
            a8w4_module.layers[i] = A8W4LlamaDecoderLayer.from_float(
                    layer, module.config, **decoder_layer_scales[i])
        return a8w4_module


class A8W4LlamaForCausalLM(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = A8W4LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def from_float(module, decoder_layer_scales):
        a8w4_module = A8W4LlamaForCausalLM(module.config)
        a8w4_module.model = A8W4LlamaModel.from_float(
            module.model, decoder_layer_scales)
        a8w4_module.lm_head = module.lm_head
        return a8w4_module

    get_input_embeddings = LlamaForCausalLM.get_input_embeddings
    set_input_embeddings = LlamaForCausalLM.set_input_embeddings
    get_output_embeddings = LlamaForCausalLM.get_output_embeddings
    set_output_embeddings = LlamaForCausalLM.set_output_embeddings
    set_decoder = LlamaForCausalLM.set_decoder
    get_decoder = LlamaForCausalLM.get_decoder
    forward = LlamaForCausalLM.forward
    prepare_inputs_for_generation = LlamaForCausalLM.prepare_inputs_for_generation
    _reorder_cache = LlamaForCausalLM._reorder_cache
