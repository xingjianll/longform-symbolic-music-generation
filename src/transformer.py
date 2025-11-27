# This is lightly adapted from https://github.com/EleutherAI/aria/blob/main/aria/model.py

from typing import Optional, Union, Tuple

import torch
import torch.utils.checkpoint

from torch import nn as nn
from torch.nn import functional as F, CrossEntropyLoss, MSELoss

from transformers import Cache, DynamicCache, StaticCache
from transformers.utils import logging
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    BaseModelOutputWithPoolingAndProjection,
)

from src.utils import EPOCHS

logger = logging.get_logger(__name__)


from transformers import PretrainedConfig


class AriaConfig(PretrainedConfig):
    model_type = "aria"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 17727,
        hidden_size: int = 1536,
        embedding_size: int | None = None,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 64,
        intermediate_size: int = 6144,
        max_seq_len: int = 8192,
        use_cache: bool = True,
        eos_token_id: int = 1,
        pad_token_id: int = 2,
        tie_word_embeddings: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.return_dict = return_dict

        if self.intermediate_size % self.hidden_size != 0:
            raise ValueError(
                "The intermediate size needs to be divisible by hidden size."
            )

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size needs to be divisible by the number of attention heads."
            )

    @property
    def ff_mult(self):
        return self.intermediate_size // self.hidden_size


class AriaPreTrainedModel(PreTrainedModel):
    config_class = AriaConfig
    base_model_prefix = "aria"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AriaBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_sdpa = True
    _supports_flex_attn = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=0.02
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=0.02
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class TransformerBlock(nn.Module):
    def __init__(self, model_config, layer_idx: int):
        super().__init__()

        self.drop_p = 0.0
        self.n_heads = model_config.num_attention_heads
        self.d_model = model_config.hidden_size
        self.d_head = (
            model_config.hidden_size // model_config.num_attention_heads
        )
        self.max_seq_len = model_config.max_seq_len
        self.layer_idx = layer_idx

        # Attention
        self.mixed_qkv = nn.Linear(
            in_features=self.d_model,
            out_features=3 * self.d_model,
            bias=False,
        )
        self.att_proj_linear = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_model,
            bias=False,
        )

        # FF Layer
        self.ff_gate_proj = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_model * model_config.ff_mult,
            bias=False,
        )
        self.ff_up_proj = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_model * model_config.ff_mult,
            bias=False,
        )
        self.ff_down_proj = nn.Linear(
            in_features=self.d_model * model_config.ff_mult,
            out_features=self.d_model,
            bias=False,
        )

        # Pre layer norms
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            Union[Cache, Tuple[Tuple[torch.FloatTensor]]]
        ] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ):
        attn_output, attn_weights, present = self._att_block(
            self.norm1(x),
            attention_mask,
            freqs_cis,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )

        x = x + attn_output
        x = x + self._ff_block(self.norm2(x))

        outputs = (x, present)
        if use_cache:
            outputs = (x, present, attn_weights)
        else:
            outputs = (x, attn_weights)

        return outputs

    def _att_block(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_key_values: Optional[
            Union[Cache, Tuple[Tuple[torch.FloatTensor]]]
        ] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len, _ = x.shape
        mixed_qkv = self.mixed_qkv(x)
        xq, xk, xv = mixed_qkv.chunk(3, -1)

        # Reshape for rotary embeddings
        # Need contiguous for q, k since in-place RoPE cannot be applied on a view
        xq = xq.reshape(
            batch_size, seq_len, self.n_heads, self.d_head
        ).contiguous()
        xk = xk.reshape(
            batch_size, seq_len, self.n_heads, self.d_head
        ).contiguous()
        xv = xv.view(batch_size, seq_len, self.n_heads, self.d_head)

        # apply_rotary_post_emb expects: (b_sz, s_len, n_head, d_head)
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        xq, xk, xv = map(lambda t: t.transpose(1, 2), (xq, xk, xv))

        if past_key_values is not None:
            cache_kwargs = {
                # "sin": sin,
                # "cos": cos,
                # "partial_rotation_size": self.rotary_ndims,
                "cache_position": cache_position,
            }
            xk, xv = past_key_values.update(
                xk, xv, self.layer_idx, cache_kwargs
            )

        att = F.scaled_dot_product_attention(
            query=xq,
            key=xk,
            value=xv,
            attn_mask=attention_mask[..., : xk.shape[2]],
        )

        # Reshape for out: (b_sz, s_len, n_head, d_head)
        out = att.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.n_heads * self.d_head)

        if not output_attentions:
            att = None

        return self.att_proj_linear(out), att, past_key_values

    def _ff_block(self, x: torch.Tensor):
        return self.ff_down_proj(
            F.silu(self.ff_gate_proj(x)) * self.ff_up_proj(x)
        )


class AriaModel(AriaPreTrainedModel):
    """Transformer decoder with no language model head.
    Args:
        model_config (ModelConfig): Model config settings.
    """

    def __init__(self, model_config: AriaConfig):
        super().__init__(model_config)
        self.model_config = model_config
        self.freqs_cis = None
        self.causal_mask = None

        self.tok_embeddings = nn.Linear(
            4, model_config.hidden_size, bias=False
        )

        self.out_layer_norm = nn.LayerNorm(model_config.hidden_size)
        self.encode_layers = nn.ModuleList()
        for i in range(model_config.num_hidden_layers):
            self.encode_layers.append(TransformerBlock(model_config, i))

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            Union[Cache, Tuple[Tuple[torch.FloatTensor]]]
        ] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ):
        """Forward pass of Transformer.
        Args:
            src (torch.tensor): Input to encoder block, of shape (batch_size,
                seq_len, d_model).
            attn_mask (Optional[torch.tensor]): Attention mask of shape
                (batch_size, seq_len). Defaults to None.
            past_kv (Optional[list[KVCache]]): a list of kv caches. The list index
                corresponds to the layer index.
        Returns:
            torch.tensor: Model outputs with shape (batch_size, seq_len,
                d_model).
        """
        if (
            input_ids is not None
            and input_ids.shape[1] > self.model_config.max_seq_len
        ):
            raise ValueError(
                f"Sequence length ({input_ids.shape[1]}) exceeds max_seq_len "
                f"({self.model_config.max_seq_len})."
            )
        if (
            inputs_embeds is not None
            and inputs_embeds.shape[1] > self.model_config.max_seq_len
        ):
            raise ValueError(
                f"Sequence length ({inputs_embeds.shape[1]}) exceeds max_seq_len "
                f"({self.model_config.max_seq_len})."
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.model_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.model_config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.model_config.use_return_dict
        )
        use_cache = (
            use_cache if use_cache is not None else self.model_config.use_cache
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.tok_embeddings(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(
                    past_key_values
                )
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        seq_length = inputs_embeds.shape[1]
        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length()
                if past_key_values is not None
                else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + seq_length,
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        hidden_states = inputs_embeds

        if self.causal_mask is None:
            self.causal_mask = precompute_causal_mask(
                max_seq_len=self.model_config.max_seq_len,
            ).to(input_ids.device)

        if self.freqs_cis is None:
            self.freqs_cis = precompute_freqs_cis(
                seq_len=self.model_config.max_seq_len,
                n_elem=self.model_config.hidden_size
                // self.model_config.num_attention_heads,
                base=500000,
                dtype=hidden_states.dtype,
            ).to(input_ids.device)

        freqs_cis = self.freqs_cis[cache_position]

        if use_cache is True:
            causal_mask = self.causal_mask[None, None, cache_position]
        else:
            causal_mask = self.causal_mask[None, None, :seq_length, :seq_length]

        if attention_mask is not None:
            pad_len = causal_mask.shape[3] - attention_mask.shape[1]
            padded_attention_mask = F.pad(attention_mask, (0, pad_len), value=1)
            padded_attention_mask = padded_attention_mask[:, None, None, :]
            padded_attention_mask = padded_attention_mask.bool()

            causal_mask = causal_mask & padded_attention_mask

        kwargs = {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
            "cache_position": cache_position,
        }
        next_decoder_cache = None
        if self.gradient_checkpointing:
            for layer in self.encode_layers:

                def create_custom_forward(module):
                    def custom_forward(*args):
                        return module(*args)[0]

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    causal_mask,
                    freqs_cis,
                    **kwargs,
                    preserve_rng_state=True,
                    use_reentrant=True,
                )
        else:
            all_attentions = () if output_attentions else None
            all_hidden_states = () if output_hidden_states else None
            for layer in self.encode_layers:
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                outputs = layer(
                    hidden_states, causal_mask, freqs_cis=freqs_cis, **kwargs
                )
                hidden_states = outputs[0]
                if use_cache is True:
                    next_decoder_cache = outputs[1]
                if output_attentions:
                    all_attentions = all_attentions + (
                        outputs[2 if use_cache else 1],
                    )
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.out_layer_norm(hidden_states)
        next_cache = next_decoder_cache if use_cache else None

        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class AriaForCausalLM(AriaPreTrainedModel, GenerationMixin):
    """Transformer decoder with head for language modelling.
    Args:
        model_config (ModelConfig): Model config settings.
    """

    def __init__(self, model_config: AriaConfig):
        super().__init__(model_config)
        self.model_config = model_config
        self.max_seq_len = model_config.max_seq_len
        self.model = AriaModel(model_config)
        self.lm_head = nn.Linear(
            model_config.hidden_size, 4, bias=False
        )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            Union[Cache, Tuple[Tuple[torch.FloatTensor]]]
        ] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ):
        """Forward pass of Transformer decoder with LM head."""
        return_dict = (
            return_dict
            if return_dict is not None
            else self.model_config.use_return_dict
        )
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden = outputs[0]
        lm_logits = self.lm_head(hidden)

        lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # we are doing next-token prediction; shift prediction scores and input ids by one
            loss_fct = MSELoss()
            lm_loss = loss_fct(lm_logits, labels)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def precompute_causal_mask(max_seq_len: int):
    return torch.tril(
        torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)
    ).cuda()


def precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    base: int = 500000,
    dtype: torch.dtype = torch.bfloat16,
):
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)

    return cache.to(dtype=dtype)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    In-place RoPE. Credits to Katherine Crowson:
    x shape (b_sz, s_len, n_head, d_head).
    cos, sin shape (s_len, d_head // 2).
    """

    d = x.shape[-1] // 2
    cos = freqs_cis[..., 0][None, :, None]
    sin = freqs_cis[..., 1][None, :, None]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    tmp = x1.clone()
    x1.mul_(cos).addcmul_(x2, sin, value=-1)
    x2.mul_(cos).addcmul_(tmp, sin, value=1)
    return x


# Copied from https://github.com/EleutherAI/aria/blob/main/aria/training/train.py
def _get_optim(
    lr: float,
    model: nn.Module,
    num_epochs: int,
    steps_per_epoch: int,
    warmup: int = 100,
    end_ratio: float = 0.1,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-5,
    )

    warmup_lrs = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.000001,
        end_factor=1,
        total_iters=warmup,
    )
    linear_decay_lrs = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1,
        end_factor=end_ratio,
        total_iters=(num_epochs * steps_per_epoch) - warmup,
    )

    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_lrs, linear_decay_lrs],
        milestones=[warmup],
    )

    return optimizer, lr_scheduler


import lightning as pl
from transformers import AutoConfig


class MidiAria(pl.LightningModule):
    def __init__(self, dataloader, lr=2e-4, warmup_steps=10):
        super().__init__()
        # self.save_hyperparameters()
        self.config = AriaConfig()
        self.model = AriaForCausalLM(self.config)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.dataloader = dataloader

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        # Handle both tuple and object returns from PEFT models
        if isinstance(outputs, tuple):
            loss = outputs[0]
        else:
            loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        # Handle both tuple and object returns from PEFT models
        if isinstance(outputs, tuple):
            val_loss = outputs[0]
        else:
            val_loss = outputs.loss
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):

        steps_per_epoch = len(self.dataloader)
        optimizer, scheduler = _get_optim(
            lr=self.lr,
            model=self,
            num_epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            warmup=self.warmup_steps,
            end_ratio=0.1,
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}