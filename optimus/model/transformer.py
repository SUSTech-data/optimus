# Copyright (c) 2021 EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
TODO: 
1. flash attention 2 support for generating (left padding decoding)
2. memory effecient GQA forward of eval mode (left padding maybe)
"""

"""Transformer."""

import logging
import math
import torch
import torch.nn.functional as F
import torch.nn as nn

from .norms import get_norm
import optimus.mpu as mpu
from optimus.model.fused_softmax import FusedScaleMaskSoftmax
from optimus.model.activations import get_activation
from optimus.model.utils import exists, get_fusion_type
from optimus.model.positional_embeddings import (
    RotaryEmbedding,
    RotaryEmbeddingLegacy,
    apply_rotary_pos_emb_torch,
    apply_rotary_pos_emb_torch_for_left_padding,
    apply_rotary_pos_emb,
    AliBi,
)
from optimus.model.fused_bias_dropout import (
    get_bias_dropout_add,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
)
from optimus.model.utils import configure_sparse_attention

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
        attention_mask_func: a function that takes `unmasked-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
               masked-attention-scores = attention_mask_func(
                                     unmasked-attention-scores, attention-mask)
"""


def flatten_sequence(batch, attention_mask):
    """
    Flattens a batch of sequences.

    Args:
    - batch (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing sequences.
    - attention_mask (torch.Tensor): A tensor of shape (batch_size, sequence_length) indicating valid tokens.

    Returns:
    - torch.Tensor: A single flattened sequence.
    - torch.Tensor: Tensor of end positions for the original sequences in the batch.
    """
    valid_tokens = batch[attention_mask.bool()]
    end_positions = attention_mask.sum(dim=1).cumsum(0).int()
    return valid_tokens, end_positions


def restore_sequence(flattened, end_positions, original_length, pad_value=0):
    """
    Restores the original batch of sequences from a flattened sequence.

    Args:
    - flattened (torch.Tensor): The flattened sequence.
    - end_positions (torch.Tensor): Tensor of end positions for the original sequences in the batch.
    - original_length (int): Length of sequences in the original batch.
    - pad_value (int, optional): Value to use for padding. Default is 0.

    Returns:
    - torch.Tensor: Tensor of shape (batch_size, original_length) containing the restored sequences.
    """
    sequences = []
    start_pos = 0
    for end_pos in end_positions:
        seq = flattened[start_pos:end_pos]
        seq = F.pad(seq, (original_length - len(seq), 0), value=pad_value)
        sequences.append(seq)
        start_pos = end_pos

    return torch.stack(sequences)


class ParallelMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(
        self, neox_args, init_method, output_layer_init_method, parallel_output=False
    ):
        super().__init__()

        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation
        self.bias_gelu_fusion = neox_args.bias_gelu_fusion

        # auto scale so geglu has equal parameters
        ff_mult = int(4 * 2 / 3) if self.activation_type == "geglu" else 4
        ff_dim = (
            int(ff_mult * neox_args.hidden_size) * 2
            if self.activation_type == "geglu"
            else ff_mult * neox_args.hidden_size
        )
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
        )
        ff_dim_in = ff_dim // 2 if self.activation_type == "geglu" else ff_dim
        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=ff_dim_in,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if (
            self.activation_type == "gelu" and self.bias_gelu_fusion
        ) or self.activation_type == "geglu":
            intermediate_parallel = self.activation_func(
                intermediate_parallel, bias_parallel
            )
        else:
            intermediate_parallel = self.activation_func(
                intermediate_parallel + bias_parallel
            )

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class LLaMAParallelMLP(nn.Module):
    """LLaMA's MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Note: multiple_of is used to compute the hidden dimension of the MLP
    """

    def __init__(
        self,
        neox_args,
        init_method,
        output_layer_init_method,
        parallel_output=False,
        multiple_of=256,
    ):
        super().__init__()

        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation

        self.multiple_of = multiple_of

        ff_dim = int(2 * neox_args.hidden_size * 4 / 3)
        ff_dim = self.multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)
        if hasattr(neox_args, "intermediate_size"):
            ff_dim = neox_args.intermediate_size

        self.w1 = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
        )
        self.w3 = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
        )
        self.w2 = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=ff_dim,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
            bias=False,
        )

    def forward(self, hidden_states):
        w1_out, _ = self.w1(hidden_states)
        w3_out, _ = self.w3(hidden_states)
        return self.w2(self.activation_func(w1_out) * w3_out)


class ParallelLinear(nn.Module):
    """
    A Parallel Linear Layer transforming the transformer outputs from hidden_size -> vocab_size
    """

    def __init__(
        self,
        neox_args,
        parallel_output=True,
        init_method=nn.init.xavier_normal_,
        is_last_layer=False,
    ):
        super().__init__()
        parallelism = neox_args.output_layer_parallelism
        if parallelism == "column":
            self.final_linear = mpu.ColumnParallelLinear(
                neox_args=neox_args,
                input_size=neox_args.hidden_size,
                output_size=neox_args.padded_vocab_size,
                bias=False,
                init_method=init_method,
                gather_output=not parallel_output,
                skip_bias_add=False,
                mup_rescale_parameters=is_last_layer,  # rescale params only called if neox_args.use_mup = True, despite it not being included here
            )

    #        else:
    #            print(
    #                'ERROR: Output layer parallelism over the hidden dim is currently broken (https://github.com/EleutherAI/gpt-neox/issues/905). Please run with output_layer_parallelism = "column" until this issue is fixed.'
    #            )
    #            exit()
    #            self.final_linear = mpu.RowParallelLinear(
    #                neox_args=neox_args,
    #                input_size=neox_args.hidden_size,
    #                output_size=neox_args.padded_vocab_size,
    #                bias=False,
    #                input_is_parallel=False,
    #                init_method=init_method,
    #                parallel_output=parallel_output,
    #                skip_bias_add=False,
    #                mup_rescale_parameters=is_last_layer,  # only called if neox_args.use_mup = True, despite it not being included here
    #            )

    def forward(self, hidden_states):
        return self.final_linear(hidden_states)


class ParallelSelfAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        use_cache=False,
        parallel_output=False,
    ):
        super().__init__()

        self.fp16 = neox_args.precision == "fp16"
        self.bf16 = neox_args.precision == "bfloat16"
        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = neox_args.apply_query_key_layer_scaling
        self.use_cache = use_cache
        self.attention_softmax_in_fp32 = neox_args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = layer_number
        # Per attention head and per partition values.
        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(neox_args.hidden_size, world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            neox_args.hidden_size, neox_args.num_attention_heads
        )
        self.num_attention_heads_per_partition = mpu.divide(
            neox_args.num_attention_heads, world_size
        )
        self.pos_emb = neox_args.pos_emb

        self.isGQA = bool(
            getattr(neox_args, "isGQA", False)
        )  # prevent return None, must clarify in config
        if self.isGQA:
            if hasattr(neox_args, "num_key_value_heads"):
                self.num_kv_heads = neox_args.num_key_value_heads
            elif hasattr(neox_args, "num_kv_heads"):
                self.num_kv_heads = neox_args.num_kv_heads
            else:
                raise AttributeError("Must have some attribute for num_kv_heads")

            self.num_q_head_per_partition = mpu.divide(
                neox_args.num_attention_heads, world_size
            )
            self.num_kv_head_per_partition = mpu.divide(
                neox_args.num_kv_heads, world_size
            )
            self.head_size = mpu.divide(
                neox_args.hidden_size, neox_args.num_attention_heads
            )
            self.qkv_linear_output_size = (
                neox_args.hidden_size + neox_args.num_kv_heads * self.head_size * 2
            )  # q + kv, [q, k, v] order
        else:
            self.qkv_linear_output_size = neox_args.hidden_size * 3

        # Strided linear layer.
        self.query_key_value = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            # output_size=3 * neox_args.hidden_size,
            output_size=self.qkv_linear_output_size,  # always map hidden state to [q, k, v]
            gather_output=False,
            init_method=init_method,
            bias=neox_args.use_bias_in_attn_linear,
        )

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = max(1, self.layer_number)
            self.norm_factor *= coeff

        if neox_args.use_mup:
            self.norm_factor = self.hidden_size_per_attention_head

        self.rpe = rpe

        if self.pos_emb == "alibi":
            self.alibi_embed = AliBi(
                neox_args.num_attention_heads,
                neox_args.model_parallel_size,
                mpu.get_model_parallel_rank(),
            )

        # TODO: this arg shouldn't need to be passed in - get from neox_args
        if rotary:
            if neox_args.rotary_pct == 1:
                self.rotary_ndims = None
            else:
                assert neox_args.rotary_pct < 1
                self.rotary_ndims = int(
                    self.hidden_size_per_attention_head * neox_args.rotary_pct
                )
            dim = (
                self.rotary_ndims
                if self.rotary_ndims is not None
                else self.hidden_size_per_attention_head
            )
            self.rotary_emb = RotaryEmbedding(
                dim, base=neox_args.rotary_emb_base, precision=neox_args.params_dtype
            )
            self.rotary_emb_legacy = RotaryEmbeddingLegacy(
                dim,
                base=neox_args.rotary_emb_base,
                max_position_embeddings=neox_args.max_position_embeddings,
            )
        else:
            self.rotary_emb = None

        self.attention_type = neox_args.attention_config[layer_number]
        self.use_flash_attention = self.attention_type == "flash"
        self.sparse = self.attention_type not in ("global", "flash")
        if self.sparse:
            self.sparse_attn = configure_sparse_attention(
                neox_args,
                self.attention_type,
                self.num_attention_heads_per_partition,
                mpu=mpu,
            )
        else:

            def map_int(x):
                try:
                    return int(x)
                except ValueError:
                    return 0

            if self.use_flash_attention:
                """
                Optimus only support flash attention > 2.1
                because only uppon 2.1, causal mask of fmha
                can handle the left padding generation
                """
                # self.use_flash_attn2 = False
                try:
                    import flash_attn

                    flash_attn_version = tuple(
                        map(map_int, flash_attn.__version__.split("."))
                    )
                    if flash_attn_version < (2, 1, 0):
                        raise ImportError("Please upgrade flash_attn to >= 2.1.0")
                    else:
                        from flash_attn import (
                            flash_attn_qkvpacked_func,
                            flash_attn_func,
                            flash_attn_kvpacked_func,
                            flash_attn_varlen_qkvpacked_func,
                            flash_attn_varlen_kvpacked_func,
                        )

                        self.flash_qkv_fn = flash_attn_qkvpacked_func
                        self.flash_attn_fn = flash_attn_func
                        self.flash_kv_fn = flash_attn_kvpacked_func

                        self.flash_var_qkv_fn = flash_attn_varlen_qkvpacked_func
                        self.flash_var_kv_fn = flash_attn_varlen_kvpacked_func
                        # self.use_flash_attn2 = True

                except ImportError:
                    logging.warning("flash_attn not found, using default attn")
                    logging.warning("Please upgrade flash_attn to >= 2.1.0")
                    self.use_flash_attention = False
                    # raise ImportError("Please install flash_attn >= 2.1.0")

            # use for flash attention not enabled
            self.scale_mask_softmax = FusedScaleMaskSoftmax(
                input_in_fp16=self.fp16,
                input_in_bf16=self.bf16,
                fusion_type=get_fusion_type(neox_args),
                mask_func=self.attention_mask_func,
                softmax_in_fp32=self.attention_softmax_in_fp32,
                scale=coeff,
            )

            # Dropout. Note that for a single iteration, this layer will generate
            # different outputs on different number of parallel partitions but
            # on average it should not be partition dependent.
            self.dropout_p = neox_args.attention_dropout
            self.attention_dropout = nn.Dropout(self.dropout_p)

        # Output.
        self.dense = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
            bias=neox_args.use_bias_in_attn_linear,
        )

    def attention(
        self, query_layer, key_layer, value_layer, layer_past, attention_mask
    ):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if self.use_cache:
            with torch.no_grad():
                attention_mask = attention_mask[
                    ..., : attention_scores.size(3), : attention_scores.size(3)
                ]

        # ===========================
        # Attention probs and dropout
        # ===========================

        if exists(self.rpe):
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
            attention_scores += rpe  # [1, np, sq, sk]

        if self.pos_emb == "alibi":
            attention_scores = self.alibi_embed(attention_scores)

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer

    def flash_attention(
        self, query_layer, key_layer, value_layer, non_causal_attention_mask
    ):
        # assert self.use_flash_attn2, "flash attn <2 not implemented in optimus"
        if self.pos_emb != "rotary":
            raise NotImplementedError("Flash attention 2 requires rotary pos emb")

        if non_causal_attention_mask is None:
            # [b, np, sq, sk]
            output_size = (
                query_layer.size(1),
                query_layer.size(2),
                query_layer.size(0),
                key_layer.size(0),
            )
            # [sq, b, np, hn] -> [b, sq, 1, np, hn] for both q,k,v
            kv_shape = key_layer.shape  # [sq, b, np_kv, hn]
            key_layer = key_layer.transpose(0, 1).reshape(
                kv_shape[1], kv_shape[0], 1, kv_shape[2], -1
            )
            value_layer = value_layer.transpose(0, 1).reshape(
                kv_shape[1], kv_shape[0], 1, kv_shape[2], -1
            )
            """
            This means we dont need to consider left padding situation,
            used in training and non-left padding generation, i.e. single sample
            generation
            """
            if self.isGQA:
                q_shape = query_layer.shape  # [sq, b, np_q, hn]
                query_layer = query_layer.transpose(0, 1).reshape(
                    q_shape[1], q_shape[0], q_shape[2], -1
                )  # reshape to continguous
                kv = torch.cat([key_layer, value_layer], dim=2)
                # logging.info
                output = self.flash_kv_fn(
                    query_layer,
                    kv,
                    self.dropout_p if self.training else 0.0,
                    causal=True,
                )

            else:
                # Combined q/k/v into [b, s, 3, np, hn].
                query_layer = query_layer.transpose(0, 1).reshape(
                    output_size[0], output_size[2], 1, output_size[1], -1
                )
                qkv = torch.cat([query_layer, key_layer, value_layer], dim=2)
                output = self.flash_qkv_fn(
                    qkv,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=None,
                    causal=True,
                )

            # qkv: (batch_size, seqlen, 3, nheads, headdim)
            # out: (batch_size, seqlen, nheads, headdim).
            matmul_result = output.view(
                output_size[0], output_size[2], output.shape[2], output.shape[3]
            )

            # [b, sq, np, hn] -> [b, np, sq, hn]
            matmul_result = matmul_result.transpose(1, 2)
            return matmul_result

        """
        Left padding generation
        """
        if self.isGQA:
            # TODO:: support MQA & GQA
            raise NotImplementedError("GQA & MQA Not support yet")

        q_origin_len, batch_size = query_layer.shape[:2]
        kv_origin_len = key_layer.shape[0]

        # input_shape of qkv [sq, b, np, hn]
        flatten_attention_mask = non_causal_attention_mask.flatten().bool()
        end_positions = non_causal_attention_mask.sum(dim=1).cumsum(0).int()

        qkv_shape = query_layer.shape  # used 2,3 for both q,k,v have
        query_layer = query_layer.transpose(0, 1).reshape(
            batch_size * q_origin_len, qkv_shape[2], qkv_shape[3]
        )
        if q_origin_len != 1:
            query_layer = query_layer[flatten_attention_mask]

        key_layer = key_layer.transpose(0, 1).reshape(
            batch_size * kv_origin_len, 1, qkv_shape[2], qkv_shape[3]
        )[flatten_attention_mask]
        value_layer = value_layer.transpose(0, 1).reshape(
            batch_size * kv_origin_len, 1, qkv_shape[2], qkv_shape[3]
        )[flatten_attention_mask]
        kv = torch.cat([key_layer, value_layer], dim=1)  # concat at second dim
        kv_cum_len = torch.cat(
            [
                torch.tensor(
                    [0], device=end_positions.device, dtype=end_positions.dtype
                ),
                end_positions,
            ]
        )
        kv_max_seqlen = non_causal_attention_mask.sum(dim=1).max().int()
        q_cum_len = (
            kv_cum_len
            if q_origin_len == kv_origin_len
            else torch.cat(
                [
                    torch.tensor([0], device=end_positions.device, dtype=torch.int32),
                    torch.arange(
                        1,
                        batch_size + 1,
                        device=end_positions.device,
                        dtype=torch.int32,
                    ),
                ]
            )
        )
        q_max_seqlen = kv_max_seqlen if q_origin_len == kv_origin_len else 1
        # out shape [b*sq, np, hn]
        out = self.flash_var_kv_fn(
            query_layer,
            kv,
            q_cum_len,
            kv_cum_len,
            q_max_seqlen,
            kv_max_seqlen,
            softmax_scale=None,
            causal=True,
        )
        # out = self.flash_var_qkv_fn(
        #     qkv,
        #     kv_cum_len,
        #     max_seqlen,
        #     softmax_scale=None,
        #     causal=True,
        # )

        # TODO: restore should be removed

        # restore
        seqs = []
        start_position = 0
        for i in q_cum_len[1:]:
            pad_length = q_origin_len - (i - start_position)
            seq = out[start_position:i]  # first dim
            # print(pad_length, seq.shape, i, cum_len)
            seq = F.pad(
                seq, (0, 0, 0, 0, pad_length, 0), value=0.0
            )  # pad to max_seqlen
            seqs.append(seq)

        # [b, sq, np, hn]
        out = torch.stack(seqs, dim=0)

        # [b, sq, np, hn] -> [b, np, sq, hn]
        out = out.transpose(1, 2)

        return out

    def sparse_attention(self, query_layer, key_layer, value_layer, attention_mask):
        # TODO: sparse attn dropout?
        # TODO: pad to block size
        # shape of q/k/v is [sq, b, np, hn] and needs to be transposed to [b, np, sq, hn]
        query_layer, key_layer, value_layer = map(
            lambda t: t.permute(1, 2, 0, 3).contiguous(),
            (query_layer, key_layer, value_layer),
        )
        # output shape [b, np(heads), sq, hn]
        attn_mask = attention_mask.to(query_layer.dtype) * -10000
        if exists(self.rpe):
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
        else:
            rpe = None
        return self.sparse_attn(
            query_layer, key_layer, value_layer, attn_mask=attn_mask, rpe=rpe
        )

    def qkv_linear_gqa(self, hidden_states):
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        # if not self.isGQA else
        # [sq, b, (np + num_kv_heads/p * 2) * hn]
        mixed_x_layer, _ = self.query_key_value(hidden_states)
        q_size = self.num_q_head_per_partition * self.head_size
        kv_size = self.num_kv_head_per_partition * self.head_size
        query_layer, key_layer, value_layer = mixed_x_layer.split(
            [q_size, kv_size, kv_size], dim=-1
        )
        query_layer = query_layer.view(
            tuple(query_layer.size()[:-1])
            + (self.num_attention_heads_per_partition, self.head_size)
        )
        key_layer = key_layer.view(
            tuple(key_layer.size()[:-1])
            + (self.num_kv_head_per_partition, self.head_size)
        )
        value_layer = value_layer.view(
            tuple(value_layer.size()[:-1])
            + (self.num_kv_head_per_partition, self.head_size)
        )
        return query_layer, key_layer, value_layer

    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_past=None,
        non_causal_attention_mask=None,
        position_ids=None,
    ):
        # Non causal attention mask is used for generating
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        # if not self.isGQA else
        # [sq, b, (np + num_kv_heads/p * 2) * hn]
        if not self.isGQA:
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = mpu.split_tensor_along_last_dim(
                mixed_x_layer, 3
            )
        else:
            (query_layer, key_layer, value_layer) = self.qkv_linear_gqa(hidden_states)

        if exists(self.rotary_emb):
            if exists(self.rotary_ndims):
                # partial rotary
                query_rot, query_pass = (
                    query_layer[..., : self.rotary_ndims],
                    query_layer[..., self.rotary_ndims :],
                )
                key_rot, key_pass = (
                    key_layer[..., : self.rotary_ndims],
                    key_layer[..., self.rotary_ndims :],
                )
            else:
                # full rotary
                query_rot, key_rot = query_layer, key_layer

            apply_rotary_fn = (
                apply_rotary_pos_emb_torch if self.bf16 else apply_rotary_pos_emb
            )

            seq_len = key_layer.shape[0]
            offset = 0
            if exists(layer_past) and layer_past.numel() > 0:
                offset = layer_past[0].shape[0]
                seq_len += offset
            if position_ids is None:
                cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
                # use position ids when left padding
                query_layer, key_layer = apply_rotary_fn(
                    query_rot, key_rot, cos, sin, offset=offset
                )
            else:
                # define for bf16 `ONLY` (torch version not jit)
                cos, sin = self.rotary_emb_legacy(value_layer, seq_len=seq_len)
                logging.debug("go to left padding rope")
                query_layer, key_layer = apply_rotary_pos_emb_torch_for_left_padding(
                    query_rot, key_rot, cos, sin, position_ids=position_ids
                )

            if exists(self.rotary_ndims):
                query_layer = torch.cat((query_layer, query_pass), dim=-1)
                key_layer = torch.cat((key_layer, key_pass), dim=-1)

        # ==================================
        # Cache key and value for inference
        # ==================================

        if exists(layer_past) and layer_past.numel() > 0:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat(
                (past_value.type_as(value_layer), value_layer), dim=0
            )

        if self.use_cache:
            present = torch.stack((key_layer, value_layer))

        if self.use_flash_attention:
            # logging.debug("USE FLASH ATTENTION")
            context_layer = self.flash_attention(
                query_layer, key_layer, value_layer, non_causal_attention_mask
            )

        elif not self.sparse:
            context_layer = self.attention(
                query_layer, key_layer, value_layer, layer_past, attention_mask
            )
        else:
            context_layer = self.sparse_attention(
                query_layer, key_layer, value_layer, attention_mask
            )

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        if self.use_cache:
            output = [output, present]

        return output, bias


class ParallelTransformerLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        use_cache=False,
    ):
        super().__init__()
        self.layer_number = layer_number

        norm, eps = get_norm(neox_args)

        # Layernorm on the input data.
        self.input_layernorm = norm(neox_args.hidden_size, eps=eps)
        self.use_cache = use_cache

        self.hidden_dropout = neox_args.hidden_dropout
        self.bias_dropout_fusion = neox_args.bias_dropout_fusion
        self.gpt_j_residual = neox_args.gpt_j_residual
        self.gpt_j_tied = neox_args.gpt_j_tied
        self.mlp_type = neox_args.mlp_type

        if self.gpt_j_residual:
            self.reduce = mpu.mappings.reduce_from_model_parallel_region

        # Self attention.
        self.attention = ParallelSelfAttention(
            neox_args=neox_args,
            attention_mask_func=attention_mask_func,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            rpe=rpe,
            use_cache=self.use_cache,
            rotary=rotary,
            parallel_output=self.gpt_j_residual,
        )

        # Layernorm on the output of the attention layer.
        # If GPT-J residuals are used, this is surpurfulous but leaving it in
        # leads to cleaner code
        self.post_attention_layernorm = norm(neox_args.hidden_size, eps=eps)

        # MLP
        if neox_args.mlp_type == "regular":
            self.mlp = ParallelMLP(
                neox_args=neox_args,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                parallel_output=self.gpt_j_residual,
            )
        elif neox_args.mlp_type == "llama":
            self.mlp = LLaMAParallelMLP(
                neox_args=neox_args,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                parallel_output=self.gpt_j_residual,
            )
        else:
            raise KeyError(neox_args.mlp_type)

        self.layer_past = None  # used to cache k/v pairs in inference

    def _get_bias_dropout(self):
        if self.bias_dropout_fusion:
            fn = (
                bias_dropout_add_fused_train
                if self.training
                else bias_dropout_add_fused_inference
            )
        else:
            fn = get_bias_dropout_add(self.training)
        return fn

    def forward(
        self,
        x,
        attention_mask,
        layer_past=None,
        non_causal_attention_mask=None,
        position_ids=None,
    ):
        layer_past = layer_past if layer_past is not None else self.layer_past
        bias_dropout_fn = self._get_bias_dropout()
        # x: [b, s, h]
        if self.gpt_j_residual:
            # pseudocode:
            # x = x + attn(ln(x)) + mlp(ln(x))
            # this means we can avoid doing the allreduce in the attn / mlp outputs
            # to save communication time (we can do a single allreduce after we add mlp / attn outputs).
            # due to a bug, the two layernorms are not tied in GPT-NeoX-20B. This is non-desirable, but
            # we preserve the functionality for backwards compatibility

            residual = x
            # applies the correct normalization depending on if the norms are tied
            if self.gpt_j_tied:
                x = self.input_layernorm(x)
                x1, x2 = x, x
            else:
                x1, x2 = self.input_layernorm(x), self.post_attention_layernorm(x)

            # attention operator
            attention_output, attention_bias = self.attention(
                x1,
                attention_mask,
                layer_past=layer_past,
                non_causal_attention_mask=non_causal_attention_mask,
                position_ids=position_ids,
            )
            if self.use_cache:
                attention_output, presents = attention_output
                self.layer_past = presents

            with torch.enable_grad():
                attention_output = bias_dropout_fn(
                    attention_output,
                    bias=attention_bias.expand_as(attention_output),
                    residual=None,
                    prob=self.hidden_dropout,
                )

            # mlp operator
            mlp_output, mlp_bias = self.mlp(x2)
            with torch.enable_grad():
                output = bias_dropout_fn(
                    mlp_output,
                    bias=mlp_bias.expand_as(mlp_output),
                    residual=attention_output,
                    prob=self.hidden_dropout,
                )

            # output = (x + attn(ln(x)) + mlp(ln(x))
            output = residual + self.reduce(output)
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))

            residual = x

            # x = x + attn(ln1(x))
            attention_output, attention_bias = self.attention(
                self.input_layernorm(x),
                attention_mask,
                layer_past=layer_past,
                non_causal_attention_mask=non_causal_attention_mask,
                position_ids=position_ids,
            )
            if self.use_cache:
                attention_output, presents = attention_output
                self.layer_past = presents
            with torch.enable_grad():
                if attention_bias is not None:
                    # Use special bias_dropout_fn if we have a bias term from the above attention layer
                    attention_output = bias_dropout_fn(
                        attention_output,
                        bias=attention_bias.expand_as(residual),
                        residual=residual,
                        prob=self.hidden_dropout,
                    )
                else:
                    # Otherwise just apply dropout + residual
                    attention_output = (
                        torch.nn.functional.dropout(
                            attention_output,
                            p=self.hidden_dropout,
                            training=self.training,
                        )
                        + residual
                    )

            # output = x + mlp(ln2(x))
            mlp_output, mlp_bias = self.mlp(
                self.post_attention_layernorm(attention_output)
            )

            with torch.enable_grad():
                if self.mlp_type == "llama":
                    # No dropout either
                    assert mlp_bias is None
                    output = mlp_output + attention_output
                else:
                    output = bias_dropout_fn(
                        mlp_output,
                        bias=mlp_bias.expand_as(attention_output),
                        residual=attention_output,
                        prob=self.hidden_dropout,
                    )

        if self.use_cache:
            return output, self.layer_past
        else:
            return output


class ParallelTransformerLayerPipe(ParallelTransformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline."""

    def forward(self, args):
        assert (
            len(args) == 2
        ), "ParallelTransformerLayerPipe expects 2 arguments - hidden_states and attention_mask"
        hidden_states, attention_mask = args
        # we are returning just [hidden_states, mask]
        return super().forward(hidden_states, attention_mask), attention_mask


class ParallelLinearPipe(ParallelLinear):
    """Another helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def forward(self, args):
        assert isinstance(
            args, torch.Tensor
        ), "ParallelLinearPipe expects a single argument - hidden_states"
        hidden_state = args
        logits, bias = super().forward(hidden_state)
        return logits


class NormPipe(nn.Module):
    """Just a helper class to pass presents through to the output when doing inference with a Pipe Parallel model"""

    def __init__(self, norm_class, hidden_size, eps):
        super().__init__()
        self.norm = norm_class(hidden_size, eps=eps)

    def forward(self, args):
        assert not isinstance(
            args, tuple
        ), "NormPipe should only receive a single tensor as input"
        return self.norm(args)


def parallel_lm_logits(input_, word_embeddings_weight, parallel_output, bias=None):
    """LM logits using word embedding weights."""
    # Parallel logits.
    input_parallel = mpu.copy_to_model_parallel_region(input_)

    # Matrix multiply.
    if bias is None:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight)
    else:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight, bias)

    # Gather if needed.
    if parallel_output:
        return logits_parallel

    return mpu.gather_from_model_parallel_region(logits_parallel)
