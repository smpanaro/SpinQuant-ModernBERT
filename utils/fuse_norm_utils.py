# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import typing
import torch

class RMSNorm(torch.nn.Module):
    # LLamaRMSNorm from `transformers`.
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

def equivalent_rms_norm(layernorm):
    assert not isinstance(layernorm, torch.nn.Identity)
    nm = RMSNorm(layernorm.weight.size(0), eps=layernorm.eps)
    return nm

def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias") and layernorm.bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_, layernorm.bias.double()
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)

def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)

def fuse_layer_norms(model):
    kwargs = {"model": model}

    # Embedding fusion
    for W in [model.model.embeddings.tok_embeddings]:
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    # Slight difference from Llama here (and in QuantizeLinear, `residual` flag)
    #  to account for the fact that the first layer norm is immediately after the
    # embeddings in ModernBERT, unlike Llama where it is immediately before the
    # first QKV projection.
    scale_matrix = torch.diag(model.model.embeddings.norm.weight).double()
    centering_matrix = torch.eye(model.config.hidden_size, dtype=torch.float64) - (1./model.config.hidden_size)
    model.model.layers[0].residual_adjustment_proj.weight.data = (centering_matrix @ scale_matrix).to(
        model.model.layers[0].attn.Wqkv.weight.dtype
    )

    fuse_ln_linear(
        model.model.embeddings.norm,
        [
            model.model.layers[0].attn.Wqkv,
        ],
    )
    model.model.embeddings.norm = equivalent_rms_norm(model.model.embeddings.norm)

    layers = [layer for layer in model.model.layers]

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        # fuse the input layernorms into the linear layers
        fuse_ln_linear(
            layer.mlp_norm, [layer.mlp.Wi]
        )
        layer.mlp_norm = equivalent_rms_norm(layer.mlp_norm)

        if not isinstance(layer.attn_norm, torch.nn.Identity):
            fuse_ln_linear(
                layer.attn_norm,
                [
                    layer.attn.Wqkv,
                ],
            )
            layer.attn_norm = equivalent_rms_norm(layer.attn_norm)

        bake_mean_into_linear(layer.attn.Wo)
        bake_mean_into_linear(layer.mlp.Wo)

    fuse_ln_linear(
        model.model.final_norm,
        [model.head.dense],
    )
    model.model.final_norm = equivalent_rms_norm(model.model.final_norm)
