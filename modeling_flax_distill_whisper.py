# COPIE FROM https://github.com/huggingface/distil-whisper/blob/914dcdf3919552d5a3826a9d5db99b059ddcc16e/training/flax/distil_whisper/modeling_flax_whisper.py
# and https://github.com/huggingface/distil-whisper/blob/914dcdf3919552d5a3826a9d5db99b059ddcc16e/training/flax/distil_whisper/layers.py

# coding=utf-8
# Copyright 2023 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
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
""" Flax whisper model."""

import random
from functools import partial
from typing import Dict, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.linen.partitioning import remat, scan_with_axes
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from transformers import WhisperConfig
from transformers.generation.flax_logits_process import (
    FlaxLogitsProcessor,
    FlaxLogitsProcessorList,
    FlaxWhisperTimeStampLogitsProcessor,
)
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPastAndCrossAttentions,
    FlaxCausalLMOutputWithCrossAttentions,
    FlaxSeq2SeqLMOutput,
    FlaxSeq2SeqModelOutput,
)
from transformers.modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

# Copyright 2022 The T5X Authors.
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

"""Dense attention classes and mask/weighting functions."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic

import dataclasses
import functools
import operator
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.linen.dtypes import promote_dtype
from jax import lax, random


# from flax.linen.partitioning import param_with_axes, with_sharding_constraint
param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint


# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision, lax.Precision]]
DotGeneralT = Callable[..., Array]
ConvGeneralDilatedT = Callable[..., Array]
PaddingLike = Union[str, int, Sequence[Union[int, Tuple[int, int]]]]
LaxPadding = Union[str, Sequence[Tuple[int, int]]]

# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]
InitializerAxis = Union[int, Tuple[int, ...]]
NdInitializer = Callable[[PRNGKey, Shape, DType, InitializerAxis, InitializerAxis], Array]

default_embed_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)


# ------------------------------------------------------------------------------
# Temporary inlined JAX N-d initializer code
# TODO(levskaya): remove once new JAX release is out.
# ------------------------------------------------------------------------------
def _compute_fans(shape: jax.core.NamedShape, in_axis=-2, out_axis=-1):
    """Inlined JAX `nn.initializer._compute_fans`."""
    if isinstance(in_axis, int):
        in_size = shape[in_axis]
    else:
        in_size = int(np.prod([shape[i] for i in in_axis]))
    if isinstance(out_axis, int):
        out_size = shape[out_axis]
    else:
        out_size = int(np.prod([shape[i] for i in out_axis]))
    receptive_field_size = shape.total / in_size / out_size
    fan_in = in_size * receptive_field_size
    fan_out = out_size * receptive_field_size
    return fan_in, fan_out


def variance_scaling(scale, mode, distribution, in_axis=-2, out_axis=-1, dtype=jnp.float_):
    """Inlined JAX `nn.initializer.variance_scaling`."""

    def init(key, shape, dtype=dtype):
        return jnp.zeros(shape, dtype=dtype)
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        shape = jax.core.as_named_shape(shape)
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError("invalid mode for variance scaling initializer: {}".format(mode))
        variance = jnp.array(scale / denominator, dtype=dtype)

        if distribution == "truncated_normal":
            # constant is stddev of standard normal truncated to (-2, 2)
            stddev = jnp.sqrt(variance) / jnp.array(0.87962566103423978, dtype)
            return random.truncated_normal(key, -2, 2, shape, dtype) * stddev
        elif distribution == "normal":
            return random.normal(key, shape, dtype) * jnp.sqrt(variance)
        elif distribution == "uniform":
            return random.uniform(key, shape, dtype, -1) * jnp.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer: {}".format(distribution))

    return init


# ------------------------------------------------------------------------------


def nd_dense_init(scale, mode, distribution):
    """Initializer with in_axis, out_axis set at call time."""

    def init_fn(key, shape, dtype, in_axis, out_axis):
        fn = variance_scaling(scale, mode, distribution, in_axis, out_axis)
        return fn(key, shape, dtype)

    return init_fn


def dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: DType = jnp.float32,
    float32_logits: bool = False,
):
    """Computes dot-product attention given query, key, and value.

    This is the core function for applying attention based on
    https://arxiv.org/abs/1706.03762. It calculates the attention weights given
    query and key and combines the values using the attention weights.

    Args:
      query: queries for calculating attention with shape of `[batch, q_length,
        num_heads, qk_depth_per_head]`.
      key: keys for calculating attention with shape of `[batch, kv_length,
        num_heads, qk_depth_per_head]`.
      value: values to be used in attention with shape of `[batch, kv_length,
        num_heads, v_depth_per_head]`.
      bias: bias for the attention weights. This should be broadcastable to the
        shape `[batch, num_heads, q_length, kv_length]` This can be used for
        incorporating causal masks, padding masks, proximity bias, etc.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      dtype: the dtype of the computation (default: float32)
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.

    Returns:
      Output of shape `[batch, length, num_heads, v_depth_per_head]`.
    """
    assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], "q, k, v batch dims must match."
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], "q, k, v num_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

    # Casting logits and softmax computation for float32 for model stability.
    if float32_logits:
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)

    # `attn_weights`: [batch, num_heads, q_length, kv_length]
    attn_weights = jnp.einsum("bqhd,bkhd->bhqk", query, key)

    # Apply attention bias: masking, dropout, proximity bias, etc.
    if bias is not None:
        attn_weights = attn_weights + bias.astype(attn_weights.dtype)

    # Normalize the attention weights across `kv_length` dimension.
    attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

    # Apply attention dropout.
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        # T5 broadcasts along the "length" dim, but unclear which one that
        # corresponds to in positional dimensions here, assuming query dim.
        dropout_shape = list(attn_weights.shape)
        dropout_shape[-2] = 1
        keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        keep = jnp.broadcast_to(keep, attn_weights.shape)
        multiplier = keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype)
        attn_weights = attn_weights * multiplier

    # Take the linear combination of `value`.
    return jnp.einsum("bhqk,bkhd->bqhd", attn_weights, value)


dynamic_vector_slice_in_dim = jax.vmap(lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))


class MultiHeadDotProductAttention(nn.Module):
    """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      head_dim: dimension of each head.
      dtype: the dtype of the computation.
      dropout_rate: dropout rate
      kernel_init: initializer for the kernel of the Dense layers.
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.
    """

    num_heads: int
    head_dim: int
    dtype: DType = jnp.float32
    dropout_rate: float = 0.0
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal")
    float32_logits: bool = False  # computes logits in float32 for stability.

    @nn.compact
    def __call__(
        self,
        inputs_q: Array,
        inputs_kv: Array,
        mask: Optional[Array] = None,
        bias: Optional[Array] = None,
        *,
        decode: bool = False,
        deterministic: bool = False,
    ) -> Array:
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        There are two modes: decoding and non-decoding (e.g., training). The mode is
        determined by `decode` argument. For decoding, this method is called twice,
        first to initialize the cache and then for an actual decoding process. The
        two calls are differentiated by the presence of 'cached_key' in the variable
        dict. In the cache initialization stage, the cache variables are initialized
        as zeros and will be filled in the subsequent decoding process.

        In the cache initialization call, `inputs_q` has a shape [batch, length,
        q_features] and `inputs_kv`: [batch, length, kv_features]. During the
        incremental decoding stage, query, key and value all have the shape [batch,
        1, qkv_features] corresponding to a single step.

        Args:
          inputs_q: input queries of shape `[batch, q_length, q_features]`.
          inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
          mask: attention mask of shape `[batch, num_heads, q_length, kv_length]`.
          bias: attention bias of shape `[batch, num_heads, q_length, kv_length]`.
          decode: Whether to prepare and use an autoregressive cache.
          deterministic: Disables dropout if set to True.

        Returns:
          output of shape `[batch, length, q_features]`.
        """
        projection = functools.partial(
            DenseGeneral,
            axis=-1,
            features=(self.num_heads, self.head_dim),
            kernel_axes=("embed", "heads", "kv"),
            dtype=self.dtype,
        )

        # NOTE: T5 does not explicitly rescale the attention logits by
        #       1/sqrt(depth_kq)!  This is folded into the initializers of the
        #       linear transformations, which is equivalent under Adafactor.
        depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)

        def query_init(*args):
            return self.kernel_init(*args) / depth_scaling

        # Project inputs_q to multi-headed q/k/v
        # dimensions are then [batch, length, num_heads, head_dim]
        query = projection(kernel_init=query_init, name="query")(inputs_q)
        key = projection(kernel_init=self.kernel_init, name="key")(inputs_kv)
        value = projection(kernel_init=self.kernel_init, name="value")(inputs_kv)

        query = with_sharding_constraint(query, ("batch", "length", "heads", "kv"))
        key = with_sharding_constraint(key, ("batch", "length", "heads", "kv"))
        value = with_sharding_constraint(value, ("batch", "length", "heads", "kv"))

        if decode:
            # Detect if we're initializing by absence of existing cache data.
            is_initialized = self.has_variable("cache", "cached_key")

            # The key and value have dimension [batch, length, num_heads, head_dim],
            # but we cache them as [batch, num_heads, head_dim, length] as a TPU
            # fusion optimization. This also enables the "scatter via one-hot
            # broadcast" trick, which means we do a one-hot broadcast instead of a
            # scatter/gather operations, resulting in a 3-4x speedup in practice.
            def swap_dims(x):
                return x[:-3] + tuple(x[i] for i in [-2, -1, -3])

            cached_key = self.variable("cache", "cached_key", jnp.zeros, swap_dims(key.shape), key.dtype)
            cached_value = self.variable("cache", "cached_value", jnp.zeros, swap_dims(value.shape), value.dtype)
            cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))
            if is_initialized:
                batch, num_heads, head_dim, length = cached_key.value.shape
                # During fast autoregressive decoding, we feed one position at a time,
                # and cache the keys and values step by step.
                # Sanity shape check of cached key against input query.
                expected_shape = (batch, 1, num_heads, head_dim)
                if expected_shape != query.shape:
                    raise ValueError(
                        "Autoregressive cache shape error, "
                        "expected query shape %s instead got %s." % (expected_shape, query.shape)
                    )

                # Create a OHE of the current index. NOTE: the index is increased below.
                cur_index = cache_index.value
                one_hot_indices = jax.nn.one_hot(cur_index, length, dtype=key.dtype)
                # In order to update the key, value caches with the current key and
                # value, we move the length axis to the back, similar to what we did for
                # the cached ones above.
                # Note these are currently the key and value of a single position, since
                # we feed one position at a time.
                one_token_key = jnp.moveaxis(key, -3, -1)
                one_token_value = jnp.moveaxis(value, -3, -1)
                # Update key, value caches with our new 1d spatial slices.
                # We implement an efficient scatter into the cache via one-hot
                # broadcast and addition.
                key = cached_key.value + one_token_key * one_hot_indices
                value = cached_value.value + one_token_value * one_hot_indices
                cached_key.value = key
                cached_value.value = value
                cache_index.value = cache_index.value + 1
                # Move the keys and values back to their original shapes.
                key = jnp.moveaxis(key, -1, -3)
                value = jnp.moveaxis(value, -1, -3)

                # Causal mask for cached decoder self-attention: our single query
                # position should only attend to those key positions that have already
                # been generated and cached, not the remaining zero elements.
                mask = combine_masks(
                    mask,
                    jnp.broadcast_to(
                        jnp.arange(length) <= cur_index,
                        # (1, 1, length) represent (head dim, query length, key length)
                        # query length is 1 because during decoding we deal with one
                        # index.
                        # The same mask is applied to all batch elements and heads.
                        (batch, 1, 1, length),
                    ),
                )

                # Grab the correct relative attention bias during decoding. This is
                # only required during single step decoding.
                if bias is not None:
                    # The bias is a full attention matrix, but during decoding we only
                    # have to take a slice of it.
                    # This is equivalent to bias[..., cur_index:cur_index+1, :].
                    bias = dynamic_vector_slice_in_dim(jnp.squeeze(bias, axis=0), jnp.reshape(cur_index, (-1)), 1, -2)

        # Convert the boolean attention mask to an attention bias.
        if mask is not None:
            # attention mask in the form of attention bias
            attention_bias = lax.select(
                mask > 0,
                jnp.full(mask.shape, 0.0).astype(self.dtype),
                jnp.full(mask.shape, -1e10).astype(self.dtype),
            )
        else:
            attention_bias = None

        # Add provided bias term (e.g. relative position embedding).
        if bias is not None:
            attention_bias = combine_biases(attention_bias, bias)

        dropout_rng = None
        if not deterministic and self.dropout_rate > 0.0:
            dropout_rng = self.make_rng("dropout")

        # Apply attention.
        x = dot_product_attention(
            query,
            key,
            value,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            dtype=self.dtype,
            float32_logits=self.float32_logits,
        )

        # Back to the original inputs dimensions.
        out = DenseGeneral(
            features=inputs_q.shape[-1],  # output dim is set to the input dim.
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            kernel_axes=("heads", "kv", "embed"),
            dtype=self.dtype,
            name="out",
        )(x)
        return out


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
    # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
    return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


def _canonicalize_tuple(x):
    if isinstance(x, Iterable):
        return tuple(x)
    else:
        return (x,)


# ------------------------------------------------------------------------------
# DenseGeneral for attention layers.
# ------------------------------------------------------------------------------
class DenseGeneral(nn.Module):
    """A linear transformation (without bias) with flexible axes.

    Attributes:
      features: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
    """

    features: Union[Iterable[int], int]
    axis: Union[Iterable[int], int] = -1
    dtype: DType = jnp.float32
    params_dtype: DType = jnp.float32
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal")
    kernel_axes: Tuple[str, ...] = ()
    use_bias: bool = True
    bias_init: Any = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along multiple dimensions.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        features = _canonicalize_tuple(self.features)
        axis = _canonicalize_tuple(self.axis)

        inputs = jnp.asarray(inputs, self.dtype)
        axis = _normalize_axes(axis, inputs.ndim)

        kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + features
        kernel_in_axis = np.arange(len(axis))
        kernel_out_axis = np.arange(len(axis), len(axis) + len(features))
        kernel = param_with_axes(
            "kernel",
            self.kernel_init,
            kernel_shape,
            self.params_dtype,
            kernel_in_axis,
            kernel_out_axis,
            axes=self.kernel_axes,
        )
        if self.use_bias:
            bias = param_with_axes(
                "bias",
                self.bias_init,
                features,
                self.params_dtype,
                axes=(self.kernel_axes[-1],),
            )
        kernel = jnp.asarray(kernel, self.dtype)

        contract_ind = tuple(range(0, len(axis)))
        y = lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))
        if self.use_bias:
            bias = jnp.asarray(bias, self.dtype)
            # y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
            y += jnp.reshape(bias, (1,) * (len(features) - y.ndim) + bias.shape[:])
        return y


def _convert_to_activation_function(fn_or_string: Union[str, Callable]) -> Callable:
    """Convert a string to an activation function."""
    if fn_or_string == "linear":
        return lambda x: x
    elif isinstance(fn_or_string, str):
        return getattr(nn, fn_or_string)
    elif callable(fn_or_string):
        return fn_or_string
    else:
        raise ValueError("don't know how to convert %s to an activation function" % (fn_or_string,))


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
      intermediate_dim: Shared dimension of hidden layers.
      activations: Type of activations for each layer.  Each element is either
        'linear', a string function name in flax.linen, or a function.
      kernel_init: Kernel function, passed to the dense layers.
      deterministic: Whether the dropout layers should be deterministic.
      intermediate_dropout_rate: Dropout rate used after the intermediate layers.
      dtype: Type for the dense layer.
    """

    intermediate_dim: int = 2048
    activations: Sequence[Union[str, Callable]] = ("relu",)
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal")
    intermediate_dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
        """Applies Transformer MlpBlock module."""
        # Iterate over specified MLP input activation functions.
        # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
        activations = []
        for idx, act_fn in enumerate(self.activations):
            dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
            x = DenseGeneral(
                self.intermediate_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                kernel_axes=("embed", "mlp"),
                name=dense_name,
            )(inputs)
            x = _convert_to_activation_function(act_fn)(x)
            activations.append(x)

        # Take elementwise product of above intermediate activations.
        x = functools.reduce(operator.mul, activations)
        # Apply dropout and final dense output projection.
        x = nn.Dropout(rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic
        )  # Broadcast along length.
        x = with_sharding_constraint(x, ("batch", "length", "mlp"))
        output = DenseGeneral(
            inputs.shape[-1],
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_axes=("mlp", "embed"),
            name="wo",
        )(x)
        return output


class Embed(nn.Module):
    """A parameterized function from integers [0, n) to d-dimensional vectors.

    Attributes:
      num_embeddings: number of embeddings.
      features: number of feature dimensions for each embedding.
      dtype: the dtype of the embedding vectors (default: float32).
      embedding_init: embedding initializer.
      one_hot: performs the gather with a one-hot contraction rather than a true
        gather. This is currently needed for SPMD partitioning.
    """

    num_embeddings: int
    features: int
    cast_input_dtype: Optional[DType] = None
    dtype: DType = jnp.float32
    params_dtype: DType = jnp.float32
    attend_dtype: Optional[DType] = None
    embedding_init: Initializer = default_embed_init
    one_hot: bool = True
    embedding: Array = dataclasses.field(init=False)

    def setup(self):
        self.embedding = param_with_axes(
            "embedding",
            self.embedding_init,
            (self.num_embeddings, self.features),
            self.params_dtype,
            axes=("vocab", "embed"),
        )

    def __call__(self, inputs: Array) -> Array:
        """Embeds the inputs along the last dimension.

        Args:
          inputs: input data, all dimensions are considered batch dimensions.

        Returns:
          Output which is embedded input data.  The output shape follows the input,
          with an additional `features` dimension appended.
        """
        if self.cast_input_dtype:
            inputs = inputs.astype(self.cast_input_dtype)
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Input type must be an integer or unsigned integer.")
        if self.one_hot:
            iota = lax.iota(jnp.int32, self.num_embeddings)
            one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
            output = jnp.dot(one_hot, jnp.asarray(self.embedding, self.dtype))
        else:
            output = jnp.asarray(self.embedding, self.dtype)[inputs]
            output = with_sharding_constraint(output, ("batch", "length", "embed"))
        return output

    def attend(self, query: Array) -> Array:
        """Attend over the embedding using a query array.

        Args:
          query: array with last dimension equal the feature depth `features` of the
            embedding.

        Returns:
          An array with final dim `num_embeddings` corresponding to the batched
          inner-product of the array of query vectors against each embedding.
          Commonly used for weight-sharing between embeddings and logit transform
          in NLP models.
        """
        dtype = self.attend_dtype if self.attend_dtype is not None else self.dtype
        return jnp.dot(query, jnp.asarray(self.embedding, dtype).T)


class RelativePositionBiases(nn.Module):
    """Adds T5-style relative positional embeddings to the attention logits.

    Attributes:
      num_buckets: Number of buckets to bucket distances between key and query
        positions into.
      max_distance: Maximum distance before everything is lumped into the last
        distance bucket.
      num_heads: Number of heads in the attention layer. Each head will get a
        different relative position weighting.
      dtype: Type of arrays through this module.
      embedding_init: initializer for relative embedding table.
    """

    num_buckets: int
    max_distance: int
    num_heads: int
    dtype: Any
    embedding_init: Callable[..., Array] = nn.linear.default_embed_init

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """Translate relative position to a bucket number for relative attention.

        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger
        buckets for larger absolute relative_positions.  All relative
        positions >=max_distance  map to the same bucket.  All relative
        positions <=-max_distance map to the same bucket.  This should allow for
        more graceful generalization to longer sequences than the model has been
        trained on.

        Args:
          relative_position: an int32 array
          bidirectional: a boolean - whether the attention is bidirectional
          num_buckets: an integer
          max_distance: an integer

        Returns:
          a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).astype(np.int32) * num_buckets
            n = np.abs(n)
        else:
            n = np.maximum(n, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            np.log(n.astype(np.float32) / max_exact + np.finfo(np.float32).eps)
            / np.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(np.int32)
        val_if_large = np.minimum(val_if_large, num_buckets - 1)
        ret += np.where(is_small, n, val_if_large)
        return ret

    @nn.compact
    def __call__(self, qlen, klen, bidirectional=True):
        """Produce relative position embedding attention biases.

        Args:
          qlen: attention query length.
          klen: attention key length.
          bidirectional: whether to allow positive memory-query relative position
            embeddings.

        Returns:
          output: `(1, len, q_len, k_len)` attention bias
        """
        # TODO(levskaya): should we be computing this w. numpy as a program
        # constant?
        context_position = np.arange(qlen, dtype=jnp.int32)[:, None]
        memory_position = np.arange(klen, dtype=jnp.int32)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        relative_attention_bias = param_with_axes(
            "rel_embedding",
            self.embedding_init,
            (self.num_heads, self.num_buckets),
            jnp.float32,
            axes=("heads", "relpos_buckets"),
        )

        relative_attention_bias = jnp.asarray(relative_attention_bias, self.dtype)
        # Instead of using a slow gather, we create a leading-dimension one-hot
        # array from rp_bucket and use it to perform the gather-equivalent via a
        # contraction, i.e.:
        # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
        # This is equivalent to relative_attention_bias[:, rp_bucket]
        bcast_iota = lax.broadcasted_iota(jnp.int32, (self.num_buckets, 1, 1), 0)
        rp_bucket_one_hot = jnp.array(rp_bucket[jnp.newaxis, ...] == bcast_iota, dtype=self.dtype)
        # --> shape (qlen, klen, num_heads)
        values = lax.dot_general(
            relative_attention_bias,
            rp_bucket_one_hot,
            (((1,), (0,)), ((), ())),  # rhs, lhs contracting dims
        )  # no batched dims
        # Add a singleton batch dimension.
        # --> shape (1, num_heads, qlen, klen)
        return values[jnp.newaxis, ...]


# ------------------------------------------------------------------------------
# T5 Layernorm - no subtraction of mean or bias.
# ------------------------------------------------------------------------------
# class LayerNorm(nn.Module):
#   """T5 Layer normalization operating on the last axis of the input data."""
#   epsilon: float = 1e-6
#   dtype: Any = jnp.float32
#   scale_init: Initializer = nn.initializers.ones

#   @nn.compact
#   def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#     """Applies layer normalization on the input."""
#     x = jnp.asarray(x, jnp.float32)
#     features = x.shape[-1]
#     mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
#     y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
#     scale = param_with_axes(
#         'scale', self.scale_init, (features,), jnp.float32, axes=('embed',))

#     scale = jnp.asarray(scale, self.dtype)
#     return y * scale


class LayerNorm(nn.Module):
    """Layer normalization (https://arxiv.org/abs/1607.06450).
    Operates on the last axis of the input data.
    It normalizes the activations of the layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within
    each example close to 0 and the activation standard deviation close to 1.
    Attributes:
      epsilon: A small float added to variance to avoid dividing by zero.
      dtype: the dtype of the computation (default: float32).
      use_bias:  If True, bias (beta) is added.
      use_scale: If True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: Initializer for bias, by default, zero.
      scale_init: Initializer for scale, by default, one.
    """

    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    params_dtype: DType = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Any], Array] = nn.initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Any], Array] = nn.initializers.ones

    @nn.compact
    def __call__(self, x):
        """Applies layer normalization on the input.
        Args:
          x: the inputs
        Returns:
          Normalized inputs (the same shape as inputs).
        """
        x = jnp.asarray(x, jnp.float32)
        features = x.shape[-1]
        mean = jnp.mean(x, axis=-1, keepdims=True)
        mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
        var = mean2 - lax.square(mean)
        mul = lax.rsqrt(var + self.epsilon)
        if self.use_scale:
            scale = param_with_axes(
                "scale",
                self.scale_init,
                (features,),
                self.params_dtype,
                axes=("embed",),
            )
            mul = mul * jnp.asarray(scale, self.dtype)
        y = (x - mean) * mul
        if self.use_bias:
            bias = param_with_axes("bias", self.bias_init, (features,), self.params_dtype, axes=("embed",))
            y = y + jnp.asarray(bias, self.dtype)
        return jnp.asarray(y, self.dtype)


# ------------------------------------------------------------------------------
# Mask-making utility functions.
# ------------------------------------------------------------------------------
def make_attention_mask(
    query_input: Array,
    key_input: Array,
    pairwise_fn: Callable = jnp.multiply,
    extra_batch_dims: int = 0,
    dtype: DType = jnp.float32,
) -> Array:
    """Mask-making helper for attention weights.

    In case of 1d inputs (i.e., `[batch, len_q]`, `[batch, len_kv]`, the
    attention weights will be `[batch, heads, len_q, len_kv]` and this
    function will produce `[batch, 1, len_q, len_kv]`.

    Args:
      query_input: a batched, flat input of query_length size
      key_input: a batched, flat input of key_length size
      pairwise_fn: broadcasting elementwise comparison function
      extra_batch_dims: number of extra batch dims to add singleton axes for, none
        by default
      dtype: mask return dtype

    Returns:
      A `[batch, 1, len_q, len_kv]` shaped mask for 1d attention.
    """
    # [batch, len_q, len_kv]
    mask = pairwise_fn(
        # [batch, len_q] -> [batch, len_q, 1]
        jnp.expand_dims(query_input, axis=-1),
        # [batch, len_q] -> [batch, 1, len_kv]
        jnp.expand_dims(key_input, axis=-2),
    )

    # [batch, 1, len_q, len_kv]. This creates the head dim.
    mask = jnp.expand_dims(mask, axis=-3)
    mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
    return mask.astype(dtype)


def make_causal_mask(x: Array, extra_batch_dims: int = 0, dtype: DType = jnp.float32) -> Array:
    """Make a causal mask for self-attention.

    In case of 1d inputs (i.e., `[batch, len]`, the self-attention weights
    will be `[batch, heads, len, len]` and this function will produce a
    causal mask of shape `[batch, 1, len, len]`.

    Note that a causal mask does not depend on the values of x; it only depends on
    the shape. If x has padding elements, they will not be treated in a special
    manner.

    Args:
      x: input array of shape `[batch, len]`
      extra_batch_dims: number of batch dims to add singleton axes for, none by
        default
      dtype: mask return dtype

    Returns:
      A `[batch, 1, len, len]` shaped causal mask for 1d attention.
    """
    idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
    return make_attention_mask(idxs, idxs, jnp.greater_equal, extra_batch_dims=extra_batch_dims, dtype=dtype)


def combine_masks(*masks: Optional[Array], dtype: DType = jnp.float32):
    """Combine attention masks.

    Args:
      *masks: set of attention mask arguments to combine, some can be None.
      dtype: final mask dtype

    Returns:
      Combined mask, reduced by logical and, returns None if no masks given.
    """
    masks = [m for m in masks if m is not None]
    if not masks:
        return None
    assert all(
        (x.ndim == masks[0].ndim for x in masks)
    ), f"masks must have same rank: {tuple((x.ndim for x in masks))}"
    mask, *other_masks = masks
    for other_mask in other_masks:
        mask = jnp.logical_and(mask, other_mask)
    return mask.astype(dtype)


def combine_biases(*masks: Optional[Array]):
    """Combine attention biases.

    Args:
      *masks: set of attention bias arguments to combine, some can be None.

    Returns:
      Combined mask, reduced by summation, returns None if no masks given.
    """
    masks = [m for m in masks if m is not None]
    if not masks:
        return None
    assert all(
        (x.ndim == masks[0].ndim for x in masks)
    ), f"masks must have same rank: {tuple((x.ndim for x in masks))}"
    mask, *other_masks = masks
    for other_mask in other_masks:
        mask = mask + other_mask
    return mask


def make_decoder_mask(
    decoder_target_tokens: Array,
    dtype: DType,
    decoder_causal_attention: Optional[Array] = None,
    decoder_segment_ids: Optional[Array] = None,
) -> Array:
    """Compute the self-attention mask for a decoder.

    Decoder mask is formed by combining a causal mask, a padding mask and an
    optional packing mask. If decoder_causal_attention is passed, it makes the
    masking non-causal for positions that have value of 1.

    A prefix LM is applied to a dataset which has a notion of "inputs" and
    "targets", e.g., a machine translation task. The inputs and targets are
    concatenated to form a new target. `decoder_target_tokens` is the concatenated
    decoder output tokens.

    The "inputs" portion of the concatenated sequence can attend to other "inputs"
    tokens even for those at a later time steps. In order to control this
    behavior, `decoder_causal_attention` is necessary. This is a binary mask with
    a value of 1 indicating that the position belonged to "inputs" portion of the
    original dataset.

    Example:

      Suppose we have a dataset with two examples.

      ds = [{"inputs": [6, 7], "targets": [8]},
            {"inputs": [3, 4], "targets": [5]}]

      After the data preprocessing with packing, the two examples are packed into
      one example with the following three fields (some fields are skipped for
      simplicity).

         decoder_target_tokens = [[6, 7, 8, 3, 4, 5, 0]]
           decoder_segment_ids = [[1, 1, 1, 2, 2, 2, 0]]
      decoder_causal_attention = [[1, 1, 0, 1, 1, 0, 0]]

      where each array has [batch, length] shape with batch size being 1. Then,
      this function computes the following mask.

                        mask = [[[[1, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 1, 0, 0],
                                  [0, 0, 0, 1, 1, 0, 0],
                                  [0, 0, 0, 1, 1, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 0]]]]

      mask[b, 1, :, :] represents the mask for the example `b` in the batch.
      Because mask is for a self-attention layer, the mask's shape is a square of
      shape [query length, key length].

      mask[b, 1, i, j] = 1 means that the query token at position i can attend to
      the key token at position j.

    Args:
      decoder_target_tokens: decoder output tokens. [batch, length]
      dtype: dtype of the output mask.
      decoder_causal_attention: a binary mask indicating which position should
        only attend to earlier positions in the sequence. Others will attend
        bidirectionally. [batch, length]
      decoder_segment_ids: decoder segmentation info for packed examples. [batch,
        length]

    Returns:
      the combined decoder mask.
    """
    masks = []
    # The same mask is applied to all attention heads. So the head dimension is 1,
    # i.e., the mask will be broadcast along the heads dim.
    # [batch, 1, length, length]
    causal_mask = make_causal_mask(decoder_target_tokens, dtype=dtype)

    # Positions with value 1 in `decoder_causal_attneition` can attend
    # bidirectionally.
    if decoder_causal_attention is not None:
        # [batch, 1, length, length]
        inputs_mask = make_attention_mask(
            decoder_causal_attention,
            decoder_causal_attention,
            jnp.logical_and,
            dtype=dtype,
        )
        masks.append(jnp.logical_or(causal_mask, inputs_mask).astype(dtype))
    else:
        masks.append(causal_mask)

    # Padding mask.
    masks.append(make_attention_mask(decoder_target_tokens > 0, decoder_target_tokens > 0, dtype=dtype))

    # Packing mask
    if decoder_segment_ids is not None:
        masks.append(make_attention_mask(decoder_segment_ids, decoder_segment_ids, jnp.equal, dtype=dtype))

    return combine_masks(*masks, dtype=dtype)


def canonicalize_padding(padding: PaddingLike, rank: int) -> LaxPadding:
    """ "Canonicalizes conv padding to a jax.lax supported format."""
    if isinstance(padding, str):
        return padding
    if isinstance(padding, int):
        return [(padding, padding)] * rank
    if isinstance(padding, Sequence) and len(padding) == rank:
        new_pad = []
        for p in padding:
            if isinstance(p, int):
                new_pad.append((p, p))
            elif isinstance(p, tuple) and len(p) == 2:
                new_pad.append(p)
            else:
                break
        if len(new_pad) == rank:
            return new_pad
    raise ValueError(
        f"Invalid padding format: {padding}, should be str, int,"
        f" or a sequence of len {rank} where each element is an"
        " int or pair of ints."
    )


def _conv_dimension_numbers(input_shape):
    """Computes the dimension numbers based on the input shape."""
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


class _Conv(nn.Module):
    """Convolution Module wrapping `lax.conv_general_dilated[_local]`.

    Attributes:
      features: number of convolution filters.
      kernel_size: shape of the convolutional kernel. For 1D convolution,
        the kernel size can be passed as an integer. For all other cases, it must
        be a sequence of integers.
      strides: an integer or a sequence of `n` integers, representing the
        inter-window strides (default: 1).
      padding: either the string `'SAME'`, the string `'VALID'`, the string
        `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
        high)` integer pairs that give the padding to apply before and after each
        spatial dimension. A single int is interpeted as applying the same padding
        in all dims and passign a single int in a sequence causes the same padding
        to be used on both sides. `'CAUSAL'` padding for a 1D convolution will
        left-pad the convolution axis, resulting in same-sized output.
      input_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of `inputs`
        (default: 1). Convolution with input dilation `d` is equivalent to
        transposed convolution with stride `d`.
      kernel_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel (default: 1). Convolution with kernel dilation
        is also known as 'atrous convolution'.
      feature_group_count: integer, default 1. If specified divides the input
        features into groups.
      use_bias: whether to add a bias to the output (default: True).
      mask: Optional mask for the weights during masked convolution. The mask must
            be the same shape as the convolution weight matrix.
      dtype: the dtype of the computation (default: infer from input and params).
      params_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the convolutional kernel.
      bias_init: initializer for the bias.
    """

    features: int
    kernel_size: Sequence[int]
    strides: Union[None, int, Sequence[int]] = 1
    padding: PaddingLike = "SAME"
    input_dilation: Union[None, int, Sequence[int]] = 1
    kernel_dilation: Union[None, int, Sequence[int]] = 1
    feature_group_count: int = 1
    use_bias: bool = True
    mask: Optional[Array] = None
    dtype: Optional[DType] = None
    params_dtype: DType = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, DType], Array] = nn.initializers.lecun_normal()
    bias_init: Callable[[PRNGKey, Shape, DType], Array] = nn.initializers.zeros
    conv_general_dilated: ConvGeneralDilatedT = lax.conv_general_dilated
    kernel_axes: Tuple[str, ...] = ()

    @property
    def shared_weights(self) -> bool:  # type: ignore
        """Defines whether weights are shared or not between different pixels.

        Returns:
          `True` to use shared weights in convolution (regular convolution).
          `False` to use different weights at different pixels, a.k.a.
          "locally connected layer", "unshared convolution", or "local convolution".

        """
        ...

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a (potentially unshared) convolution to the inputs.

        Args:
          inputs: input data with dimensions (*batch_dims, spatial_dims...,
            features). This is the channels-last convention, i.e. NHWC for a 2d
            convolution and NDHWC for a 3D convolution. Note: this is different from
            the input convention used by `lax.conv_general_dilated`, which puts the
            spatial dimensions last.
            Note: If the input has more than 1 batch dimension, all batch dimensions
            are flattened into a single dimension for the convolution and restored
            before returning.  In some cases directly vmap'ing the layer may yield
            better performance than this default flattening approach.  If the input
            lacks a batch dimension it will be added for the convolution and removed
            n return, an allowance made to enable writing single-example code.

        Returns:
          The convolved data.
        """

        if isinstance(self.kernel_size, int):
            raise TypeError(
                "Expected Conv kernel_size to be a"
                " tuple/list of integers (eg.: [3, 3]) but got"
                f" {self.kernel_size}."
            )
        else:
            kernel_size = tuple(self.kernel_size)

        def maybe_broadcast(x: Optional[Union[int, Sequence[int]]]) -> Tuple[int, ...]:
            if x is None:
                # backward compatibility with using None as sentinel for
                # broadcast 1
                x = 1
            if isinstance(x, int):
                return (x,) * len(kernel_size)
            return tuple(x)

        # Combine all input batch dimensions into a single leading batch axis.
        num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
        if num_batch_dimensions != 1:
            input_batch_shape = inputs.shape[:num_batch_dimensions]
            total_batch_size = int(np.prod(input_batch_shape))
            flat_input_shape = (total_batch_size,) + inputs.shape[num_batch_dimensions:]
            inputs = jnp.reshape(inputs, flat_input_shape)

        # self.strides or (1,) * (inputs.ndim - 2)
        strides = maybe_broadcast(self.strides)
        input_dilation = maybe_broadcast(self.input_dilation)
        kernel_dilation = maybe_broadcast(self.kernel_dilation)

        padding_lax = canonicalize_padding(self.padding, len(kernel_size))
        if padding_lax == "CIRCULAR":
            kernel_size_dilated = [(k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)]
            zero_pad: List[Tuple[int, int]] = [(0, 0)]
            pads = zero_pad + [((k - 1) // 2, k // 2) for k in kernel_size_dilated] + [(0, 0)]
            inputs = jnp.pad(inputs, pads, mode="wrap")
            padding_lax = "VALID"
        elif padding_lax == "CAUSAL":
            if len(kernel_size) != 1:
                raise ValueError("Causal padding is only implemented for 1D convolutions.")
            left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
            pads = [(0, 0), (left_pad, 0), (0, 0)]
            inputs = jnp.pad(inputs, pads)
            padding_lax = "VALID"

        dimension_numbers = _conv_dimension_numbers(inputs.shape)
        in_features = jnp.shape(inputs)[-1]

        if self.shared_weights:
            # One shared convolutional kernel for all pixels in the output.
            assert in_features % self.feature_group_count == 0
            kernel_shape = kernel_size + (
                in_features // self.feature_group_count,
                self.features,
            )

        else:
            if self.feature_group_count != 1:
                raise NotImplementedError(
                    "`lax.conv_general_dilated_local` does not support "
                    f"`feature_group_count != 1`, got `{self.feature_group_count}`."
                )

            # Need to know the spatial output shape of a standard convolution to
            # create the unshared convolution kernel.
            conv_output_shape = jax.eval_shape(
                lambda lhs, rhs: self.conv_general_dilated(  # pylint: disable=g-long-lambda
                    lhs=lhs,
                    rhs=rhs,
                    window_strides=strides,
                    padding=padding_lax,
                    dimension_numbers=dimension_numbers,
                    lhs_dilation=input_dilation,
                    rhs_dilation=kernel_dilation,
                ),
                inputs,
                jax.ShapedArray(kernel_size + (in_features, self.features), inputs.dtype),
            ).shape

            # One (unshared) convolutional kernel per each pixel in the output.
            kernel_shape = conv_output_shape[1:-1] + (
                np.prod(kernel_size) * in_features,
                self.features,
            )

        if self.mask is not None and self.mask.shape != kernel_shape:
            raise ValueError(
                "Mask needs to have the same shape as weights. " f"Shapes are: {self.mask.shape}, {kernel_shape}"
            )

        kernel = param_with_axes(
            "kernel",
            self.kernel_init,
            kernel_shape,
            self.params_dtype,
            axes=self.kernel_axes,
        )

        if self.mask is not None:
            kernel *= self.mask

        if self.use_bias:
            if self.shared_weights:
                # One bias weight per output channel, shared between pixels.
                bias_shape = (self.features,)
            else:
                # One bias weight per output entry, unshared betwen pixels.
                bias_shape = conv_output_shape[1:]

            bias = param_with_axes(
                "bias",
                self.bias_init,
                bias_shape,
                self.params_dtype,
                axes=(self.kernel_axes[-1],),
            )
        else:
            bias = None

        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
        if self.shared_weights:
            y = self.conv_general_dilated(
                inputs,
                kernel,
                strides,
                padding_lax,
                lhs_dilation=input_dilation,
                rhs_dilation=kernel_dilation,
                dimension_numbers=dimension_numbers,
                feature_group_count=self.feature_group_count,
                precision=self.precision,
            )
        else:
            y = lax.conv_general_dilated_local(
                lhs=inputs,
                rhs=kernel,
                window_strides=strides,
                padding=padding_lax,
                filter_shape=kernel_size,
                lhs_dilation=input_dilation,
                rhs_dilation=kernel_dilation,
                dimension_numbers=dimension_numbers,
                precision=self.precision,
            )

        if self.use_bias:
            bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
            y += bias

        if num_batch_dimensions != 1:
            output_shape = input_batch_shape + y.shape[1:]
            y = jnp.reshape(y, output_shape)
        return y


class Conv(_Conv):
    """Convolution Module wrapping `lax.conv_general_dilated`.

    Attributes:
      features: number of convolution filters.
      kernel_size: shape of the convolutional kernel. For 1D convolution,
        the kernel size can be passed as an integer. For all other cases, it must
        be a sequence of integers.
      strides: an integer or a sequence of `n` integers, representing the
        inter-window strides (default: 1).
      padding: either the string `'SAME'`, the string `'VALID'`, the string
        `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
        high)` integer pairs that give the padding to apply before and after each
        spatial dimension. A single int is interpeted as applying the same padding
        in all dims and passign a single int in a sequence causes the same padding
        to be used on both sides. `'CAUSAL'` padding for a 1D convolution will
        left-pad the convolution axis, resulting in same-sized output.
      input_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of `inputs`
        (default: 1). Convolution with input dilation `d` is equivalent to
        transposed convolution with stride `d`.
      kernel_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel (default: 1). Convolution with kernel dilation
        is also known as 'atrous convolution'.
      feature_group_count: integer, default 1. If specified divides the input
        features into groups.
      use_bias: whether to add a bias to the output (default: True).
      mask: Optional mask for the weights during masked convolution. The mask must
            be the same shape as the convolution weight matrix.
      dtype: the dtype of the computation (default: infer from input and params).
      params_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the convolutional kernel.
      bias_init: initializer for the bias.
    """

    @property
    def shared_weights(self) -> bool:
        return True

logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "openai/whisper-tiny"
_CONFIG_FOR_DOC = "WhisperConfig"


WHISPER_START_DOCSTRING = r"""
    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.) This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.
    Finally, this model supports inherent JAX features such as:
    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`WhisperConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs). This can be used to enable mixed-precision training or half-precision
            inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.** If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`]
            and [`~FlaxPreTrainedModel.to_bf16`].
"""

WHISPER_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`numpy.ndarray` of shape `(batch_size, feature_size, sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`WhisperFeatureExtractor`] should be used for extracting the features, padding and conversion into a
            tensor of type `numpy.ndarray`. See [`~WhisperFeatureExtractor.__call__`]
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Whisper does not support masking of the `input_features`, this argument is preserved for compatibility, but
            is not used. By default the silence in the input log mel spectrogram are ignored.
        decoder_input_ids (`numpy.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary. Indices can be obtained using
            [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.
            [What are decoder input IDs?](../glossary#decoder-input-ids) Whisper uses the `decoder_start_token_id` as
            the starting token for `decoder_input_ids` generation.
        decoder_attention_mask (`numpy.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default. If you want to change padding behavior, you should modify to your needs. See diagram 1
            in [the paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Whisper does not use `position_ids` in the encoder as `input_features` is always the same size and doesn't
            use masking, but this argument is preserved for compatibility. By default the silence in the input log mel
            spectrogram are ignored.
        decoder_position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
            range `[0, config.max_position_embeddings - 1]`.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

WHISPER_ENCODE_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`numpy.ndarray` of shape `(batch_size, feature_size, sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`WhisperFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
            tensor of type `numpy.ndarray`. See [`~WhisperFeatureExtractor.__call__`].
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Whisper does not support masking of the `input_features`, this argument is preserved for compatibility, but
            is not used. By default the silence in the input log mel spectrogram are ignored.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

WHISPER_DECODE_INPUTS_DOCSTRING = r"""
    Args:
        decoder_input_ids (`numpy.ndarray` of shape `(batch_size, target_sequence_length)`):
            Indices of decoder input sequence tokens in the vocabulary. Indices can be obtained using
            [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.
            [What are decoder input IDs?](../glossary#decoder-input-ids)
        encoder_outputs (`tuple(tuple(numpy.ndarray)`):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        encoder_attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
           Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
            but it is not used. By default the silence in the input log mel spectrogram are ignored.
        decoder_attention_mask (`numpy.ndarray` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default. If you want to change padding behavior, you should modify to your needs. See diagram 1
            in [the paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
        decoder_position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
            range `[0, config.max_position_embeddings - 1]`.
        past_key_values (`Dict[str, numpy.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class FlaxStaticForceTokensLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] that takes a list of pairs of integers which indicates a mapping from generation indices to
    token indices that will be forced before sampling. The processor will set their log probs to 0 and all other tokens
    to `-inf` so that they are sampled at their corresponding index. This is a static version of the `transformers` logit
    processor [`FlaxForceTokensLogitsProcessor`] that is compatible with sharded forced tokens.

    Args:
        force_token_map (`list`):
            Map giving token ids and indices where they will be forced to be sampled.
    """

    def __init__(self, force_token_map):
        # The generic `transformers` logit processor builds `force_token_array` as a dictionary - this is not a valid
        # JAX type, and so we switch to using a JAX array instead
        force_token_map = jnp.array(force_token_map)
        # Converts the array of format [[index, token]] containing the tokens to be forced to an array, where the
        # index of the array corresponds to the index of the token to be forced. For XLA compatibility,
        # indexes without forced tokens will have a negative value. Note that the last token we ever need to force in
        # Whisper is at position 3, so we only construct an array up to this index. The native version constructs a tensor
        # dynamically according to the length of the `force_token_map`. Array shapes need to be concrete for XLA compatibility,
        # so this is not permitted here.
        force_token_array = jnp.ones(3, dtype=jnp.int32) * -1
        for index, token in force_token_map:
            force_token_array = force_token_array.at[index].set(token)
        self.force_token_array = jnp.int32(force_token_array)

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        def _force_token(generation_idx):
            batch_size = scores.shape[0]
            current_token = self.force_token_array[generation_idx]

            new_scores = jnp.ones_like(scores, dtype=scores.dtype) * -float("inf")
            updates = jnp.zeros((batch_size, 1), dtype=scores.dtype)
            new_scores = lax.dynamic_update_slice(new_scores, updates, (0, current_token))
            return new_scores

        scores = lax.cond(
            cur_len >= self.force_token_array.shape[0],
            # If the current length is geq than the length of force_token_array, the processor does nothing.
            lambda: scores,
            # Otherwise, it may force a certain token.
            lambda: lax.cond(
                self.force_token_array[cur_len] >= 0,
                # Only valid (positive) tokens are forced
                lambda: _force_token(cur_len),
                # Otherwise, the processor does nothing.
                lambda: scores,
            ),
        )
        return scores


class FlaxWhisperAttention(nn.Module):
    config: WhisperConfig
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    causal: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32
    params_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                "embed_dim must be divisible by num_heads (got `embed_dim`:"
                f" {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        dense = partial(
            DenseGeneral,
            self.embed_dim,
            axis=-1,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
            kernel_axes=("embed", "joined_kv"),
        )

        self.q_proj = dense(use_bias=self.bias)
        self.k_proj = dense(use_bias=False)
        self.v_proj = dense(use_bias=self.bias)

        self.out_proj = DenseGeneral(
            self.embed_dim,
            axis=-1,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
            kernel_axes=("joined_kv", "embed"),
            use_bias=self.bias,
        )

        if self.causal:
            self.causal_mask = make_causal_mask(
                jnp.ones((1, self.config.max_target_positions), dtype="bool"),
                dtype="bool",
            )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        is_cross_attention = key_value_states is not None
        batch_size = hidden_states.shape[0]

        query_states = self.q_proj(hidden_states)

        if is_cross_attention:
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        query_states = with_sharding_constraint(query_states, ("batch", "length", "heads", "kv"))
        key_states = with_sharding_constraint(key_states, ("batch", "length", "heads", "kv"))
        value_states = with_sharding_constraint(value_states, ("batch", "length", "heads", "kv"))

        if self.causal:
            query_length, key_length = query_states.shape[1], key_states.shape[1]
            if self.has_variable("cache", "cached_key"):
                mask_shift = self.variables["cache"]["cache_index"]
                # max_length of cached_key is last dim
                max_decoder_length = self.variables["cache"]["cached_key"].shape[-1]
                causal_mask = lax.dynamic_slice(
                    self.causal_mask,
                    (0, 0, mask_shift, 0),
                    (1, 1, query_length, max_decoder_length),
                )
            else:
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]
            causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        # combine masks if needed
        if attention_mask is not None and self.causal:
            attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
            attention_mask = combine_masks(attention_mask, causal_mask)
        elif self.causal:
            attention_mask = causal_mask
        elif attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.

        if self.causal and (self.has_variable("cache", "cached_key") or init_cache):
            key_states, value_states, attention_mask = self._concatenate_to_cache(
                key_states, value_states, query_states, attention_mask
            )

        # Convert the boolean attention mask to an attention bias.
        if attention_mask is not None:
            # attention mask in the form of attention bias
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
        else:
            attention_bias = None

        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

    def _split_heads(self, hidden_state) -> jnp.ndarray:
        return hidden_state.reshape(hidden_state.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_state) -> jnp.ndarray:
        return hidden_state.reshape(hidden_state.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        # The following code is largely copied from: https://github.com/google-research/t5x/blob/63d9addf628c6d8c547a407a32095fcb527bb20b/t5x/examples/scalable_t5/layers.py#L280-L284
        is_initialized = self.has_variable("cache", "cached_key")

        # The key and value have dimension [batch_size, seq_length, num_heads, head_dim],
        # but we cache them as [batch_size, num_heads, head_dim, seq_length] as a TPU
        # fusion optimization. This also enables the "scatter via one-hot
        # broadcast" trick, which means we do a one-hot broadcast instead of a
        # scatter/gather operations, resulting in a 3-4x speedup in practice.
        def swap_dims(x):
            return x[:-3] + tuple(x[i] for i in [-2, -1, -3])

        cached_key = self.variable("cache", "cached_key", jnp.zeros, swap_dims(key.shape), key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, swap_dims(value.shape), value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            batch_size, num_heads, head_dim, seq_length = cached_key.value.shape
            # During fast autoregressive decoding, we feed one position at a time,
            # and cache the keys and values step by step.
            # Sanity shape check of cached key against input query.
            num_updated_cache_vectors = query.shape[1]
            expected_shape = (batch_size, 1, num_heads, head_dim)
            if num_updated_cache_vectors == 1 and expected_shape != query.shape:
                raise ValueError(
                    "Autoregressive cache shape error, expected query shape"
                    f" {expected_shape} instead got {query.shape}"
                )

            # Create a OHE of the current index. NOTE: the index is increased below.
            cur_index = cache_index.value

            # In order to update the key, value caches with the current key and
            # value, we move the seq_length axis to the back, similar to what we did for
            # the cached ones above.
            # Note these are currently the key and value of a single position, since
            # we feed one position at a time.
            one_token_key = jnp.moveaxis(key, -3, -1)
            one_token_value = jnp.moveaxis(value, -3, -1)

            # Update key, value caches with our new 1d spatial slices.
            # We implement an efficient scatter into the cache via one-hot
            # broadcast and addition.
            if num_updated_cache_vectors > 1:
                indices = jnp.eye(num_updated_cache_vectors, seq_length)[None, None]
                key = cached_key.value + jnp.matmul(one_token_key, indices)
                value = cached_value.value + jnp.matmul(one_token_value, indices)
            else:
                one_hot_indices = jax.nn.one_hot(cur_index, seq_length, dtype=key.dtype)
                key = cached_key.value + one_token_key * one_hot_indices
                value = cached_value.value + one_token_value * one_hot_indices

            cached_key.value = key
            cached_value.value = value
            cache_index.value = cache_index.value + num_updated_cache_vectors

            # Move the keys and values back to their original shapes.
            key = jnp.moveaxis(key, -1, -3)
            value = jnp.moveaxis(value, -1, -3)

            # causal mask for cached decoder self-attention: our single query position should only
            # attend to those key positions that have already been generated and cached, not the
            # remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(seq_length) < cur_index + num_updated_cache_vectors,
                (batch_size,) + (1, num_updated_cache_vectors, seq_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)

        return key, value, attention_mask


class FlaxWhisperEncoderLayer(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    params_dtype: jnp.dtype = jnp.float32
    use_scan: bool = False

    def setup(self) -> None:
        self.embed_dim = self.config.d_model
        self.self_attn = FlaxWhisperAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.encoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
        )
        self.self_attn_layer_norm = LayerNorm(dtype=self.dtype, epsilon=1e-05, params_dtype=self.params_dtype)
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)
        self.fc1 = DenseGeneral(
            self.config.encoder_ffn_dim,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
            kernel_axes=("embed", "mlp"),
        )
        self.fc2 = DenseGeneral(
            self.embed_dim,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
            kernel_axes=("mlp", "embed"),
        )
        self.final_layer_norm = LayerNorm(dtype=self.dtype, epsilon=1e-05, params_dtype=self.params_dtype)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        output_attentions: bool = True,
        deterministic: bool = True,
        all_hidden_states=None,  # only used when `use_scan=True` -> we have to fetch the hidden states from within the layer
    ) -> Tuple[jnp.ndarray]:
        if self.use_scan:
            hidden_states = hidden_states[0]

        hidden_states = with_sharding_constraint(hidden_states, ("batch", "length", "embed"))

        residual = hidden_states

        layernorm_output = self.self_attn_layer_norm(hidden_states)
        layernorm_output = with_sharding_constraint(layernorm_output, ("batch", "length", "embed"))

        attn_output, attn_weights = self.self_attn(hidden_states=layernorm_output, attention_mask=attention_mask)
        attn_output = self.dropout_layer(attn_output, deterministic=deterministic)
        attn_output = residual + attn_output
        attn_output = with_sharding_constraint(attn_output, ("batch", "length", "embed"))

        residual = attn_output

        post_layer_norm = self.final_layer_norm(attn_output)
        post_layer_norm = with_sharding_constraint(post_layer_norm, ("batch", "length", "embed"))

        fc1_output = self.activation_fn(self.fc1(post_layer_norm))
        fc1_output = self.activation_dropout_layer(fc1_output, deterministic=deterministic)
        fc1_output = with_sharding_constraint(fc1_output, ("batch", "length", "mlp"))

        hidden_states = self.fc2(fc1_output)
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states
        hidden_states = with_sharding_constraint(hidden_states, ("batch", "length", "embed"))

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if self.use_scan:
            if all_hidden_states is not None:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = (
                outputs,
                all_hidden_states,
            )

        return outputs


class FlaxWhisperEncoderLayerCollection(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    params_dtype: jnp.dtype = jnp.float32
    use_scan: bool = False
    gradient_checkpointing: bool = False

    @nn.compact
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        FlaxWhisperEncoderCheckpointLayer = (
            remat(
                FlaxWhisperEncoderLayer,
                static_argnums=(2, 3),
                prevent_cse=not self.use_scan,
            )
            if self.gradient_checkpointing
            else FlaxWhisperEncoderLayer
        )

        if self.use_scan:
            if output_attentions:
                raise ValueError("Cannot use `scan` with `output_attentions` set to True")

            # nicest behaviour for scan is to let the compiler figure out the correct shapes for the hidden states
            # so we'll just pass an empty tuple as the carry initializer and hold on to the first hidden states for later
            input_hidden_states = hidden_states
            hidden_states = (hidden_states,)

            hidden_states, all_hidden_states = scan_with_axes(
                FlaxWhisperEncoderCheckpointLayer,
                variable_axes={"params": 0, "cache": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=(
                    nn.broadcast,
                    nn.broadcast,
                    nn.broadcast,
                    nn.broadcast,
                ),
                variable_carry="all_hidden_states",
                length=self.config.encoder_layers,
            )(
                self.config,
                dtype=self.dtype,
                params_dtype=self.params_dtype,
                use_scan=True,
                name="FlaxEncoderScanLayers",
            )(
                hidden_states,
                attention_mask,
                output_attentions,
                deterministic,
                all_hidden_states,  # tuple intializer (or None if not using output_hidden_states)
            )

            # remove the scan dimension
            hidden_states = hidden_states[0]

            if output_hidden_states:
                # if we're using scan we'll surely be training -> return hidden states as a tensor rather than tuple
                all_hidden_states = jnp.vstack([input_hidden_states[None, ...], all_hidden_states[0]])

        else:
            for layer_idx in range(self.config.encoder_layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                dropout_probability = random.uniform(0, 1)
                if not deterministic and (dropout_probability < self.config.encoder_layerdrop):  # skip the layer
                    layer_outputs = (None, None)
                else:
                    layer_outputs = FlaxWhisperEncoderCheckpointLayer(
                        self.config,
                        dtype=self.dtype,
                        params_dtype=self.params_dtype,
                        name=str(layer_idx),
                    )(
                        hidden_states,
                        attention_mask,
                        output_attentions,
                        deterministic,
                    )
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        outputs = (hidden_states, all_hidden_states, all_attentions)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class FlaxWhisperDecoderLayer(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    params_dtype: jnp.dtype = jnp.float32
    use_scan: bool = False

    def setup(self) -> None:
        self.embed_dim = self.config.d_model
        self.self_attn = FlaxWhisperAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            causal=True,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
        )
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.activation_dropout_layer = nn.Dropout(rate=self.config.activation_dropout)

        self.self_attn_layer_norm = LayerNorm(dtype=self.dtype, epsilon=1e-05, params_dtype=self.params_dtype)
        self.encoder_attn = FlaxWhisperAttention(
            config=self.config,
            embed_dim=self.embed_dim,
            num_heads=self.config.decoder_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
        )
        self.encoder_attn_layer_norm = LayerNorm(dtype=self.dtype, epsilon=1e-05, params_dtype=self.params_dtype)
        self.fc1 = DenseGeneral(
            self.config.decoder_ffn_dim,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
            kernel_axes=("embed", "mlp"),
        )
        self.fc2 = DenseGeneral(
            self.embed_dim,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
            kernel_axes=("mlp", "embed"),
        )
        self.final_layer_norm = LayerNorm(dtype=self.dtype, epsilon=1e-05, params_dtype=self.params_dtype)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        output_attentions: bool = True,
        deterministic: bool = True,
        all_hidden_states=None,  # only used when `use_scan=True` -> we have to fetch the hidden states from within the layer
    ) -> Tuple[jnp.ndarray]:
        if self.use_scan:
            hidden_states = hidden_states[0]

        hidden_states = with_sharding_constraint(hidden_states, ("batch", "length", "embed"))

        residual = hidden_states

        layer_norm_output = self.self_attn_layer_norm(hidden_states)
        layer_norm_output = with_sharding_constraint(layer_norm_output, ("batch", "length", "embed"))

        # Self Attention
        self_attn_output, self_attn_weights = self.self_attn(
            hidden_states=layer_norm_output,
            attention_mask=attention_mask,
            init_cache=init_cache,
        )
        self_attn_output = self.dropout_layer(self_attn_output, deterministic=deterministic)
        self_attn_output = residual + self_attn_output
        self_attn_output = with_sharding_constraint(self_attn_output, ("batch", "length", "embed"))

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = self_attn_output

            encoder_layer_norm_output = self.encoder_attn_layer_norm(self_attn_output)
            encoder_layer_norm_output = with_sharding_constraint(
                encoder_layer_norm_output, ("batch", "length", "embed")
            )

            cross_attn_output, cross_attn_weights = self.encoder_attn(
                hidden_states=encoder_layer_norm_output,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            cross_attn_output = self.dropout_layer(cross_attn_output, deterministic=deterministic)
            cross_attn_output = residual + cross_attn_output
            cross_attn_output = with_sharding_constraint(cross_attn_output, ("batch", "length", "embed"))

        # Fully Connected
        residual = cross_attn_output

        post_layer_norm = self.final_layer_norm(cross_attn_output)
        post_layer_norm = with_sharding_constraint(post_layer_norm, ("batch", "length", "embed"))

        fc1_output = self.activation_fn(self.fc1(post_layer_norm))
        fc1_output = self.activation_dropout_layer(fc1_output, deterministic=deterministic)
        fc1_output = with_sharding_constraint(fc1_output, ("batch", "length", "mlp"))

        hidden_states = self.fc2(fc1_output)
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states
        hidden_states = with_sharding_constraint(hidden_states, ("batch", "length", "embed"))

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if self.use_scan:
            if all_hidden_states is not None:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = (
                outputs,
                all_hidden_states,
            )

        return outputs


class FlaxWhisperDecoderLayerCollection(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    params_dtype: jnp.dtype = jnp.float32
    use_scan: bool = False
    gradient_checkpointing: bool = False

    @nn.compact
    def __call__(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        FlaxWhisperDecoderCheckpointLayer = (
            remat(
                FlaxWhisperDecoderLayer,
                static_argnums=(4, 5, 6),
                prevent_cse=not self.use_scan,
            )
            if self.gradient_checkpointing
            else FlaxWhisperDecoderLayer
        )

        if self.use_scan:
            if output_attentions:
                raise ValueError("Cannot use `scan` with `output_attentions` set to True")

            input_hidden_states = hidden_states
            hidden_states = (hidden_states,)

            hidden_states, all_hidden_states = scan_with_axes(
                FlaxWhisperDecoderCheckpointLayer,
                variable_axes={"params": 0, "cache": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=(
                    nn.broadcast,
                    nn.broadcast,
                    nn.broadcast,
                    nn.broadcast,
                    nn.broadcast,
                    nn.broadcast,
                    nn.broadcast,
                ),
                variable_carry="all_hidden_states",
                length=self.config.decoder_layers,
            )(
                self.config,
                dtype=self.dtype,
                params_dtype=self.params_dtype,
                use_scan=True,
                name="FlaxDecoderScanLayers",
            )(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                init_cache,
                output_attentions,
                deterministic,
                all_hidden_states,
            )
            hidden_states = hidden_states[0]

            if output_hidden_states:
                # if we're using scan we'll surely be training -> return hidden states as a tensor rather than tuple
                all_hidden_states = jnp.vstack([input_hidden_states[None, ...], all_hidden_states[0]])

        else:
            for layer_idx in range(self.config.decoder_layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                    # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                dropout_probability = random.uniform(0, 1)
                if not deterministic and (dropout_probability < self.config.decoder_layerdrop):
                    layer_outputs = (None, None, None)
                else:
                    layer_outputs = FlaxWhisperDecoderCheckpointLayer(
                        self.config,
                        dtype=self.dtype,
                        params_dtype=self.params_dtype,
                        name=str(layer_idx),
                    )(
                        hidden_states,
                        attention_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        init_cache,
                        output_attentions,
                        deterministic,
                    )

                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                    if encoder_hidden_states is not None:
                        all_cross_attentions += (layer_outputs[2],)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        outputs = [
            hidden_states,
            all_hidden_states,
            all_self_attns,
            all_cross_attentions,
        ]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class FlaxWhisperEncoder(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    params_dtype: jnp.dtype = jnp.float32
    use_scan: bool = False
    gradient_checkpointing: bool = False

    def setup(self) -> None:
        self.conv1 = Conv(
            self.config.d_model,
            kernel_size=(3,),
            padding=1,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
            kernel_axes=("channels", "num_mel", "embed"),
        )
        self.conv2 = Conv(
            self.config.d_model,
            kernel_size=(3,),
            strides=2,
            padding=1,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
            kernel_axes=("channels", "embed", "num_mel"),
        )

        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        self.layers = FlaxWhisperEncoderLayerCollection(
            self.config,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
            use_scan=self.use_scan,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.embed_positions = Embed(
            self.config.max_source_positions,
            self.config.d_model,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
        )

        self.layer_norm = LayerNorm(dtype=self.dtype, epsilon=1e-05, params_dtype=self.params_dtype)

    def __call__(
        self,
        input_features: jnp.ndarray,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        if input_features.shape[1:] != (
            self.config.num_mel_bins,
            self.config.max_source_positions * 2,
        ):
            raise ValueError(
                "input_features.shape[1:], must be equal to (self.config.num_mel_bins,"
                " self.config.max_source_positions * 2) (got"
                f" {input_features.shape[1:]}, but should be"
                f" ({self.config.num_mel_bins},"
                f" {self.config.max_source_positions * 2}))"
            )

        input_features = input_features.transpose(0, 2, 1)
        hidden_states = jax.nn.gelu(self.conv1(input_features), approximate=False)
        hidden_states = with_sharding_constraint(hidden_states, ("batch", "embed", "num_mel"))
        hidden_states = jax.nn.gelu(self.conv2(hidden_states), approximate=False)
        hidden_states = with_sharding_constraint(hidden_states, ("batch", "length", "embed"))

        embed_positions = self.embed_positions(jnp.arange(self.config.max_source_positions))
        # sinusoidal positional embeddings should not be trained
        embed_positions = jax.lax.stop_gradient(embed_positions)
        hidden_states = hidden_states + embed_positions

        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        outputs = self.layers(
            hidden_states,
            attention_mask=None,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_states = outputs[0]
        last_hidden_states = self.layer_norm(last_hidden_states)

        # update the last element in `hidden_states` after applying `layernorm` above
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            if self.use_scan:
                hidden_states = jnp.vstack([hidden_states[:-1], last_hidden_states[None, ...]])
            else:
                hidden_states = hidden_states[:-1] + (last_hidden_states,)

        if not return_dict:
            outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=last_hidden_states,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )


class FlaxWhisperDecoder(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    params_dtype: jnp.dtype = jnp.float32
    use_scan: bool = False
    gradient_checkpointing: bool = False

    def setup(self) -> None:
        self.embed_tokens = Embed(
            self.config.vocab_size,
            self.config.d_model,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
        )
        self.embed_positions = Embed(
            self.config.max_target_positions,
            self.config.d_model,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
        )

        self.layers = FlaxWhisperDecoderLayerCollection(
            self.config,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
            use_scan=self.use_scan,
            gradient_checkpointing=self.gradient_checkpointing,
        )

        self.dropout_layer = nn.Dropout(rate=self.config.dropout)

        self.layer_norm = LayerNorm(dtype=self.dtype, epsilon=1e-5, params_dtype=self.params_dtype)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
        position_ids: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        input_embeds = self.embed_tokens(input_ids)
        position_embeds = self.embed_positions(position_ids)

        hidden_states = input_embeds + position_embeds
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)

        outputs = self.layers(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_states = outputs[0]
        last_hidden_states = self.layer_norm(last_hidden_states)

        # update the last element in `hidden_states` after applying `layernorm` above
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            if self.use_scan:
                hidden_states = jnp.vstack([hidden_states[:-1], last_hidden_states[None, ...]])
            else:
                hidden_states = hidden_states[:-1] + (last_hidden_states,)

        if not return_dict:
            outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_states,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class FlaxWhisperModule(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    params_dtype: jnp.dtype = jnp.float32
    use_scan: bool = False
    gradient_checkpointing: bool = False

    def setup(self) -> None:
        self.encoder = FlaxWhisperEncoder(
            self.config,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
            use_scan=self.use_scan,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.decoder = FlaxWhisperDecoder(
            self.config,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
            use_scan=self.use_scan,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    def __call__(
        self,
        input_features: jnp.ndarray,
        decoder_input_ids: jnp.ndarray,
        decoder_attention_mask: jnp.ndarray,
        decoder_position_ids: jnp.ndarray,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        freeze_encoder: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        encoder_outputs = self.encoder(
            input_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        encoder_hidden_states = encoder_outputs[0]

        if freeze_encoder:
            encoder_hidden_states = jax.lax.stop_gradient(encoder_hidden_states)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def _get_encoder_module(self):
        return self.encoder

    def _get_decoder_module(self):
        return self.decoder


class FlaxWhisperPreTrainedModel(FlaxPreTrainedModel):
    config_class = WhisperConfig
    base_model_prefix: str = "model"
    main_input_name = "input_features"
    module_class: nn.Module = None

    def __init__(
        self,
        config: WhisperConfig,
        input_shape: Tuple[int, int, int] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        # Can only use_scan=True in init if loading scanned weights -> need to handle use_scan=True and unrolled weights
        use_scan: bool = False,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        self.use_scan = use_scan
        self.gradient_checkpointing = gradient_checkpointing

        module = self.module_class(
            config=config,
            dtype=dtype,
            params_dtype=params_dtype,
            use_scan=use_scan,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs,
        )

        if input_shape is None:
            input_shape = (1, config.num_mel_bins, 2 * config.max_source_positions)

        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_features = jnp.zeros(input_shape, dtype="f4")
        input_features = input_features.at[(..., -1)].set(self.config.eos_token_id)

        decoder_input_ids = jnp.zeros((input_shape[0], 1), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        batch_size, sequence_length = decoder_input_ids.shape
        decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(
            rngs,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
        )["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            use_scan=self.use_scan,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    def enable_scan(self):
        self.use_scan = True
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            use_scan=self.use_scan,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        init_fn = partial(self.init_weights, input_shape=self.input_shape)
        params_shape_tree = jax.eval_shape(init_fn, self.key)

        # get the shape of the parameters
        self._params_shape_tree = params_shape_tree

        # save required_params as set
        self._required_params = set(flatten_dict(unfreeze(params_shape_tree)).keys())

        # initialize the parameters
        if self._is_initialized:
            self.params = self.convert_unroll_to_scan(self.params)

    def disable_scan(self):
        self.use_scan = False
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            use_scan=self.use_scan,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        init_fn = partial(self.init_weights, input_shape=self.input_shape)
        params_shape_tree = jax.eval_shape(init_fn, self.key)

        # get the shape of the parameters
        self._params_shape_tree = params_shape_tree

        # save required_params as set
        self._required_params = set(flatten_dict(unfreeze(params_shape_tree)).keys())

        # initialize the parameters
        if self._is_initialized:
            self.params = self.convert_scan_to_unroll(self.params)

    def convert_unroll_to_scan(self, params: Union[Dict, FrozenDict]):
        r"""
        Convert a `PyTree` of unrolled model parameters to a scanned block of model parameters. This method can be used
        to explicitly convert the model parameters to scanned format. This returns a new `params` tree and does not
        convert the `params` in place.

        To illustrate the workings of this method, take the Flax BERT model. The unrolled structure for the query
        projection params is as follows:
            ('bert', 'encoder', 'layer', '0', 'self_attn', 'q_proj') ('bert', 'encoder', 'layer', '1', 'self_attn',
            'q_proj') ... ('bert', 'encoder', 'layer', '23', 'self_attn', 'q_proj')
        This method takes each of the `q_proj` matrices for layers (0, ..., 23) and stacks them into a single 'super'
        matrix, giving a *single* block of weights for all 24 layers compatible with the scanned model:
            ('bert', 'encoder', 'layer', 'ScanLayers', 'self_attn', 'q_proj')

        When enabling scan with _do_init=True (default), this method will be called automatically under the hood. With
        _do_init=False, it will have to be called explicitly (see example below).

        Arguments:
            params (`Union[Dict, FrozenDict]`):
                A `PyTree` of model parameters.

        Examples:

        ```python
        >>> from distil_whisper import FlaxWhisperForConditionalGeneration

        >>> # Download model and configuration from huggingface.co
        >>> model, params = FlaxWhisperModel.from_pretrained("openai/whisper-tiny.en", _do_init=False)
        >>> # By default, the model params will be in unrolled format. To illustrate the use of this method,
        >>> # we'll first convert to scan format and then back to unrolled
        >>> model.enable_scan()
        >>> params = model.convert_unroll_to_scan(params)
        >>> # now convert back to unrolled
        >>> model.disable_scan()
        >>> params = model.convert_scan_to_unroll(params)
        ```"""
        if isinstance(params, FrozenDict):
            params = unfreeze(params)

        params = flatten_dict(params, sep="/")
        keys = list(params.keys())

        for k in keys:
            # Identify all "unrolled" layers formed as part of the FlaxBertLayerCollection
            # These params contain the identifier `layer` in their key
            if "layers/0" in k:
                if "decoder" in k:
                    block_prefix = "Decoder"
                    num_hidden_layers = self.config.decoder_layers
                else:
                    block_prefix = "Encoder"
                    num_hidden_layers = self.config.encoder_layers

                # Squash the keys for the N unrolled layers into one single key:
                # (layer/0, ..., layer/N) -> layer/FlaxScanLayers
                scan_key = k.replace("0", f"Flax{block_prefix}ScanLayers")
                stacked_params = []

                # Iterate over the unrolled layers (1,...,N)
                for i in range(num_hidden_layers):
                    # Stack the params for the N layers into one super block
                    # and remove the unrolled layer params on the fly
                    # -> no memory overhead for conversion!
                    unrolled_layer = params.pop(k.replace("0", str(i)))
                    stacked_params.append(unrolled_layer)

                params[scan_key] = jnp.stack(stacked_params)

        # Finally, unflatten the dict to restore the nested pytree structure
        params = unflatten_dict(params, sep="/")
        return params

    def convert_scan_to_unroll(self, params: Union[Dict, FrozenDict]):
        r"""
        Convert a `PyTree` of scanned model parameters to an unrolled stack of model parameters. This method can be
        used to explicitly convert the model parameters to unrolled format. This returns a new `params` tree and does
        not convert the `params` in place.

        To illustrate the workings of this method, take the Flax BERT model. The scanned structure for the query
        projection (`q_proj`) params is a single, stacked matrix of parameters over all N layers:
            ('bert', 'encoder', 'layer', 'FlaxScanLayers', 'self_attn', 'q_proj')

        This method slices each layer of the `q_proj` scanned matrix into single, standalone layers, and replaces the
        scanned matrix of parameteres on the fly:
            ('bert', 'encoder', 'layer', '0', 'self_attn', 'q_proj') ('bert', 'encoder', 'layer', '1', 'self_attn',
            'q_proj') ... ('bert', 'encoder', 'layer', 'N', 'self_attn', 'q_proj')

        When enabling scan with _do_init=True (default), this method will be called automatically under the hood. With
        _do_init=False, it will have to be called explicitly (see example below).

        Arguments:
            params (`Union[Dict, FrozenDict]`):
                A `PyTree` of model parameters.

        Examples:

        ```python
        >>> from distil_whisper import FlaxWhisperForConditionalGeneration

        >>> # Download model and configuration from huggingface.co
        >>> model, params = FlaxWhisperModel.from_pretrained("openai/whisper-tiny.en", _do_init=False)
        >>> # By default, the model params will be in unrolled format. To illustrate the use of this method,
        >>> # we'll first convert to scan format and then back to unrolled
        >>> model.enable_scan()
        >>> params = model.convert_unroll_to_scan(params)
        >>> # now convert back to unrolled
        >>> model.disable_scan()
        >>> params = model.convert_scan_to_unroll(params)
        ```"""

        if isinstance(params, FrozenDict):
            params = unfreeze(params)

        params = flatten_dict(params, sep="/")
        keys = list(params.keys())

        for k in keys:
            # Identify all "scan" layers formed as part of the FlaxBertLayerCollection
            # These params contain the identifier `FlaxScanLayers` in their key
            if "FlaxEncoderScanLayers" in k:
                # Remove the scan layer from the PyTree of params
                scan_layer = params.pop(k)

                # Unroll the key for the stacked scan matrix into N separate keys, indexed by layer number
                # layer/FlaxScanLayers -> (layer/0, ..., layer/N)
                for i in range(self.config.encoder_layers):
                    # Unstack the params for the i-th scan layer to unrolled
                    # and remove corresponding scan params on the fly
                    # -> no memory overhead for conversion!
                    unrolled_key = k.replace("FlaxEncoderScanLayers", str(i))
                    params[unrolled_key], scan_layer = scan_layer[0], scan_layer[1:]

            elif "FlaxDecoderScanLayers" in k:
                # Remove the scan layer from the PyTree of params
                scan_layer = params.pop(k)

                # Unroll the key for the stacked scan matrix into N separate keys, indexed by layer number
                # layer/FlaxScanLayers -> (layer/0, ..., layer/N)
                for i in range(self.config.decoder_layers):
                    # Unstack the params for the i-th scan layer to unrolled
                    # and remove corresponding scan params on the fly
                    # -> no memory overhead for conversion!
                    unrolled_key = k.replace("FlaxDecoderScanLayers", str(i))
                    params[unrolled_key], scan_layer = scan_layer[0], scan_layer[1:]

        params = unflatten_dict(params, sep="/")
        return params

    # Copied from transformers.models.whisper.modeling_flax_whisper.FlaxWhisperPreTrainedModel.init_cache
    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*:
                `attentions`). `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*)
                is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
                cross-attention of the decoder.
        """
        # init input variables to retrieve cache
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        decoder_position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(decoder_input_ids).shape[-1]),
            decoder_input_ids.shape,
        )

        def _decoder_forward(
            module,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_position_ids,
            **kwargs,
        ):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                decoder_input_ids,
                decoder_attention_mask,
                decoder_position_ids,
                **kwargs,
            )

        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            init_cache=True,
            method=_decoder_forward,  # we only need to call the decoder to init the cache
        )
        return unfreeze(init_variables["cache"])

    @add_start_docstrings(WHISPER_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=WhisperConfig)
    def encode(
        self,
        input_features: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        **kwargs,
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
        >>> input_features = inputs.input_features
        >>> encoder_outputs = model.encode(input_features=input_features)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _encoder_forward(module, input_features, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_features, **kwargs)

        return self.module.apply(
            {"params": params or self.params},
            input_features=jnp.array(input_features, dtype="f4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            method=_encoder_forward,
        )

    @add_start_docstrings(WHISPER_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=FlaxBaseModelOutputWithPastAndCrossAttentions,
        config_class=WhisperConfig,
    )
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
        >>> input_features = inputs.input_features
        >>> encoder_outputs = model.encode(input_features=input_features)
        >>> decoder_start_token_id = model.config.decoder_start_token_id

        >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
        >>> last_decoder_hidden_states = outputs.last_hidden_state
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        encoder_hidden_states = encoder_outputs[0]

        batch_size, sequence_length = decoder_input_ids.shape
        if decoder_position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `decoder_position_ids` when passing `past_key_values`.")

            if decoder_attention_mask is not None:
                decoder_position_ids = (decoder_attention_mask.cumsum(-1) * decoder_attention_mask) - 1
            else:
                decoder_position_ids = jnp.broadcast_to(
                    jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
                )

        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be
        # passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that
        # it can be changed by FlaxWhisperAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        def _decoder_forward(
            module,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_position_ids,
            **kwargs,
        ):
            decoder_module = module._get_decoder_module()
            return decoder_module(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                position_ids=decoder_position_ids,
                **kwargs,
            )

        outputs = self.module.apply(
            inputs,
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            mutable=mutable,
            method=_decoder_forward,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past = outputs
            outputs["past_key_values"] = unfreeze(past["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past = outputs
            outputs = outputs[:1] + (unfreeze(past["cache"]),) + outputs[1:]

        return outputs

    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_features: jnp.ndarray,
        decoder_input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        freeze_encoder: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # prepare decoder inputs
        if decoder_position_ids is None:
            if decoder_attention_mask is not None:
                decoder_position_ids = (decoder_attention_mask.cumsum(-1) * decoder_attention_mask) - 1
            else:
                batch_size, sequence_length = decoder_input_ids.shape
                decoder_position_ids = jnp.broadcast_to(
                    jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
                )
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}

        return self.module.apply(
            {"params": params or self.params},
            input_features=jnp.array(input_features, dtype="f4"),
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            freeze_encoder=freeze_encoder,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
        )


@add_start_docstrings(
    ("The bare Whisper Model transformer outputting raw hidden-states without any specific head on top."),
    WHISPER_START_DOCSTRING,
)
class FlaxWhisperModel(FlaxWhisperPreTrainedModel):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    params_dtype: jnp.dtype = jnp.float32
    module_class = FlaxWhisperModule


append_call_sample_docstring(FlaxWhisperModel, _CHECKPOINT_FOR_DOC, FlaxSeq2SeqModelOutput, _CONFIG_FOR_DOC)


class FlaxWhisperForConditionalGenerationModule(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    params_dtype: jnp.dtype = jnp.float32
    use_scan: bool = False
    gradient_checkpointing: bool = False

    def setup(self) -> None:
        self.model = FlaxWhisperModule(
            config=self.config,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
            use_scan=self.use_scan,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.lm_head = DenseGeneral(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
            kernel_axes=("embed", "vocab"),
        )

    def _get_encoder_module(self):
        return self.model.encoder

    def _get_decoder_module(self):
        return self.model.decoder

    def __call__(
        self,
        input_features,
        decoder_input_ids,
        decoder_attention_mask: jnp.ndarray = None,
        decoder_position_ids: jnp.ndarray = None,
        position_ids: jnp.ndarray = None,
        attention_mask: jnp.ndarray = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        freeze_encoder: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
    ):
        outputs = self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            freeze_encoder=freeze_encoder,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_embedding = self.model.decoder.embed_tokens.variables["params"]["embedding"]
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return output

        return FlaxSeq2SeqLMOutput(
            logits=lm_logits,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


@add_start_docstrings("The Whisper Model with a language modeling head.", WHISPER_START_DOCSTRING)
class FlaxWhisperForConditionalGeneration(FlaxWhisperPreTrainedModel):
    module_class = FlaxWhisperForConditionalGenerationModule

    @add_start_docstrings(WHISPER_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=WhisperConfig)
    def decode(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
        >>> input_features = inputs.input_features
        >>> encoder_outputs = model.encode(input_features=input_features)
        >>> decoder_start_token_id = model.config.decoder_start_token_id

        >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
        >>> last_decoder_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        encoder_hidden_states = encoder_outputs[0]

        batch_size, sequence_length = decoder_input_ids.shape
        if decoder_position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `decoder_position_ids` when passing `past_key_values`.")

            if decoder_attention_mask is not None:
                decoder_position_ids = (decoder_attention_mask.cumsum(-1) * decoder_attention_mask) - 1
            else:
                decoder_position_ids = jnp.broadcast_to(
                    jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
                )
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones((batch_size, sequence_length), dtype="i4")

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be
        # passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that
        # it can be changed by FlaxWhisperAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        def _decoder_forward(
            module,
            decoder_input_ids,
            decoder_attention_mask,
            decoder_position_ids,
            **kwargs,
        ):
            decoder_module = module._get_decoder_module()
            outputs = decoder_module(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                position_ids=decoder_position_ids,
                **kwargs,
            )
            hidden_states = outputs[0]

            if self.config.tie_word_embeddings:
                shared_embedding = module.model.decoder.embed_tokens.variables["params"]["embedding"]
                lm_logits = module.lm_head.apply({"params": {"kernel": shared_embedding.T}}, hidden_states)
            else:
                lm_logits = module.lm_head(hidden_states)

            return lm_logits, outputs

        outputs = self.module.apply(
            inputs,
            decoder_input_ids=jnp.array(decoder_input_ids, dtype="i4"),
            decoder_attention_mask=jnp.array(decoder_attention_mask, dtype="i4"),
            decoder_position_ids=jnp.array(decoder_position_ids, dtype="i4"),
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=not train,
            rngs=rngs,
            mutable=mutable,
            method=_decoder_forward,
        )

        if past_key_values is None:
            lm_logits, decoder_outputs = outputs
        else:
            (lm_logits, decoder_outputs), past = outputs

        if return_dict:
            outputs = FlaxCausalLMOutputWithCrossAttentions(
                logits=lm_logits,
                hidden_states=decoder_outputs.hidden_states,
                attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
            )
        else:
            outputs = (lm_logits,) + decoder_outputs[1:]

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs["past_key_values"] = unfreeze(past["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs = outputs[:1] + (unfreeze(past["cache"]),) + outputs[1:]

        return outputs

    def generate(
        self,
        input_features,
        generation_config=None,
        logits_processor=None,
        return_timestamps=None,
        task=None,
        language=None,
        is_multilingual=None,
        **kwargs,
    ):
        if generation_config is None:
            generation_config = self.generation_config

        if return_timestamps is not None:
            generation_config.return_timestamps = return_timestamps

        if task is not None:
            generation_config.task = task

        if is_multilingual is not None:
            generation_config.is_multilingual = is_multilingual

        if language is not None:
            generation_config.language = language

        if kwargs is not None and "decoder_input_ids" in kwargs:
            decoder_input_length = len(kwargs["decoder_input_ids"])
        else:
            decoder_input_length = 1

        forced_decoder_ids = []

        if hasattr(generation_config, "is_multilingual") and generation_config.is_multilingual:
            if hasattr(generation_config, "language"):
                forced_decoder_ids.append((1, generation_config.lang_to_id[generation_config.language]))
            else:
                forced_decoder_ids.append((1, None))

            if hasattr(generation_config, "task"):
                forced_decoder_ids.append((2, generation_config.task_to_id[generation_config.task]))
            else:
                forced_decoder_ids.append((2, generation_config.task_to_id["transcribe"]))

        if (
            hasattr(generation_config, "return_timestamps") and generation_config.return_timestamps
        ) or return_timestamps:
            logits_processor = [
                FlaxWhisperTimeStampLogitsProcessor(generation_config, self.config, decoder_input_length)
            ]
        else:
            if forced_decoder_ids and forced_decoder_ids[-1][0] != generation_config.no_timestamps_token_id:
                idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
                forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))

        if len(forced_decoder_ids) > 0:
            generation_config.forced_decoder_ids = forced_decoder_ids

        return super().generate(
            input_features,
            generation_config,
            logits_processor=logits_processor,
            **kwargs,
        )

    def pipeline_generate(
        self,
        input_features,
        forced_decoder_ids,
        return_timestamps=False,
        generation_config=None,
        **kwargs,
    ):
        if generation_config is None:
            generation_config = self.generation_config

        # override the generation config forced decoder ids in preference of the ones we have set
        generation_config.forced_decoder_ids = None

        logits_processor = FlaxLogitsProcessorList()
        logits_processor.append(FlaxStaticForceTokensLogitsProcessor(forced_decoder_ids))

        if hasattr(generation_config, "return_timestamps") and return_timestamps:
            logits_processor.append(FlaxWhisperTimeStampLogitsProcessor(generation_config, self.config, 1))

        return super().generate(
            input_features,
            generation_config,
            logits_processor=logits_processor,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        attention_mask: Optional[jax.Array] = None,
        decoder_attention_mask: Optional[jax.Array] = None,
        encoder_outputs=None,
        **kwargs,
    ):
        # initializing the cache
        batch_size, seq_length = decoder_input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since the decoder uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if decoder_attention_mask is not None:
            position_ids = decoder_attention_mask.cumsum(-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, decoder_attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "encoder_attention_mask": attention_mask,
            "decoder_attention_mask": extended_attention_mask,
            "decoder_position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        return model_kwargs


FLAX_WHISPER_CONDITIONAL_GENERATION_DOCSTRING = r"""
    Returns:

    Transcription example:

    ```python
    >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
    >>> from datasets import load_dataset

    >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
    >>> input_features = inputs.input_features
    >>> generated_ids = model.generate(input_ids=input_features)
    >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    >>> transcription
    ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
    ```
"""

overwrite_call_docstring(
    FlaxWhisperForConditionalGeneration,
    WHISPER_INPUTS_DOCSTRING + FLAX_WHISPER_CONDITIONAL_GENERATION_DOCSTRING,
)
append_replace_return_docstrings(
    FlaxWhisperForConditionalGeneration,
    output_type=FlaxSeq2SeqLMOutput,
    config_class=_CONFIG_FOR_DOC,
)
