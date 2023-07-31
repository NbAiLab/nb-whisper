import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze
from jax.sharding import PartitionSpec as P
from transformers import WhisperProcessor
from flax.training.common_utils import shard
import numpy as np
from datasets import load_dataset
from transformers import WhisperProcessor
import numpy as np
from copy import deepcopy


from whisper_jax import FlaxWhisperForConditionalGeneration, InferenceState, PjitPartitioner


# 2D parameter and activation partitioning for DP
logical_axis_rules_dp = [
    ("batch", "data"),
    ("mlp", None),
    ("heads", None),
    ("vocab", None),
    ("embed", None),
    ("embed", None),
    ("joined_kv", None),
    ("kv", None),
    ("length", None),
    ("num_mel", None),
    ("channels", None),
]

model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-tiny.en",
    _do_init=False,
    dtype=jnp.bfloat16,
)


def init_fn():
    input_shape = (1, 80, 3000)

    input_features = jnp.zeros(input_shape, dtype="f4")
    input_features = input_features.at[(..., -1)].set(model.config.eos_token_id)

    decoder_input_ids = jnp.zeros((input_shape[0], 1), dtype="i4")
    decoder_attention_mask = jnp.ones_like(decoder_input_ids)

    batch_size, sequence_length = decoder_input_ids.shape
    decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

    rng = jax.random.PRNGKey(0)
    init_params = model.module.init(
        rng,
        input_features=input_features,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        decoder_position_ids=decoder_position_ids,
        return_dict=False,
    )
    return init_params


# Axis names metadata
param_axes = jax.eval_shape(init_fn)["params_axes"]

# Create InferenceState, since the partitioner expects it
state = InferenceState(
    step=jnp.array(0),
    params=freeze(model.params_shape_tree),
    params_axes=freeze(param_axes),
    flax_mutables=None,
    flax_mutables_axes=param_axes,
)

# Define the pjit partitioner with 1 model partition
partitioner = PjitPartitioner(
    num_partitions=1,
    logical_axis_rules=logical_axis_rules_dp,
)

mesh_axes = partitioner.get_mesh_axes(state)
params_spec = mesh_axes.params

p_shard_params = partitioner.partition(model.to_bf16, (params_spec,), params_spec)


def generate(params, input_features):
    output_ids = model.generate(input_features, params=params, max_length=model.config.max_length).sequences
    return output_ids


p_generate = partitioner.partition(
    generate,
    in_axis_resources=(params_spec, P("data")),
    out_axis_resources=P("data"),
)


# I need to add these two lines to 
params_list = [deepcopy(params) for _ in range(jax.local_device_count())]
params = jax.device_put_sharded(params_list, jax.devices())

# This will auto-magically run in mesh context
params = p_shard_params(freeze(params))

# Prepare some data
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

# Load a batch of 4 samples
batch = [ds[i]["audio"] for i in range(4)]
input_features = [processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="np").input_features for sample in batch]

# Stack the input features into a single array
input_features = np.stack(input_features, axis=0)

# Shard
sharded_input_features = shard(input_features)

# Single example of shape (1, 80, 3000)
sample = ds[0]["audio"]
single_input_feature = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="np").input_features

# single_sharded_input_feature = shard(single_input_feature)

# you can now run the forward pass with: 
pred_ids = p_generate(params,input_features)
