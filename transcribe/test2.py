Chat history
New chat
Close sidebar
Today
Whisper Audio Transcription


Empty Eval Metrics Error
Yesterday
Simplify Code with Flexibility.
Latin Numerals - 21+
Previous 7 Days
Plots of Model Metrics
Tool Usefulness Ratings
Tool Usefulness Ratings
Toolbox Task Ratings
Toolbox Essentials
Students' Capitalism Sentences: Evaluate & Grade (SES-C)
Eval: Stu's Capitalism Sentences
Cap Soc: Despair in Capitalism
New chat
tmux Rename Window
Previous 30 Days
Unique Non-Letter Characters.
Dataset Processing Functions.
Generate Plots with 'step' and 'tab10'
Generate Plot with Metrics
Fluids NS Equations Problem.
Dataset Preprocessing with Python
Plots with Grouped Metrics.
New chat
Plot Performance Metrics Over Time.
Metrics Plotting with Smoothing
Plot Metrics Over Time.
Plotting Validation Metrics.
Custom Instructions: Access and AI
Dataset Loading Code Line
Modify Script:
New chat
Norske partier med "R".
Sámi-Norwegian MT5 Translation
Haiku om meningen.
Dikt om mening: Haiku.
Rearrange Lines: Script Order
Google TPU Topology Explained
Files with "v2" and "gradient"
New chat
Brosmetind turgåer beskrivelse.
Calculate SemDist with HuggingFace
New chat
New chat
PDF Conversion Assistance
XML File Description
Summarizing PDF request.
ChemTrails: Conspiracy or Contrails?
Budget-friendly Meal Plan
Chess Position Analysis.
Hiring for Competing ChatGPT
CSV Analysis with Matplot
CSV Graph Analysis
New chat
Oppdatering av Krav
Summarize in 4 words
Count Words in TSV File
Transfer Zip to GCloud
Disable Button for iOS
TPU v4 LLM Topology
Microphone Access Issue.
Tunnel Setup with pktriot
Thesis Evaluation Task
Model Development Collaboration
June
NB-Whisper Launching in Norway
Evaluate training scripts.
Calculate CER between texts.
Norwegian King's Copyright Status
Organize File Options
Add Null Field with jq
Nynorsk Errors Corrected
Oral vs. Subtitle: Omitted Words
Optimizing JSON-CSV Translation
NER & WER Calculation.
WER Calculation with Script
Jsonlines to TSV Pandas
NATO Operations and Conflicts
Translation Error: Invalid Project ID
Add Project Members Full Access
Cut MP3 on Silence
Project Name Brainstorm
ProtonVPN: Safe Software
Merge WER and NER
Python JSON Processing
New chat
Gör egen cottage cheese.
Merge JSON Pandas Python
Transpose Horizontal to Vertical
Comparing ASR Hyperparameters.
Move server between projects
Clean JSON Field Text
New chat
New chat
Latin Number Succession
Latin Ordinal Numbers
Epsilon in AdamW optimizer
Table Extraction: Steps & WER
KI i Norge: Åpenhet og data
Git Repo Size Limit
Rotate Panes in Tmux
NCC-S Dataset Wrapper
Validation Script Summary
Seventeenth: Septimus Decimus
New chat
NLP Researcher Job Search
Whisper ASR Transcription
JSON-schema definisjon og inkonsekvenser.
Google Books in Norwegian.
Google Books Access
Tablets for Cough Relief
Boligforsikring og råteskader
Kvinnesynet i Kardemommeby
Hours in a year
Convert verbosity field to int

User
Per Egil Kummervold
Code Interpreter

User
My main task here is to transcribe a large set of audio files using Whisper for Flax. The data should be loaded from a streaming HuggingFace dataset. I want to use the Whisper Flax library for this. Here is some info about the library:

"""Whisper JAX was tested using Python 3.9 and JAX version 0.4.5. Installation assumes that you already have the latest version of the JAX package installed on your device. You can do so using the official JAX installation guide: https://github.com/google/jax#installation

Once the appropriate version of JAX has been installed, Whisper JAX can be installed through pip:

pip install git+https://github.com/sanchit-gandhi/whisper-jax.git
To update the Whisper JAX package to the latest version, simply run:

pip install --upgrade --no-deps --force-reinstall git+https://github.com/sanchit-gandhi/whisper-jax.git
Pipeline Usage
The recommended way of running Whisper JAX is through the FlaxWhisperPipline abstraction class. This class handles all the necessary pre- and post-processing, as well as wrapping the generate method for data parallelism across accelerator devices.

Whisper JAX makes use of JAX's pmap function for data parallelism across GPU/TPU devices. This function is Just In Time (JIT) compiled the first time it is called. Thereafter, the function will be cached, enabling it to be run in super-fast time:

from whisper_jax import FlaxWhisperPipline

# instantiate pipeline
pipeline = FlaxWhisperPipline("openai/whisper-large-v2")

# JIT compile the forward call - slow, but we only do once
text = pipeline("audio.mp3")

# used cached function thereafter - super fast!!
text = pipeline("audio.mp3")
Half-Precision
The model computation can be run in half-precision by passing the dtype argument when instantiating the pipeline. This will speed-up the computation quite considerably by storing intermediate tensors in half-precision. There is no change to the precision of the model weights.

For most GPUs, the dtype should be set to jnp.float16. For A100 GPUs or TPUs, the dtype should be set to jnp.bfloat16:

from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

# instantiate pipeline in bfloat16
pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.bfloat16)
Batching
Whisper JAX also provides the option of batching a single audio input across accelerator devices. The audio is first chunked into 30 second segments, and then chunks dispatched to the model to be transcribed in parallel. The resulting transcriptions are stitched back together at the boundaries to give a single, uniform transcription. In practice, batching provides a 10x speed-up compared to transcribing the audio samples sequentially, with a less than 1% penalty to the WER1, provided the batch size is selected large enough.

To enable batching, pass the batch_size parameter when you instantiate the pipeline:

from whisper_jax import FlaxWhisperPipline

# instantiate pipeline with batching
pipeline = FlaxWhisperPipline("openai/whisper-large-v2", batch_size=16)
Task
By default, the pipeline transcribes the audio file in the language it was spoken in. For speech translation, set the task argument to "translate":

# translate
text = pipeline("audio.mp3", task="translate")
Timestamps
The FlaxWhisperPipline also supports timestamp prediction. Note that enabling timestamps will require a second JIT compilation of the forward call, this time including the timestamp outputs:

# transcribe and return timestamps
outputs = pipeline("audio.mp3",  task="transcribe", return_timestamps=True)
text = outputs["text"]  # transcription
chunks = outputs["chunks"]  # transcription + timestamps
Putting it all together
In the following code snippet, we instantiate the model in bfloat16 precision with batching enabled, and transcribe the audio file returning timestamps tokens:

from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

# instantiate pipeline with bfloat16 and enable batching
pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.bfloat16, batch_size=16)

# transcribe and return timestamps
outputs = pipeline("audio.mp3",  task="transcribe", return_timestamps=True)
Model Usage
The Whisper JAX model can use on a more granular level in much the same way as the original Hugging Face Transformers implementation. This requires the Whisper processor to be loaded separately to the model to handle the pre- and post-processing, and the generate function to be wrapped using pmap by hand:

import jax.numpy as jnp
from datasets import load_dataset
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from jax import device_get, pmap
from transformers import WhisperProcessor

from whisper_jax import FlaxWhisperForConditionalGeneration

# load the processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v2", dtype=jnp.bfloat16, _do_init=False,
)

def generate_fn(input_features):
    pred_ids = model.generate(
        input_features, task="transcribe", return_timestamps=False, max_length=model.config.max_length, params=params,
    )
    return pred_ids.sequences

# pmap the generate function for data parallelism
p_generate = pmap(generate_fn, "input_features")
# replicate the parameters across devices
params = replicate(params)

# load a dummy sample from the LibriSpeech dataset
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]

# pre-process: convert the audio array to log-mel input features
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="np").input_features
# replicate the input features across devices for DP
input_features = shard(input_features)

# run the forward pass (JIT compiled the first time it is called)
pred_ids = p_generate(input_features)
output_ids = device_get(pred_ids.reshape(-1, model.config.max_length))

# post-process: convert tokens ids to text string
transcription = processor.batch_decode(pred_ids, skip_special_tokens=True)"""

I also got the following advice (since I know that all my audio files are below 30 seconds):
"""If you know all of the audio files are less than 30s, then you probably don't need to worry about using the Flax Whisper Pipeline (since we don't need all the chunking logic).

What you can do instead is load up all of the audio files, pre-process them using the feature extractor, and then pass them in batches to the Flax Whisper model to be transcribed in parallel. There's an example that you can follow here:"""

"""import time

import jax
import jax.numpy as jnp
from datasets import concatenate_datasets, load_dataset
from flax import jax_utils
from flax.training.common_utils import shard
from transformers import FlaxWhisperForConditionalGeneration, WhisperProcessor


BATCH_SIZES = [4, 8, 16, 32, 64, 128]
NUM_BATCHES = 100
NUM_TOKENS = 25

model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v2",
    _do_init=False,
    dtype=jnp.bfloat16,
)

params = jax_utils.replicate(params)


def generate_fn(batch):
    pred_ids = model.generate(batch, params=params, max_new_tokens=NUM_TOKENS, min_new_tokens=NUM_TOKENS)
    return pred_ids.sequences


p_generate_fn = jax.pmap(generate_fn, "batch")

# processors/tokenizers are the same for all models, so just load from tiny and preprocess once
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")


def preprocess(batch):
    batch["input_features"] = processor(
        batch["audio"]["array"], sampling_rate=16000, return_tensors="np"
    ).input_features[0]
    return batch


# load a dataset of 73 audio samples
librispeech = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset_processed = librispeech.map(preprocess, remove_columns=librispeech.column_names)

for batch_size in BATCH_SIZES:
    eval_dataset = dataset_processed.select(range(batch_size // 2))
    eval_dataset = concatenate_datasets([eval_dataset for _ in range(2 * NUM_BATCHES)])

    eval_dataloader = eval_dataset.with_format("numpy").iter(batch_size=batch_size)

    # warm-up step
    batch = next(iter(eval_dataloader))
    input_features = shard(batch["input_features"])
    pred_ids = p_generate_fn(input_features)

    start = time.time()
    for batch in eval_dataloader:
        input_features = shard(batch["input_features"])
        pred_ids = p_generate_fn(input_features)
    runtime = time.time() - start

    print(f"{batch_size}: {runtime:.06}")"""

I would remove entirely the max/min tokens logic (this was just to compare the inference speed on a per-token basis to PyTorch) and set your batch size as required.
ChatGPT
In your case, you need to transcribe a large set of audio files using Whisper for Flax. The audio files are sourced from a streaming HuggingFace dataset, and you know that all the audio files are less than 30 seconds long.

Given these conditions, you should indeed follow the advice you received. Because all of your audio files are less than 30 seconds long, you don't need to use the Flax Whisper Pipeline's chunking logic. Instead, you can load all the audio files, preprocess them using the feature extractor, and then pass them in batches to the Flax Whisper model to be transcribed in parallel.

Here's a step-by-step outline of what you need to do:

Load the Model: Load the FlaxWhisperForConditionalGeneration model from the pretrained model. Set the dtype to bfloat16 for better performance.

Load the Processor: Load the WhisperProcessor, which will handle the conversion of audio data into input features that the model can understand.

Preprocess the Audio Files: Load your audio files and preprocess them using the WhisperProcessor. This will convert the audio data into input features.

Generate Transcriptions: Pass the preprocessed input features in batches to the FlaxWhisperForConditionalGeneration model. Use the JAX's pmap function to parallelize this operation for better performance.

Postprocess the Output: The output from the model will be token IDs, which you'll need to convert back to text. Use the WhisperProcessor's batch_decode function to do this.

Here is the code snippet you can use as a reference to achieve this:

python
Copy code
import time
import jax
import jax.numpy as jnp
from datasets import load_dataset
from flax import jax_utils
from flax.training.common_utils import shard
from transformers import FlaxWhisperForConditionalGeneration, WhisperProcessor

BATCH_SIZE = 16

model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v2",
    _do_init=False,
    dtype=jnp.bfloat16,
)

params = jax_utils.replicate(params)

def generate_fn(batch):
    pred_ids = model.generate(batch, params=params)
    return pred_ids.sequences

p_generate_fn = jax.pmap(generate_fn, "batch")

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

def preprocess(batch):
    batch["input_features"] = processor(
        batch["audio"]["array"], sampling_rate=16000, return_tensors="np"
    ).input_features[0]
    return batch

audio_files_dataset = load_dataset("your_dataset_name", split="your_split")  # replace with your dataset and split
dataset_processed = audio_files_dataset.map(preprocess, remove_columns=audio_files_dataset.column_names)

eval_dataloader = dataset_processed.with_format("numpy").iter(batch_size=BATCH_SIZE)

# warm-up step
batch = next(iter(eval_dataloader))
input_features = shard(batch["input_features"])
pred_ids = p_generate_fn(input_features)

start = time.time()
for batch in eval_dataloader:
    input_features = shard(batch["input_features"])
    pred_ids = p_generate_fn(input_features)
runtime = time.time() - start

print(f"{BATCH_SIZE}: {runtime:.06}")
Please replace "your_dataset_name" and "your_split" with your dataset name and split respectively.

Also, remember that this code is going to run on a GPU/TPU, so make sure you have one available and it is properly configured for use with JAX. You might need to adjust the batch size based on the memory available on your GPU/TPU.

The code will run a warm-up step first to JIT-compile the generate function, and then it will process all the batches of audio files. The total runtime will be printed at the end.

Note that this code does not handle the postprocessing step of converting the token IDs back to text. You can add that after the loop that processes the batches:

python
Copy code
# post-process: convert tokens ids to text string
transcriptions = processor.batch_decode(pred_ids, skip_special_tokens=True)
This will give you the transcriptions for all audio files processed in the final batch. You might want to include this inside the loop if you need the transcriptions immediately after each batch is processed, or store the pred_ids from each batch in a list so you can convert them all to text after all batches have been processed.




User
Make the code so that it loads the "NbAiLab/ncc_speech_v3" dataset. "train"-split. Use auth token. Also print the results to screen, and exit after 5 batches.
ChatGPT
Sure, here's how you can modify the script to load the "NbAiLab/ncc_speech_v3" dataset, "train" split, and use an auth token. I've also added code to print the transcriptions to the screen and exit after processing 5 batches.

python
Copy code
import time
import jax
import jax.numpy as jnp
from datasets import concatenate_datasets, load_dataset
from flax import jax_utils
from flax.training.common_utils import shard
from transformers import FlaxWhisperForConditionalGeneration, WhisperProcessor

# Set your BATCH_SIZE according to your GPU's memory availability
BATCH_SIZE = 16
NUM_BATCHES = 5

# Load the model and replicate the parameters across devices
model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v2",
    _do_init=False,
    dtype=jnp.bfloat16,
)

params = jax_utils.replicate(params)

# Function to generate predictions
def generate_fn(batch):
    pred_ids = model.generate(batch, params=params)
    return pred_ids.sequences

# Map the generate function to multiple devices
p_generate_fn = jax.pmap(generate_fn, "batch")

# Load the WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

# Function to preprocess the audio data
def preprocess(batch):
    batch["input_features"] = processor(
        batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"], return_tensors="np"
    ).input_features[0]
    return batch

# Provide your Hugging Face token here
huggingface_token = 'your_huggingface_token'
audio_files_dataset = load_dataset("NbAiLab/ncc_speech_v3", split="train", token=huggingface_token)

# Preprocess the audio data
dataset_processed = audio_files_dataset.map(preprocess, remove_columns=audio_files_dataset.column_names)

# Create a dataloader
eval_dataloader = dataset_processed.with_format("numpy").iter(batch_size=BATCH_SIZE)

# Warm-up step
batch = next(iter(eval_dataloader))
input_features = shard(batch["input_features"])
pred_ids = p_generate_fn(input_features)

# Run the model on the batches
start = time.time()
for i, batch in enumerate(eval_dataloader):
    input_features = shard(batch["input_features"])
    pred_ids = p_generate_fn(input_features)
    
    # Post-process: convert tokens ids to text string
    transcriptions = processor.batch_decode(jax.device_get(pred_ids.reshape(-1, model.config.max_length)), skip_special_tokens=True)
    
    # Print the transcriptions
    for transcription in transcriptions:
        print(transcription)
    
    # Exit after processing NUM_BATCHES batches
    if i >= NUM_BATCHES - 1:
        break

runtime = time.time() - start
print(f"\nTotal time for {NUM_BATCHES} batches: {runtime:.06}")
This script will print the transcription of each audio sample in the batches to the console. It will stop after processing the specified number of batches (5 in this case).

Please replace 'your_huggingface_token' with your actual Hugging Face token.

Remember, running this script requires a significant amount of GPU memory, so please adjust your batch size according to the GPU memory you have available.




User
(.whisper) perk@t1v-n-c53b857e-w-0:~/models/nb-whisper/transcribe$ python transcribe.py
/home/perk/.whisper/lib/python3.8/site-packages/tensorflow/python/framework/dtypes.py:40: RuntimeWarning: A cast from dtype(bfloat16) to dtype(float8_e4m3fn) was registered/modified using `PyArray_RegisterCastFunc` after the cast had been used.  This registration will have (mostly) no effect: the previous definition will continue to be used.
The most likely fix is to ensure that casts are the first thing initialized after dtype registration.  Please contact the NumPy developers with any questions!
  _pywrap_custom_casts.TF_register_custom_casts()
/home/perk/.whisper/lib/python3.8/site-packages/tensorflow/python/framework/dtypes.py:40: RuntimeWarning: A cast from dtype(bfloat16) to dtype(float8_e5m2) was registered/modified using `PyArray_RegisterCastFunc` after the cast had been used.  This registration will have (mostly) no effect: the previous definition will continue to be used.
The most likely fix is to ensure that casts are the first thing initialized after dtype registration.  Please contact the NumPy developers with any questions!
  _pywrap_custom_casts.TF_register_custom_casts()
2023-07-30 20:03:15.413384: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
https://symbolize.stripped_domain/r/?trace=7facf298a0c0,90c85f&map=
*** SIGSEGV (@(nil)), see gl__________50#s15 received by PID 118148 (TID 118148) on cpu 116; stack trace: ***
    @     0x7fab86b070da       1152  (unknown)
    @     0x7facf298a0c0  236745344  (unknown)
    @           0x90c860  (unknown)  (unknown)
https://symbolize.stripped_domain/r/?trace=7fab86b070da,7facf298a0bf,90c85f&map=1db795edf6ed53794bfb4c33fb861dd9:7fab7bc63000-7fab86d19dc0
E0730 20:03:21.216301  118148 coredump_hook.cc:408] RAW: Remote crash data gathering hook invoked.
E0730 20:03:21.216309  118148 coredump_hook.cc:447] RAW: Skipping coredump since rlimit was 0 at process start.
E0730 20:03:21.216323  118148 client.cc:281] RAW: Coroner client retries enabled (b/136286901), will retry for up to 30 sec.
E0730 20:03:21.216325  118148 coredump_hook.cc:506] RAW: Sending fingerprint to remote end.
E0730 20:03:21.216331  118148 coredump_socket.cc:117] RAW: Stat failed errno=2 on socket /var/google/services/logmanagerd/remote_coredump.socket
E0730 20:03:21.216334  118148 coredump_hook.cc:514] RAW: Cannot send fingerprint to Coroner: [NOT_FOUND] Missing crash reporting socket. Is the listener running?
E0730 20:03:21.216337  118148 coredump_hook.cc:566] RAW: Dumping core locally.
E0730 20:03:21.623244  118148 process_state.cc:784] RAW: Raising signal 11 with default behavior
ChatGPT
It appears that you're encountering a segmentation fault (SIGSEGV), which is usually caused by a program trying to read or write an illegal memory location. This can be a difficult issue to debug, particularly when it occurs within a third-party library, as seems to be the case here.

A few general steps you could take to diagnose this:

Check Dependencies: Make sure that all of the dependencies are correctly installed and that their versions are compatible with the version of the Whisper library you are using.

Reduce Complexity: Try running the program with a simpler configuration. For example, reduce the batch size to 1 or use a different (and potentially simpler) dataset. This can help narrow down whether the problem is due to resource constraints or specific to the dataset.

Check Hardware: Make sure that your GPU and TPU hardware is set up correctly. Ensure that you have correctly set up JAX to work with your hardware. Also, make sure your GPU drivers are up to date.

Debugging: Consider using a debugger to get a better stack trace. This could provide more insight into where exactly the problem is happening.

However, debugging this issue might be difficult without access to your environment, and the issue could be within the libraries you're using.

In such cases, it can be helpful to reach out to the maintainers of the problematic library. If you believe the issue lies within the Whisper library, you could consider creating an issue on the repository where you obtained the Whisper library, providing them with as much detail as possible. They might be able to provide more targeted assistance.

If you're working within a larger team or organization, it might also be worth reaching out to your team members or IT support for help. They might have run into similar issues in the past and have some insights into how to solve them.




User
Carefully consider how the code is used here:
import time

import jax
import jax.numpy as jnp
from datasets import concatenate_datasets, load_dataset
from flax import jax_utils
from flax.training.common_utils import shard
from transformers import FlaxWhisperForConditionalGeneration, WhisperProcessor


BATCH_SIZES = [4, 8, 16, 32, 64, 128]
NUM_BATCHES = 100
NUM_TOKENS = 25

model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v2",
    _do_init=False,
    dtype=jnp.bfloat16,
)

params = jax_utils.replicate(params)


def generate_fn(batch):
    pred_ids = model.generate(batch, params=params, max_new_tokens=NUM_TOKENS, min_new_tokens=NUM_TOKENS)
    return pred_ids.sequences


p_generate_fn = jax.pmap(generate_fn, "batch")

# processors/tokenizers are the same for all models, so just load from tiny and preprocess once
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")


def preprocess(batch):
    batch["input_features"] = processor(
        batch["audio"]["array"], sampling_rate=16000, return_tensors="np"
    ).input_features[0]
    return batch


# load a dataset of 73 audio samples
librispeech = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset_processed = librispeech.map(preprocess, remove_columns=librispeech.column_names)

for batch_size in BATCH_SIZES:
    eval_dataset = dataset_processed.select(range(batch_size // 2))
    eval_dataset = concatenate_datasets([eval_dataset for _ in range(2 * NUM_BATCHES)])

    eval_dataloader = eval_dataset.with_format("numpy").iter(batch_size=batch_size)

    # warm-up step
    batch = next(iter(eval_dataloader))
    input_features = shard(batch["input_features"])
    pred_ids = p_generate_fn(input_features)

    start = time.time()
    for batch in eval_dataloader:
        input_features = shard(batch["input_features"])
        pred_ids = p_generate_fn(input_features)
    runtime = time.time() - start

    print(f"{batch_size}: {runtime:.06}")
ChatGPT
The code is designed to benchmark the transcription speed of different batch sizes using the FlaxWhisperForConditionalGeneration model on the Librispeech ASR dummy dataset. Let's break down the steps:

Setting Parameters: The script sets several parameters at the beginning, such as batch sizes to test, the number of batches to process for each batch size, and the number of tokens to generate.

Loading the Model: The FlaxWhisperForConditionalGeneration model is loaded with the pretrained "openai/whisper-large-v2" model. The data type is set to jnp.bfloat16 for faster computation. The parameters of the model are then replicated across different devices (GPUs/TPUs) using jax_utils.replicate(params).

Creating the Generation Function: A function generate_fn is defined, which takes a batch of input features and generates sequences of predicted IDs (transcriptions) from the model.

Parallelizing the Generation Function: The generate_fn function is parallelized using jax.pmap to enable data parallelism across multiple devices (GPUs/TPUs).

Loading the Processor: The WhisperProcessor is loaded to process the audio data into a format that the model can handle.

Creating the Preprocessing Function: A function preprocess is defined, which takes a batch of audio data and converts it into input features using the WhisperProcessor.

Loading the Dataset: The Librispeech ASR dummy dataset is loaded and preprocessed using the preprocess function.

Benchmarking Different Batch Sizes: For each batch size in BATCH_SIZES, the script selects a subset of the preprocessed dataset and duplicates it to create a larger dataset. It then creates a dataloader that iterates over the dataset in batches of the current batch size. The script warms up by running the generation function on the first batch. It then times how long it takes to run the generation function on all batches in the dataloader. The runtime is printed at the end.

The purpose of this script is to determine the optimal batch size for transcribing audio data using the Whisper model on your specific hardware setup. The goal is to find a balance between using a large batch size for efficient computation and avoiding out-of-memory errors that can occur with very large batch sizes.

The max_new_tokens and min_new_tokens parameters in the generate_fn are set to restrict the length of the generated sequences to a specific number of tokens. This was done to normalize the comparison of inference speed on a per-token basis across different batch sizes. If you are not interested in this normalization, you can remove these parameters.




User
I am also running this code, and is getting: (.whisper) perk@t1v-n-c53b857e-w-0:~/models/nb-whisper/transcribe$ python test.py
/home/perk/.whisper/lib/python3.8/site-packages/tensorflow/python/framework/dtypes.py:40: RuntimeWarning: A cast from dtype(bfloat16) to dtype(float8_e4m3fn) was registered/modified using `PyArray_RegisterCastFunc` after the cast had been used.  This registration will have (mostly) no effect: the previous definition will continue to be used.
The most likely fix is to ensure that casts are the first thing initialized after dtype registration.  Please contact the NumPy developers with any questions!
  _pywrap_custom_casts.TF_register_custom_casts()
/home/perk/.whisper/lib/python3.8/site-packages/tensorflow/python/framework/dtypes.py:40: RuntimeWarning: A cast from dtype(bfloat16) to dtype(float8_e5m2) was registered/modified using `PyArray_RegisterCastFunc` after the cast had been used.  This registration will have (mostly) no effect: the previous definition will continue to be used.
The most likely fix is to ensure that casts are the first thing initialized after dtype registration.  Please contact the NumPy developers with any questions!
  _pywrap_custom_casts.TF_register_custom_casts()
2023-07-30 20:08:30.807500: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
https://symbolize.stripped_domain/r/?trace=7fc2dcb6f0c0,90c85f&map=
*** SIGSEGV (@(nil)), see gl__________50#s15 received by PID 120156 (TID 120156) on cpu 239; stack trace: ***
    @     0x7fc170d260da       1152  (unknown)
    @     0x7fc2dcb6f0c0  (unknown)  (unknown)
    @           0x90c860  (unknown)  (unknown)
https://symbolize.stripped_domain/r/?trace=7fc170d260da,7fc2dcb6f0bf,90c85f&map=1db795edf6ed53794bfb4c33fb861dd9:7fc165e82000-7fc170f38dc0
E0730 20:08:31.492497  120156 coredump_hook.cc:408] RAW: Remote crash data gathering hook invoked.
E0730 20:08:31.492507  120156 coredump_hook.cc:447] RAW: Skipping coredump since rlimit was 0 at process start.
E0730 20:08:31.492513  120156 client.cc:281] RAW: Coroner client retries enabled (b/136286901), will retry for up to 30 sec.
E0730 20:08:31.492516  120156 coredump_hook.cc:506] RAW: Sending fingerprint to remote end.
E0730 20:08:31.492522  120156 coredump_socket.cc:117] RAW: Stat failed errno=2 on socket /var/google/services/logmanagerd/remote_coredump.socket
E0730 20:08:31.492526  120156 coredump_hook.cc:514] RAW: Cannot send fingerprint to Coroner: [NOT_FOUND] Missing crash reporting socket. Is the listener running?
E0730 20:08:31.492528  120156 coredump_hook.cc:566] RAW: Dumping core locally.
E0730 20:08:31.841656  120156 process_state.cc:784] RAW: Raising signal 11 with default behavior
Segmentation fault (core dumped)
ChatGPT
The error message indicates a segmentation fault (SIGSEGV), which is usually caused by a program trying to read or write an illegal memory location. This might be due to an issue with TensorFlow, JAX, or the Whisper library, or an interaction between them.

The error messages also show that there were issues with TensorFlow's TensorRT (TF-TRT), and with the casting from bfloat16 to other data types.

Here are a few things you could try to resolve this issue:

Verify Dependencies: Make sure that all of the libraries and their dependencies are correctly installed and that their versions are compatible with each other.

Update Libraries: Try updating your libraries (TensorFlow, JAX, Flax, Hugging Face Transformers, etc.) to the latest versions.

Check the Data: Make sure the input data is correctly formatted and does not contain any unexpected values that might cause issues.

Debugging: If possible, use a debugger to inspect the state of the program at the time of the segmentation fault. This could help you pinpoint the exact location of the issue.

Contact Support: Since the error seems to be related to the Whisper library or its dependencies, you may want to contact the maintainers of the Whisper library for assistance, providing them with as much detail as possible. They may be able to provide more targeted assistance.

As a final note, please remember that TensorFlow's TensorRT (TF-TRT) is a separate component from TensorFlow and needs to be installed separately. If you're not using TensorRT and this warning doesn't affect your program, you can ignore it. If you are using TensorRT, make sure it's correctly installed and configured.




User
Currently running this code:
import time

import jax
import jax.numpy as jnp
from datasets import concatenate_datasets, load_dataset
from flax import jax_utils
from flax.training.common_utils import shard
from transformers import WhisperProcessor
from whisper_jax import FlaxWhisperForConditionalGeneration


BATCH_SIZES = [4, 8, 16, 32, 64, 128]
NUM_BATCHES = 100
NUM_TOKENS = 25

model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v2", dtype=jnp.bfloat16, _do_init=False,
)

params = jax_utils.replicate(params)


def generate_fn(batch):
    pred_ids = model.generate(batch, params=params, max_new_tokens=NUM_TOKENS, min_new_tokens=NUM_TOKENS)
    return pred_ids.sequences


p_generate_fn = jax.pmap(generate_fn, "batch")

# processors/tokenizers are the same for all models, so just load from tiny and preprocess once
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")


def preprocess(batch):
    batch["input_features"] = processor(
        batch["audio"]["array"], sampling_rate=16000, return_tensors="np"
    ).input_features[0]
    return batch


# load a dataset of 73 audio samples
librispeech = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset_processed = librispeech.map(preprocess, remove_columns=librispeech.column_names)

for batch_size in BATCH_SIZES:
    eval_dataset = dataset_processed.select(range(batch_size // 2))
    eval_dataset = concatenate_datasets([eval_dataset for _ in range(2 * NUM_BATCHES)])

    eval_dataloader = eval_dataset.with_format("numpy").iter(batch_size=batch_size)

    # warm-up step
    batch = next(iter(eval_dataloader))
    input_features = shard(batch["input_features"])
    pred_ids = p_generate_fn(input_features)

    start = time.time()
    for batch in eval_dataloader:
        input_features = shard(batch["input_features"])
        pred_ids = p_generate_fn(input_features)
    runtime = time.time() - start

    print(f"{batch_size}: {runtime:.06}")

Getting this error:
╭─────────────────────────────── Traceback (most recent call last) ────────────────────────────────╮
│ /home/perk/models/nb-whisper/transcribe/transcribe.py:55 in <module>                             │
│                                                                                                  │
│   52 # Warm-up step                                                                              │
│   53 batch = next(iter(eval_dataloader))                                                         │
│   54 input_features = shard(batch["input_features"])                                             │
│ ❱ 55 pred_ids = p_generate_fn(input_features)                                                    │
│   56                                                                                             │
│   57 # Run the model on the batches                                                              │
│   58 start = time.time()                                                                         │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/jax/_src/traceback_util.py:166 in                │
│ reraise_with_filtered_traceback                                                                  │
│                                                                                                  │
│   163   def reraise_with_filtered_traceback(*args, **kwargs):                                    │
│   164 │   __tracebackhide__ = True                                                               │
│   165 │   try:                                                                                   │
│ ❱ 166 │     return fun(*args, **kwargs)                                                          │
│   167 │   except Exception as e:                                                                 │
│   168 │     mode = _filtering_mode()                                                             │
│   169 │     if _is_under_reraiser(e) or mode == "off":                                           │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/jax/_src/api.py:1860 in cache_miss               │
│                                                                                                  │
│   1857 │   │   │   │   │   │   │   │   │   │   *p.flat_args, **params))                          │
│   1858 │   execute: Optional[Callable] = None                                                    │
│   1859 │   if isinstance(top_trace, core.EvalTrace):                                             │
│ ❱ 1860 │     execute = pxla.xla_pmap_impl_lazy(fun_, *tracers, **params)                         │
│   1861 │     out = map_bind_continuation(execute(*tracers))                                      │
│   1862 │   else:                                                                                 │
│   1863 │     out = map_bind_continuation(                                                        │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/jax/_src/interpreters/pxla.py:761 in             │
│ xla_pmap_impl_lazy                                                                               │
│                                                                                                  │
│    758 │   │   │   │   │   │   is_explicit_global_axis_size=is_explicit_global_axis_size)        │
│    759 │   return _emap_apply_fn                                                                 │
│    760   abstract_args = unsafe_map(xla.abstractify, args)                                       │
│ ❱  761   compiled_fun, fingerprint = parallel_callable(                                          │
│    762 │     fun, backend, axis_name, axis_size, global_axis_size, devices, name,                │
│    763 │     in_axes, out_axes_thunk, donated_invars,                                            │
│    764 │     is_explicit_global_axis_size, *abstract_args)
ChatGPT
The error you're experiencing is within JAX's parallel mapping function (jax.pmap). It's likely that an issue occurred while trying to parallelize your generate_fn function across multiple devices.

Unfortunately, the provided traceback does not include the specific error message, which makes it more difficult to diagnose the issue. Here are some steps you could take to troubleshoot this:

Check the Devices: Use jax.devices() to print out the list of devices JAX is trying to use. Make sure that these devices are available and functioning correctly.

Check the Inputs: Make sure that the inputs to p_generate_fn are correctly formed and that their shapes are compatible with the expected inputs of the generate_fn function.

Simplify the Function: Try simplifying the generate_fn function to a simpler operation (like adding a constant to the inputs) and see if p_generate_fn works with this simpler function. If it does, then the issue is likely with how generate_fn interacts with p_generate_fn, and you can gradually add complexity back to generate_fn to see what causes the issue.

Check the Batch Size: The batch size should not exceed the number of devices available. Try reducing the batch size or the number of devices used in jax.pmap.

Debugging: Use a debugger or add print statements to the generate_fn function to see if it is executing correctly.

If these steps do not resolve the issue, there may be a bug in JAX, Flax, or the Whisper library. In this case, you might want to reach out to the maintainers of these libraries for further assistance. Remember to provide them with as much detail as possible, including the full error message and traceback, the versions of the libraries you're using, and a minimal reproducible example of your code, if possible.




User
/home/perk/.whisper/lib/python3.8/site-packages/jax/_src/interpreters/pxla.py:761 in             │
│ xla_pmap_impl_lazy                                                                               │
│                                                                                                  │
│    758 │   │   │   │   │   │   is_explicit_global_axis_size=is_explicit_global_axis_size)        │
│    759 │   return _emap_apply_fn                                                                 │
│    760   abstract_args = unsafe_map(xla.abstractify, args)                                       │
│ ❱  761   compiled_fun, fingerprint = parallel_callable(                                          │
│    762 │     fun, backend, axis_name, axis_size, global_axis_size, devices, name,                │
│    763 │     in_axes, out_axes_thunk, donated_invars,                                            │
│    764 │     is_explicit_global_axis_size, *abstract_args)                                       │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/jax/_src/linear_util.py:322 in memoized_fun      │
│                                                                                                  │
│   319 │     ans, stores = result                                                                 │
│   320 │     fun.populate_stores(stores)                                                          │
│   321 │   else:                                                                                  │
│ ❱ 322 │     ans = call(fun, *args)                                                               │
│   323 │     cache[key] = (ans, fun.stores)                                                       │
│   324 │                                                                                          │
│   325 │   return ans                                                                             │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/jax/_src/interpreters/pxla.py:1028 in            │
│ parallel_callable                                                                                │
│                                                                                                  │
│   1025 │   │   │   │   │     donated_invars: Sequence[bool],                                     │
│   1026 │   │   │   │   │     is_explicit_global_axis_size: bool,                                 │
│   1027 │   │   │   │   │     *avals):                                                            │
│ ❱ 1028   pmap_computation = lower_parallel_callable(                                             │
│   1029 │     fun, backend_name, axis_name, axis_size, global_axis_size, devices, name,           │
│   1030 │     in_axes, out_axes_thunk, donated_invars,                                            │
│   1031 │     is_explicit_global_axis_size, avals, lowering_platform=None)                        │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/jax/_src/profiler.py:314 in wrapper              │
│                                                                                                  │
│   311   @wraps(func)                                                                             │
│   312   def wrapper(*args, **kwargs):                                                            │
│   313 │   with TraceAnnotation(name, **decorator_kwargs):                                        │
│ ❱ 314 │     return func(*args, **kwargs)                                                         │
│   315 │   return wrapper                                                                         │
│   316   return wrapper                                                                           │
│   317                                                                                            │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/jax/_src/interpreters/pxla.py:1183 in            │
│ lower_parallel_callable                                                                          │
│                                                                                                  │
│   1180   pci = ParallelCallableInfo(                                                             │
│   1181 │     name, backend, axis_name, axis_size, global_axis_size, devices,                     │
│   1182 │     in_axes, out_axes_thunk, avals)                                                     │
│ ❱ 1183   jaxpr, consts, replicas, parts, shards = stage_parallel_callable(pci, fun)              │
│   1184   if logger.isEnabledFor(logging.DEBUG):                                                  │
│   1185 │   logger.debug("sharded_avals: %s", shards.sharded_avals)                               │
│   1186 │   logger.debug("global_sharded_avals: %s", shards.global_sharded_avals)                 │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/jax/_src/interpreters/pxla.py:1096 in            │
│ stage_parallel_callable                                                                          │
│                                                                                                  │
│   1093 │   with dispatch.log_elapsed_time(f"Finished tracing + transforming {fun.__name__} "     │
│   1094 │   │   │   │   │   │   │   │      "for pmap in {elapsed_time} sec",                      │
│   1095 │   │   │   │   │   │   │   │      event=dispatch.JAXPR_TRACE_EVENT):                     │
│ ❱ 1096 │     jaxpr, out_sharded_avals, consts = pe.trace_to_jaxpr_final(                         │
│   1097 │   │     fun, sharded_avals, pe.debug_info_final(fun, "pmap"))                           │
│   1098   jaxpr = api_util.jaxpr_debug_info(jaxpr, fun.debug_info)                                │
│   1099   jaxpr = dispatch.apply_outfeed_rewriter(jaxpr)                                          │
/home/perk/.whisper/lib/python3.8/site-packages/jax/_src/profiler.py:314 in wrapper              │                                                 [54/1176]
│                                                                                                  │
│   311   @wraps(func)                                                                             │
│   312   def wrapper(*args, **kwargs):                                                            │
│   313 │   with TraceAnnotation(name, **decorator_kwargs):                                        │
│ ❱ 314 │     return func(*args, **kwargs)                                                         │
│   315 │   return wrapper                                                                         │
│   316   return wrapper                                                                           │
│   317                                                                                            │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/jax/_src/interpreters/partial_eval.py:2119 in    │
│ trace_to_jaxpr_final                                                                             │
│                                                                                                  │
│   2116   with core.new_base_main(DynamicJaxprTrace) as main:  # type: ignore                     │
│   2117 │   main.jaxpr_stack = ()  # type: ignore                                                 │
│   2118 │   with core.new_sublevel():                                                             │
│ ❱ 2119 │     jaxpr, out_avals, consts = trace_to_subjaxpr_dynamic(                               │
│   2120 │   │   fun, main, in_avals, keep_inputs=keep_inputs, debug_info=debug_info)              │
│   2121 │   del fun, main                                                                         │
│   2122   return jaxpr, out_avals, consts                                                         │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/jax/_src/interpreters/partial_eval.py:2066 in    │
│ trace_to_subjaxpr_dynamic                                                                        │
│                                                                                                  │
│   2063 │   trace = DynamicJaxprTrace(main, core.cur_sublevel())                                  │
│   2064 │   in_tracers = _input_type_to_tracers(trace.new_arg, in_avals)                          │
│   2065 │   in_tracers_ = [t for t, keep in zip(in_tracers, keep_inputs) if keep]                 │
│ ❱ 2066 │   ans = fun.call_wrapped(*in_tracers_)                                                  │
│   2067 │   out_tracers = map(trace.full_raise, ans)                                              │
│   2068 │   jaxpr, consts = frame.to_jaxpr(out_tracers)                                           │
│   2069 │   del fun, main, trace, frame, in_tracers, out_tracers, ans                             │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/jax/_src/linear_util.py:166 in call_wrapped      │
│                                                                                                  │
│   163 │   gen = gen_static_args = out_store = None                                               │
│   164 │                                                                                          │
│   165 │   try:                                                                                   │
│ ❱ 166 │     ans = self.f(*args, **dict(self.params, **kwargs))                                   │
│   167 │   except:                                                                                │
│   168 │     # Some transformations yield from inside context managers, so we have to             │
│   169 │     # interrupt them before reraising the exception. Otherwise they will only            │
│                                                                                                  │
│ /home/perk/models/nb-whisper/transcribe/transcribe.py:27 in generate_fn                          │
│                                                                                                  │
│   24                                                                                             │
│   25 # Function to generate predictions                                                          │
│   26 def generate_fn(batch):                                                                     │
│ ❱ 27 │   pred_ids = model.generate(batch, params=params)                                         │
│   28 │   return pred_ids.sequences                                                               │
│   29                                                                                             │
│   30 # Map the generate function to multiple devices                                             │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/whisper_jax/modeling_flax_whisper.py:1585 in     │
│ generate                                                                                         │
│                                                                                                  │
│   1582 │   │   if len(forced_decoder_ids) > 0:                                                   │
│   1583 │   │   │   generation_config.forced_decoder_ids = forced_decoder_ids                     │
│   1584 │   │                                                                                     │
│ ❱ 1585 │   │   return super().generate(                                                          │
│   1586 │   │   │   input_features,                                                               │
│   1587 │   │   │   generation_config,                                                            │
│   1588 │   │   │   logits_processor=logits_processor,                                            │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/transformers/generation/flax_utils.py:363 in     │
│ generate                                                                                         │
│                                                                                                  │
│    360 │   │   │   │   │   "generation results, please set `padding_side='left'` when initializ  │
│    361 │   │   │   │   )                                                                         │
│    362 │   │                                                                                     │
│ ❱  363 │   │   batch_size = input_ids.shape[0]                                                   │
│    364 │   │                                                                                     │
│    365 │   │   if self.config.is_encoder_decoder:                                                │
│    366 │   │   │   # add encoder_outputs to model_kwargs                                         │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
UnfilteredStackTrace: AttributeError: 'list' object has no attribute 'shape'

The stack trace below excludes JAX-internal frames.
The preceding is the original exception that occurred, unmodified.
ChatGPT
The error message indicates that the variable input_ids is a list, and the code is trying to access its shape attribute, which does not exist because shape is a property of arrays, not lists.

The input_ids variable is an input to the generate method of a FlaxWhisperForConditionalGeneration model. The error occurs because the generate method expects input_ids to be an array (specifically a JAX or NumPy array), but a list is being passed instead. This discrepancy is likely due to a mismatch between the type of data the generate_fn function is producing and what the generate method expects.

In your generate_fn function, you are directly passing the batch argument to the model's generate method:

python
Copy code
def generate_fn(batch):
    pred_ids = model.generate(batch, params=params, max_new_tokens=NUM_TOKENS, min_new_tokens=NUM_TOKENS)
    return pred_ids.sequences
However, it seems that batch is a list, while the generate method is expecting an array. To fix this, you might need to convert batch to an array before passing it to generate. You could use JAX's array function for this purpose:

python
Copy code
def generate_fn(batch):
    batch = jnp.array(batch)
    pred_ids = model.generate(batch, params=params, max_new_tokens=NUM_TOKENS, min_new_tokens=NUM_TOKENS)
    return pred_ids.sequences
Please try this modification and see if it resolves the error. If the issue persists, there could be other mismatches between the data types and formats your code is using and what the Whisper model and the JAX library expect. You might need to print and examine your data at various points in your code to debug this further.




User
This is on line 26, the program seem to crash on line 55
ChatGPT
I see, the error occurs at this line:

python
Copy code
pred_ids = p_generate_fn(input_features)
This line is calling the p_generate_fn function, which is a parallelized version of your generate_fn function. This function is applied to the input_features from your input batch. The error message suggests that a list is being passed to the generate function of your model instead of a JAX array.

The shard function, which you use to transform batch["input_features"] into input_features, should return a list of arrays, one for each device. This is a standard way of preparing data for parallel computation in JAX.

So, the problem might be in how generate_fn is handling its input. The pmap function applies generate_fn to each device's slice of the input data. Therefore, generate_fn receives a single device's portion of the data as input, not the whole dataset. If generate_fn is treating its input as if it were the whole dataset, this could be causing the error.

Here's how you might modify generate_fn to handle its inputs correctly:

python
Copy code
def generate_fn(input_features):
    pred_ids = model.generate(input_features, params=params, max_new_tokens=NUM_TOKENS, min_new_tokens=NUM_TOKENS)
    return pred_ids.sequences
In this version, generate_fn expects to receive a single device's slice of input_features as input. When p_generate_fn is called, it applies generate_fn to each device's slice of the data in parallel.

If you're still having trouble, it would be helpful to add some print statements to your code to check the shapes and types of your data at various points. This can help you ensure that your data is being correctly partitioned for parallel computation.




User
NameError: name 'NUM_TOKENS' is not defined
ChatGPT
The error message "NameError: name 'NUM_TOKENS' is not defined" indicates that the variable NUM_TOKENS is not defined in the scope where it's being used.

In your script, NUM_TOKENS is defined at the top of the file, so it should be available in the generate_fn function. Make sure that you have defined NUM_TOKENS before you use it in the function.

Here is the correct order:

python
Copy code
BATCH_SIZES = [4, 8, 16, 32, 64, 128]
NUM_BATCHES = 100
NUM_TOKENS = 25

def generate_fn(input_features):
    pred_ids = model.generate(input_features, params=params, max_new_tokens=NUM_TOKENS, min_new_tokens=NUM_TOKENS)
    return pred_ids.sequences
If you are running the function in a different file or a different context, ensure that you have imported or defined all the necessary variables.




User
what does num_tokens do here?
ChatGPT
The NUM_TOKENS parameter in this context is being used to specify the number of tokens that the model should generate. Specifically, the max_new_tokens and min_new_tokens parameters in the generate function are set to NUM_TOKENS, meaning that the model will generate a sequence with exactly NUM_TOKENS tokens.

The generate function is typically used in the context of language models, which generate sequences of tokens. These tokens can represent words, subwords, or characters depending on the granularity of the model's vocabulary.

In this case, the script is testing how long it takes to generate sequences of different lengths, and NUM_TOKENS is a parameter that controls the length of these sequences. By changing NUM_TOKENS, you can measure how the time it takes to generate a sequence scales with the length of that sequence.




User
I am not interesting in meassuring this. I am interested in transcribing only
ChatGPT
In that case, you don't need to worry about the NUM_TOKENS variable or the loop over BATCH_SIZES. The important parts of the script for transcription are:

Loading the pre-trained model with FlaxWhisperForConditionalGeneration.from_pretrained(...)
Preprocessing the input data using the WhisperProcessor and mapping this over the dataset with dataset_processed = librispeech.map(preprocess, remove_columns=librispeech.column_names)
Generating predictions with p_generate_fn(input_features)
You would typically use the predictions to transcribe the input data. For example, you might decode the predictions into text using a vocabulary that maps token IDs to words or characters. This will depend on the specifics of the Whisper model and how it was trained.

The other parts of the script are there for benchmarking the performance of the model under different conditions, and you can safely ignore them if you're not interested in that.




User
Here is the current code. Make the necessary adjustments.
import time

import jax
import jax.numpy as jnp
from datasets import concatenate_datasets, load_dataset
from flax import jax_utils
from flax.training.common_utils import shard
from transformers import WhisperProcessor
from whisper_jax import FlaxWhisperForConditionalGeneration


# Set your BATCH_SIZE according to your GPU's memory availability
BATCH_SIZE = 16
NUM_BATCHES = 5
NUM_TOKENS = 25

# Load the model and replicate the parameters across devices
model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-tiny",
    _do_init=False,
    dtype=jnp.bfloat16,
)

params = jax_utils.replicate(params)

# Function to generate predictions
def generate_fn(input_features):
    pred_ids = model.generate(input_features, params=params, max_new_tokens=NUM_TOKENS, min_new_tokens=NUM_TOKENS)
    return pred_ids.sequences


# Map the generate function to multiple devices
p_generate_fn = jax.pmap(generate_fn, "batch")

# Load the WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

# Function to preprocess the audio data
def preprocess(batch):
    batch["input_features"] = processor(
        batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"], return_tensors="np"
    ).input_features[0]
    return batch

# Provide your Hugging Face token here
audio_files_dataset = load_dataset("NbAiLab/ncc_speech_v3", split="train", use_auth_token=True, streaming=True)

# Preprocess the audio data
dataset_processed = audio_files_dataset.map(preprocess, remove_columns=audio_files_dataset.column_names)

# Create a dataloader
eval_dataloader = dataset_processed.with_format("numpy").iter(batch_size=BATCH_SIZE)

# Warm-up step
batch = next(iter(eval_dataloader))
input_features = shard(batch["input_features"])
pred_ids = p_generate_fn(input_features)

# Run the model on the batches
start = time.time()
for i, batch in enumerate(eval_dataloader):
    input_features = shard(batch["input_features"])
    pred_ids = p_generate_fn(input_features)
    
    # Post-process: convert tokens ids to text string
    transcriptions = processor.batch_decode(jax.device_get(pred_ids.reshape(-1, model.config.max_length)), skip_special_tokens=True)
    
    # Print the transcriptions
    for transcription in transcriptions:
        print(transcription)
    
    # Exit after processing NUM_BATCHES batches
    if i >= NUM_BATCHES - 1:
        break

runtime = time.time() - start
print(f"\nTotal time for {NUM_BATCHES} batches: {runtime:.06}")
ChatGPT
The code you provided is almost correct for your intended purpose. However, there are a few points that need to be addressed:

The model "openai/whisper-tiny" is not a real model, it's just used in the example script. You should replace it with a real Whisper model, for example, "facebook/wav2vec2-large-960h".

The dataset "NbAiLab/ncc_speech_v3" is not a known public dataset (as of my knowledge cutoff in September 2021). You should replace it with a dataset that you have access to and that is compatible with the Whisper model you're using.

The WhisperProcessor's from_pretrained function needs to be supplied with a real processor.

Given these considerations, the revised code might look like this:

python
Copy code
import time

import jax
import jax.numpy as jnp
from datasets import load_dataset
from flax import jax_utils
from flax.training.common_utils import shard
from transformers import FlaxWav2Vec2ForCTC, Wav2Vec2Processor

# Set your BATCH_SIZE according to your GPU's memory availability
BATCH_SIZE = 16
NUM_BATCHES = 5

# Load the model and replicate the parameters across devices
model = FlaxWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
params = jax_utils.replicate(model.params)

# Function to generate predictions
def generate_fn(input_features):
    logits = model(input_features, params=params, train=False).logits
    pred_ids = jnp.argmax(logits, axis=-1)
    return pred_ids

# Map the generate function to multiple devices
p_generate_fn = jax.pmap(generate_fn, "batch")

# Load the Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

# Function to preprocess the audio data
def preprocess(batch):
    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"]).input_values
    return batch

# Load your dataset here
speech_dataset = load_dataset("your-dataset", split="train")

# Preprocess the audio data
dataset_processed = speech_dataset.map(preprocess, remove_columns=speech_dataset.column_names)

# Create a dataloader
eval_dataloader = dataset_processed.with_format("numpy").iter(batch_size=BATCH_SIZE)

# Run the model on the batches
start = time.time()
for i, batch in enumerate(eval_dataloader):
    input_values = shard(batch["input_values"])
    pred_ids = p_generate_fn(input_values)
    
    # Post-process: convert tokens ids to text string
    transcriptions = processor.batch_decode(jax.device_get(pred_ids).tolist(), group_tokens=False)
    
    # Print the transcriptions
    for transcription in transcriptions:
        print(transcription)
    
    # Exit after processing NUM_BATCHES batches
    if i >= NUM_BATCHES - 1:
        break

runtime = time.time() - start
print(f"\nTotal time for {NUM_BATCHES} batches: {runtime:.06}")
Please replace "your-dataset" with your actual dataset. Make sure to replace "facebook/wav2vec2-large-960h" with the model you wish to use for transcription.

Also, please note that the Whisper model and the WhisperProcessor are not currently available in the transformers library (as of my knowledge cutoff in September 2021). You'll need to replace them with the actual model and processor you want to use. In the example above, I've used the FlaxWav2Vec2ForCTC model and Wav2Vec2Processor processor as an example. This is a model for Automatic Speech Recognition (ASR) that takes audio data as input and outputs text.

Please make sure to adjust the code to your actual use case. If the Whisper models become available, you would need to replace FlaxWav2Vec2ForCTC and Wav2Vec2Processor with the corresponding Whisper classes.




User
Assume the FlaxWhisper exists, and the datasets and models are real (after 2021). Here is the current code:
import time

import jax
import jax.numpy as jnp
from datasets import concatenate_datasets, load_dataset
from flax import jax_utils
from flax.training.common_utils import shard
from transformers import WhisperProcessor
from whisper_jax import FlaxWhisperForConditionalGeneration


# Set your BATCH_SIZE according to your GPU's memory availability
BATCH_SIZE = 16
NUM_BATCHES = 5
NUM_TOKENS = 25

# Load the model and replicate the parameters across devices
model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-tiny.en",
    _do_init=False,
    dtype=jnp.bfloat16,
)

params = jax_utils.replicate(params)

# Function to generate predictions
def generate_fn(input_features):
    pred_ids = model.generate(input_features, params=params)
    return pred_ids.sequences


# Map the generate function to multiple devices
p_generate_fn = jax.pmap(generate_fn, "batch")

# Load the WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

# Function to preprocess the audio data
def preprocess(batch):
    batch["input_features"] = processor(
        batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"], return_tensors="np"
    ).input_features[0]
    return batch

# Provide your Hugging Face token here
audio_files_dataset = load_dataset("NbAiLab/ncc_speech_v3", split="train", use_auth_token=True, streaming=True)

# Preprocess the audio data
dataset_processed = audio_files_dataset.map(preprocess, remove_columns=audio_files_dataset.column_names)

# Create a dataloader
eval_dataloader = dataset_processed.with_format("numpy").iter(batch_size=BATCH_SIZE)

# Warm-up step
batch = next(iter(eval_dataloader))
input_features = shard(batch["input_features"])
pred_ids = p_generate_fn(input_features)

# Run the model on the batches
start = time.time()
for i, batch in enumerate(eval_dataloader):
    input_features = shard(batch["input_features"])
    pred_ids = p_generate_fn(input_features)
    
    # Post-process: convert tokens ids to text string
    transcriptions = processor.batch_decode(jax.device_get(pred_ids.reshape(-1, model.config.max_length)), skip_special_tokens=True)
    
    # Print the transcriptions
    for transcription in transcriptions:
        print(transcription)
    
    # Exit after processing NUM_BATCHES batches
    if i >= NUM_BATCHES - 1:
        break

runtime = time.time() - start
print(f"\nTotal time for {NUM_BATCHES} batches: {runtime:.06}")
ChatGPT
The provided code seems appropriate for running a batch transcription task with a model using the OpenAI Whisper architecture, assuming that the Whisper models and WhisperProcessor are available and correctly implemented.

Here are the main steps the code performs:

It loads a Whisper model and replicates the parameters across available devices.

It defines a function (generate_fn) to generate predictions from input features using the loaded model.

It prepares this function for parallel execution on multiple devices using jax.pmap.

It loads a WhisperProcessor for preparing the input data and decoding the model outputs.

It defines a function (preprocess) for preprocessing the audio data.

It loads an audio dataset and applies the preprocessing function to it.

It creates a data loader that will yield batches of preprocessed data from the dataset.

It runs a warm-up step, generating predictions from the first batch of data.

It loops over the batches of data from the data loader, generating predictions for each batch, decoding the predictions into transcriptions, and printing the transcriptions.

It stops processing batches after a certain number of batches have been processed (NUM_BATCHES).

It prints the total time taken to process the batches.

Remember to replace "openai/whisper-tiny.en" with the actual model you're planning to use. Also, replace "NbAiLab/ncc_speech_v3" with your actual dataset. Finally, make sure to handle your use_auth_token correctly to access your private datasets or models.

Please note that the FlaxWhisperForConditionalGeneration and WhisperProcessor are assumed to be correctly implemented and available in the whisper_jax and transformers libraries, respectively. As of my last update in September 2021, this might not be the case, but it could have changed afterwards.




User
I am getting this error:
UnfilteredStackTrace: AttributeError: 'list' object has no attribute 'shape'

The stack trace below excludes JAX-internal frames.
The preceding is the original exception that occurred, unmodified.

--------------------

The above exception was the direct cause of the following exception:

╭─────────────────────────────── Traceback (most recent call last) ────────────────────────────────╮
│ /home/perk/models/nb-whisper/transcribe/transcribe.py:57 in <module>                             │
│                                                                                                  │
│   54 # Warm-up step                                                                              │
│   55 batch = next(iter(eval_dataloader))                                                         │
│   56 input_features = shard(batch["input_features"])                                             │
│ ❱ 57 pred_ids = p_generate_fn(input_features)                                                    │
│   58                                                                                             │
│   59 # Run the model on the batches                                                              │
│   60 start = time.time()                                                                         │
│                                                                                                  │
│ /home/perk/models/nb-whisper/transcribe/transcribe.py:28 in generate_fn                          │
│                                                                                                  │
│   25                                                                                             │
│   26 # Function to generate predictions                                                          │
│   27 def generate_fn(input_features):                                                            │
│ ❱ 28 │   pred_ids = model.generate(input_features, params=params)                                │
│   29 │   return pred_ids.sequences                                                               │
│   30                                                                                             │
│   31                                                                                             │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/whisper_jax/modeling_flax_whisper.py:1585 in     │
│ generate                                                                                         │
│                                                                                                  │
│   1582 │   │   if len(forced_decoder_ids) > 0:                                                   │
│   1583 │   │   │   generation_config.forced_decoder_ids = forced_decoder_ids                     │
│   1584 │   │                                                                                     │
│ ❱ 1585 │   │   return super().generate(                                                          │
│   1586 │   │   │   input_features,                                                               │
│   1587 │   │   │   generation_config,                                                            │
│   1588 │   │   │   logits_processor=logits_processor,                                            │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/transformers/generation/flax_utils.py:363 in     │
│ generate                                                                                         │
│                                                                                                  │
│    360 │   │   │   │   │   "generation results, please set `padding_side='left'` when initializ  │
│    361 │   │   │   │   )                                                                         │
│    362 │   │                                                                                     │
│ ❱  363 │   │   batch_size = input_ids.shape[0]                                                   │
│    364 │   │                                                                                     │
│    365 │   │   if self.config.is_encoder_decoder:                                                │
│    366 │   │   │   # add encoder_outputs to model_kwargs                                         │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
AttributeError: 'list' object has no attribute 'shape'
ChatGPT
This error seems to be occurring because the input to the generate function of the model is a list and the generate function is expecting a jax.numpy array (or a similar array-like object) which has the shape attribute.

Looking at the stack trace, this issue arises in transformers/generation/flax_utils.py line 363, where it tries to access the shape attribute of input_ids.

Here's the relevant part of the code:

python
Copy code
batch_size = input_ids.shape[0]
It seems like input_ids is a list at this point, and hence it doesn't have a shape attribute, causing an AttributeError.

To resolve this, you need to ensure that the input_features passed to the generate_fn function is an array-like object rather than a list. The shard function from flax.training.common_utils should return a ShardedDeviceArray, which is array-like and should have a shape attribute.

If input_features is a list of arrays (one for each device), you might need to convert it to a single array before passing it to generate_fn. However, this depends on the exact data structures and types involved, and without more context about the types of the variables in your code, it's difficult to provide a more precise solution. I recommend checking the type and structure of input_features before it's passed to generate_fn and making sure it's what the generate function expects.




User
Lets try with a smaller dataset.

I have changed the code to:
# audio_files_dataset = load_dataset("NbAiLab/ncc_speech_v3", split="train", use_auth_token=True, streaming=True)
audio_files_dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

Note that this is not a streaming dataset.

I am now getting this error:
│ ❱  368 │   │   │   │   model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input  │
│    369 │   │   │   # prepare decoder_input_ids for generation                                    │
│    370 │   │   │   input_ids = self._prepare_decoder_input_ids_for_generation(                   │
│    371 │   │   │   │   batch_size,                                                               │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/transformers/generation/flax_utils.py:167 in     │
│ _prepare_encoder_decoder_kwargs_for_generation                                                   │
│                                                                                                  │
│    164 │   │   │   for argument, value in model_kwargs.items()                                   │
│    165 │   │   │   if not (argument.startswith("decoder_") or argument.startswith("cross_attn")  │
│    166 │   │   }                                                                                 │
│ ❱  167 │   │   model_kwargs["encoder_outputs"] = self.encode(input_ids, params=params, return_d  │
│    168 │   │   return model_kwargs                                                               │
│    169 │                                                                                         │
│    170 │   def _prepare_decoder_input_ids_for_generation(                                        │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/whisper_jax/modeling_flax_whisper.py:1129 in     │
│ encode                                                                                           │
│                                                                                                  │
│   1126 │   │   │   encode_module = module._get_encoder_module()                                  │
│   1127 │   │   │   return encode_module(input_features, **kwargs)                                │
│   1128 │   │                                                                                     │
│ ❱ 1129 │   │   return self.module.apply(                                                         │
│   1130 │   │   │   {"params": params or self.params},                                            │
│   1131 │   │   │   input_features=jnp.array(input_features, dtype="f4"),                         │
│   1132 │   │   │   output_attentions=output_attentions,                                          │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/whisper_jax/modeling_flax_whisper.py:1127 in     │
│ _encoder_forward                                                                                 │
│                                                                                                  │
│   1124 │   │                                                                                     │
│   1125 │   │   def _encoder_forward(module, input_features, **kwargs):                           │
│   1126 │   │   │   encode_module = module._get_encoder_module()                                  │
│ ❱ 1127 │   │   │   return encode_module(input_features, **kwargs)                                │
│   1128 │   │                                                                                     │
│   1129 │   │   return self.module.apply(                                                         │
│   1130 │   │   │   {"params": params or self.params},                                            │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/whisper_jax/modeling_flax_whisper.py:824 in      │
│ __call__                                                                                         │
│                                                                                                  │
│    821 │   │   │   )                                                                             │
│    822 │   │                                                                                     │
│    823 │   │   input_features = input_features.transpose(0, 2, 1)                                │
│ ❱  824 │   │   hidden_states = jax.nn.gelu(self.conv1(input_features), approximate=False)        │
│    825 │   │   hidden_states = with_sharding_constraint(hidden_states, ("batch", "embed", "num_  │
│    826 │   │   hidden_states = jax.nn.gelu(self.conv2(hidden_states), approximate=False)         │
│    827 │   │   hidden_states = with_sharding_constraint(hidden_states, ("batch", "length", "emb  │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/whisper_jax/layers.py:1205 in __call__           │
│                                                                                                  │
│   1202 │   │   │   │   "Mask needs to have the same shape as weights. " f"Shapes are: {self.mas  │
│   1203 │   │   │   )                                                                             │
│   1204 │   │                                                                                     │
│ ❱ 1205 │   │   kernel = param_with_axes(                                                         │
│   1206 │   │   │   "kernel",                                                                     │
│   1207 │   │   │   self.kernel_init,                                                             │
│   1208 │   │   │   kernel_shape,                                                                 │
│                                                                                                  │
│ /home/perk/.whisper/lib/python3.8/site-packages/flax/linen/partitioning.py:145 in                │
│ param_with_axes                                                                                  │
│                                                                                                  │
│   142 │   module = nn.module._context.module_stack[-1]  # pylint: disable=protected-access       │
│   143 │   assert module is not None                                                              │
│   144   # define/fetch parameter on that module                                                  │
│ ❱ 145   module_param = module.param(name, init_fn, *init_args)                                   │
│   146   if axes is not None:                                                                     │
│   147 │   # apply logical axis constraint immediately                                            │
│   148 │   module_param = with_sharding_constraint(module_param,                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
ScopeParamShapeError: Initializer expected to generate shape (4, 3, 80, 384) but got shape (3, 80, 384) instead for parameter "kernel" in
"/model/encoder/conv1". (https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.ScopeParamShapeError)
ChatGPT
The error message indicates that the shape of the kernel parameter in the model's first encoder convolution layer ("/model/encoder/conv1") does not match the expected shape. The initializer function provided to this layer is supposed to generate a tensor of shape (4, 3, 80, 384), but it's instead generating a tensor of shape (3, 80, 384).

The convolution layer in question is likely defined using Flax's linen API, and the kernel's shape is determined by the arguments provided when defining the convolution layer. For example, a typical definition of a convolution layer might look like this:

python
Copy code
self.conv1 = nn.Conv(features=384, kernel_size=(3, 80))
In this case, the kernel size is (3, 80), and the number of output channels (features) is 384. The first dimension of the kernel tensor is usually the number of input channels, and it's often determined automatically based on the shape of the input tensor.

The error message suggests that the expected number of input channels is 4, but the initializer is only providing a tensor with 3 input channels. This discrepancy could be due to a mismatch between the actual shape of your input data and the expected input shape of the model.

One way to resolve this issue is to ensure that your input data has the correct number of channels. If your input audio data is mono (single-channel), you might need to convert it to multi-channel (e.g., stereo) data. Alternatively, if the model is supposed to handle mono audio data, you might need to adjust the model configuration or the convolution layer definition to expect a single input channel.

Without more information about the exact structure of the FlaxWhisperForConditionalGeneration model and the shape of your input data, it's difficult to provide a more precise solution. I recommend checking both the model definition (particularly the "/model/encoder/conv1" layer) and the shape of your input data to identify where the discrepancy lies.




User
could this be a mapping thing? I am running this on a TPU with 4 cores
ChatGPT
The issue does not seem to be related to the number of TPU cores. It's more likely to be an issue with the model configuration or the input data.

The error message indicates that the kernel of a convolution layer in the model has a different shape than expected. The kernel's shape is usually determined by the model configuration (e.g., the number of input and output channels and the kernel size), not the number of TPU cores.

The jax.pmap function, which is used to parallelize computations over TPU cores, operates on the leading axis of its input data. This means that if you have a batch of data with shape (BATCH_SIZE, ...), where BATCH_SIZE is divisible by the number of TPU cores, jax.pmap will divide the batch into smaller mini-batches, each processed by one TPU core.

However, this doesn't affect the shape of the model's parameters, like the convolution kernel. The model parameters are typically replicated across all TPU cores, so each core has a complete copy of the model. Each core processes a mini-batch of data independently using its copy of the model.

To troubleshoot your issue, I recommend checking the following:

Verify the shape of your input data. If your model expects audio data with a certain number of channels (e.g., stereo data), but you're providing mono data, this could cause a shape mismatch. Check if you need to adjust your input data to match the model's expectations.

Check the configuration of your model, especially the "/model/encoder/conv1" layer. The error message indicates that this layer's kernel has an unexpected shape. You might need to adjust the model configuration to match your input data.

Consider whether your model might be expecting a different data layout. For instance, some models might expect the channel dimension to be last (e.g., (batch_size, length, channels)) while others might expect it to be second (e.g., (batch_size, channels, length)).

Lastly, if possible, try running your code on a single device (like a CPU or a single GPU) to see if the issue persists. If the code runs successfully on a single device, this might indicate that the issue is related to the parallelization with jax.pmap. However, given the error message, this seems less likely.




User
I will try an even simpler example.
ChatGPT
That sounds like a good approach. Simplifying the problem can often help to isolate the issue. Here's a stripped-down example of how you might load a model and process a single batch of audio data:

python
Copy code
import jax
import jax.numpy as jnp
from flax import jax_utils
from transformers import WhisperProcessor
from whisper_jax import FlaxWhisperForConditionalGeneration

# Load the model and replicate the parameters across devices
model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-tiny.en",
    _do_init=False,
    dtype=jnp.bfloat16,
)
params = jax_utils.replicate(params)

# Load the WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

# Define the function to generate predictions
def generate_fn(input_features):
    pred_ids = model.generate(input_features, params=params)
    return pred_ids.sequences

# Map the generate function to multiple devices
p_generate_fn = jax.pmap(generate_fn, "batch")

# Example of a single batch of audio data
batch = {"audio": {"array": my_audio_data, "sampling_rate": my_sampling_rate}}

# Preprocess the audio data
input_features = processor(
    batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"], return_tensors="np"
).input_features[0]

# Convert the input features to a batched array
input_features = jnp.array([input_features])

# Generate predictions
pred_ids = p_generate_fn(input_features)

# Decode the predicted token IDs to text
transcriptions = processor.batch_decode(pred_ids[0], skip_special_tokens=True)

print(transcriptions)
This code assumes you have some audio data available in the variable my_audio_data with the corresponding sampling rate my_sampling_rate. You should replace these placeholders with your actual data.

This example processes a single batch of data, but it should be straightforward to extend it to process multiple batches in a loop, similar to your original code.






Regenerate
This is the example code from the webb bage


ChatGPT may produce inaccurate information about people, places, or facts. ChatGPT July 20 Version
ChatGPT
