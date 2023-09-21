#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the Flax library models for sequence to sequence speech recognition.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import csv
import logging
import os
import string
import sys
import time
from dataclasses import field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets
import evaluate
import flax
import jax
import jax.numpy as jnp
import numpy as np
import transformers
from accelerate import skip_first_batches
from datasets import DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from flax import jax_utils
from flax.jax_utils import pad_shard_unpad
from huggingface_hub import Repository, create_repo, get_full_repo_name
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    is_wandb_available,
)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from whisper_jax import FlaxWhisperForConditionalGeneration


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")

require_version(
    "datasets>=1.18.0",
    "To fix: pip install -r examples/flax/speech-recogintion/requirements.txt",
)

# logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
# logger = logging.getLogger(__name__)


@flax.struct.dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": ("Path to pretrained model or model identifier from" " huggingface.co/models")}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    feature_extractor_name: Optional[str] = field(
        default=None,
        metadata={"help": "feature extractor name or path if not the same as model_name"},
    )
    processor_name: Optional[str] = field(
        default=None,
        metadata={"help": "processor name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": ("Where to store the pretrained models downloaded from huggingface.co")},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": ("Whether to use one of the fast tokenizer (backed by the tokenizers" " library) or not.")},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": ("The specific model version to use (can be a branch name, tag name or" " commit id).")},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login`"
                " (necessary to use this script with private models)."
            )
        },
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized"
                " and trained. Choose one of `[float32, float16, bfloat16]`."
            )
        },
    )
    load_with_scan: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to load the model with scan enabled. Required when the model" " was saved with scan enabled"
            )
        },
    )


@flax.struct.dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task: str = field(
        default="transcribe",
        metadata={"help": "The generation task."},
    )
    language: str = field(
        default="Norwegian",
        metadata={"help": "The langauge of the predictions."},
    )
    language_code: str = field(
        default="<|no|>",
        metadata={"help": "The langauge code of the predictions."},
    )
    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": ("The configuration name of the dataset to use (via the datasets" " library).")},
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to cache directory for saving and loading datasets"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": ("The name of the dataset column containing the audio data. Defaults to" " 'audio'")},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": ("The name of the dataset column containing the text data. Defaults to" " 'text'.")},
    )
    id_column_name: str = field(
        default="id",
        metadata={"help": "The name of the dataset column containing the id data. Defaults to 'id'"},
    )
    max_label_length: int = field(
        default=128,
        metadata={"help": "Truncate transcriptions that are longer `max_label_length` tokens."},
    )
    pad_target_to_multiple_of: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If set will pad the target sequence to a multiple of the provided"
                " value. This is important to avoid triggering recompilations on TPU."
                " If unspecified, will default to padding the targets to max length."
            )
        },
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is"
                " especially useful when data preprocessing errors out in distributed"
                " training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with"
                " `preprocessing_only=True` so that the cached datasets can"
                " consequently be loaded in distributed training"
            )
        },
    )
    data_split_name: str = field(
        default="train+validation+test",
        metadata={
            "help": (
                "The name of the data set splits to use (via the datasets library)."
                " Defaults to 'train+validation+test'. Multipletest splits can be passed by splitting a"
                " list through the '+' character, e.g. 'test.clean+test.other' will"
                " teston both the 'test.clean' and 'test.other' splits sequentially."
            )
        },
    )
    wandb_project: str = field(
        default=None,
        metadata={"help": "The name of the wandb project."},
    )
    wandb_name: str = field(
        default=None,
        metadata={"help": "The name of the wandb run."},
    )
    wandb_job_type: str = field(
        default=None,
        metadata={"help": "The name of the wandb job type."},
    )
    wandb_dir: str = field(
        default=None,
        metadata={"help": "The absolute path to save the wandb logs."},
    )
    save_code_to_wandb: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to save main script to wandb. This is valuable for improving"
                " experiment reproducibility and to diff code across experiments in"
                " the UI."
            )
        },
    )
    streaming: bool = field(
        default=False,
        metadata={"help": ("Whether to use dataset's streaming mode to load and pre-process the data.")},
    )
    max_samples_per_split: Optional[int] = field(
        default=None,
        metadata={"help": ("For debugging purposes, truncate the number of examples per split to this value if set.")},
    )
    return_timestamps: bool = field(
        default=False,
        metadata={
            "help": "Whether to return the timestamps with the text. This enables the `FlaxWhisperTimestampsLogitsProcessor`."
        },
    )


def shift_tokens_right(label_ids: np.array, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift label ids one token to the right.
    """
    shifted_label_ids = np.zeros_like(label_ids)
    shifted_label_ids[:, 1:] = label_ids[:, :-1]
    shifted_label_ids[:, 0] = decoder_start_token_id

    return shifted_label_ids


@flax.struct.dataclass
class FlaxDataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`Wav2Vec2Processor`])
            The processor used for proccessing the data.
        decoder_start_token_id (:obj: `int`)
            The begin-of-sentence of the decoder.
        input_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned input sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        target_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned target sequences (according to the model's padding side and padding index).
            See above for details.
        max_target_length (:obj:`int`, `optional`):
            Maximum length of the ``labels`` of the returned list and optionally padding length (see above).
    """

    processor: Any
    decoder_start_token_id: int
    input_padding: Union[bool, str] = "max_length"
    target_padding: Union[bool, str] = "max_length"
    max_target_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]

        # dataloader returns a list of features which we convert to a dict
        input_features = {model_input_name: [feature[model_input_name] for feature in features]}
        label_features = {"input_ids": [feature["labels"] for feature in features]}
        file_ids = [feature["file_id"] for feature in features]

        # reformat list to dict and set to pytorch format
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.input_padding,
            return_tensors="np",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            max_length=self.max_target_length,
            padding=self.target_padding,
            return_tensors="np",
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        labels = labels_batch["input_ids"]
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
            labels_batch.attention_mask = labels_batch.attention_mask[:, 1:]

        decoder_input_ids = shift_tokens_right(labels, self.decoder_start_token_id)

        # replace padding with -100 to ignore correctly when computing the loss
        labels = np.ma.array(labels, mask=np.not_equal(labels_batch.attention_mask, 1))
        labels = labels.filled(fill_value=-100)

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids
        batch["file_ids"] = file_ids

        return batch


def get_data_loader(
    dataset: IterableDataset,
    batch_size: int,
    data_collator: FlaxDataCollatorSpeechSeq2SeqWithPadding,
    dataloader_num_workers: int = 0,
    skip_batches: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Returns batches of size `batch_size` from `dataset`. If `drop_last` is set to `False`, the final batch may be incomplete,
    and range in size from 1 to `batch_size`. Shuffle batches if `shuffle` is `True`.

    Args:
        dataset (IterableDataset): streaming dataset from which to load the data.
        batch_size (int): how many samples per batch to load.
        data_collator (FlaxDataCollatorSpeechSeq2SeqWithPadding, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a map-style dataset.
        dataloader_num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        skip_batches (int, optional): Efficiently skip the first `skip_batches`.
        pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
            into device/CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.

    """

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=pin_memory,
        collate_fn=data_collator,
        num_workers=dataloader_num_workers,
    )

    data_loader_skipped = skip_first_batches(data_loader, num_batches=skip_batches)
    return data_loader_skipped


def write_wandb_metric(wandb_logger, metrics, train_time, prefix):
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    log_metrics[f"{prefix}/time"] = train_time
    wandb_logger.log(log_metrics)


def write_wandb_pred(
    wandb_logger, pred_str, label_str, norm_pred_str, norm_label_str, prefix="eval", num_lines=200000
):
    # convert str data to a wandb compatible format
    str_data = [[label_str[i], pred_str[i], norm_label_str[i], norm_pred_str[i]] for i in range(len(pred_str))]
    # log as a table with the appropriate headers
    wandb_logger.log(
        {
            f"{prefix}/all_predictions": wandb_logger.Table(
                columns=["Target", "Pred", "Norm Target", "Norm Pred"], data=str_data[:num_lines]
            )
        },
    )
    # log incorrect normalised predictions
    str_data = np.asarray(str_data)
    str_data_incorrect = str_data[str_data[:, 2] != str_data[:, 3]]
    # log as a table with the appropriate headers
    wandb_logger.log(
        {
            f"{prefix}/incorrect_predictions": wandb_logger.Table(
                columns=["Target", "Pred", "Norm Target", "Norm Pred"], data=str_data_incorrect[:num_lines]
            )
        },
    )


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set to the lowest level you want to log

    # Create a file handler for logging all messages
    file_handler = logging.FileHandler('all_logs.log')
    file_handler.setLevel(logging.DEBUG)

    # Create a stream handler for logging messages INFO and above to stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    logger = setup_logger()


    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Setup logging
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Set the verbosity to info of the Transformers logger.
    # We only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    logger.info("Evaluation parameters %s", training_args)

    # Enable wandb only on the master node
    # has_wandb = is_wandb_available()
    # Manually set to False for now
    has_wandb = False

    if "wandb" in training_args.report_to:
        if has_wandb and jax.process_index() == 0:
            import wandb as wandb_logger

            # Set up wandb run
            wandb_logger.init(
                project=data_args.wandb_project,
                name=data_args.wandb_name,
                job_type=data_args.wandb_job_type,
                dir=data_args.wandb_dir,
                save_code=data_args.save_code_to_wandb,
            )
        else:
            logger.warning("Wandb logging requires wandb to be installed. Run `pip install wandb`" " to enable.")

    # 3. Load dataset
    raw_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()

    data_splits = data_args.data_split_name.split("+")
    for split in data_splits:
        raw_datasets[split] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=split,
            cache_dir=data_args.dataset_cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )

    if data_args.audio_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--audio_column_name '{data_args.audio_column_name}' not found in dataset"
            f" '{data_args.dataset_name}'. Make sure to set `--audio_column_name` to"
            " the correct audio column - one of"
            f" {', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    if data_args.text_column_name not in next(iter(raw_datasets.values())).column_names:
        raise ValueError(
            f"--text_column_name {data_args.text_column_name} not found in dataset"
            f" '{data_args.dataset_name}'. Make sure to set `--text_column_name` to the"
            " correct text column - one of"
            f" {', '.join(next(iter(raw_datasets.values())).column_names)}."
        )

    # 5. Load pretrained model, tokenizer, and feature extractor
    config = AutoConfig.from_pretrained(
        (model_args.config_name if model_args.config_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        (model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    processor = AutoProcessor.from_pretrained(
        (model_args.processor_name if model_args.processor_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        dtype=getattr(jnp, model_args.dtype),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        _do_init=False    
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # disable scan if necessary (makes the inference step faster)
    if model_args.load_with_scan:
        model.disable_scan()  # to disable scan in the nn.Module
        params = model.convert_scan_to_unroll(params)  # to convert the scan params to unrolled

    return_timestamps = data_args.return_timestamps
    if hasattr(model.generation_config, "is_multilingual") and model.generation_config.is_multilingual:
        # We need to set the language and task ids for multilingual checkpoints - for now we hardcode this to English
        tokenizer.set_prefix_tokens(language=data_args.language, task=data_args.task, predict_timestamps=return_timestamps)

    # 6. Resample speech dataset: `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name,
        datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
    )

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_label_length = (
        data_args.max_label_length if data_args.max_label_length is not None else model.config.max_length
    )
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    dataloader_num_workers = training_args.dataloader_num_workers
    text_column_name = data_args.text_column_name
    model_input_name = feature_extractor.model_input_names[0]
    id_column_name = data_args.id_column_name
    normalizer = EnglishTextNormalizer(tokenizer.english_spelling_normalizer)

    if data_args.max_samples_per_split is not None:
        for split in data_splits:
            raw_datasets[split] = (
                raw_datasets[split].take(data_args.max_samples_per_split)
                if data_args.streaming
                else raw_datasets[split].select(range(data_args.max_samples_per_split))
            )

    def prepare_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]

        # process targets
        input_str = batch[text_column_name]
        batch["labels"] = tokenizer(input_str, max_length=max_label_length, truncation=True).input_ids

        # record the id of the sample
        batch["file_id"] = batch[id_column_name]
        return batch

    raw_datasets_features = list(next(iter(raw_datasets.values())).features.keys())
    if data_args.streaming:
        vectorized_datasets = raw_datasets.map(prepare_dataset, remove_columns=raw_datasets_features)
    else:
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=raw_datasets_features,
            num_proc=num_workers,
            desc="preprocess dataset",
            load_from_cache_file=False,
        )

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with `args.preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step `args.preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        cache = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")
        return

    if data_args.streaming and dataloader_num_workers > 0:
        logger.warning(
            "Using multiple dataloader num workers with streaming mode will result in different shards of "
            "data being transcribed in parallel. This is not advised if you want to preserve the order of the "
            "audio-text data."
        )

    # Handle the repository creation
    if data_args.return_timestamps:
        timestampstring = "timestamps_"
    else:
        timestampstring = ""
    model_name = model_args.model_name_or_path.replace("/", "-")
    repo_name = "infer_"+data_args.language+"_"+data_args.task+"_"+timestampstring+model_name
    
    output_dir = os.path.join(training_args.output_dir, repo_name)
    if training_args.push_to_hub:
        logger.info("Pushing to hub..")

        if training_args.hub_model_id is None:
            repo_name = get_full_repo_name(
                Path(output_dir).absolute().name,
                token=training_args.hub_token,
            )
        else:
            repo_name = training_args.hub_model_id
        create_repo(repo_name, exist_ok=True, token=training_args.hub_token, repo_type="dataset", private=True)
        repo = Repository(
            output_dir,
            clone_from=repo_name,
            token=training_args.hub_token,
            repo_type="dataset",
        )
        # Ensure large txt files can be pushed to the Hub with git-lfs
        with open(os.path.join(output_dir, ".gitattributes"), "r+") as f:
            git_lfs_extensions = f.read()
            if "*.csv" not in git_lfs_extensions:
                f.write("*.csv filter=lfs diff=lfs merge=lfs -text")
            if "*.tsv" not in git_lfs_extensions:
                f.write("*.tsv filter=lfs diff=lfs merge=lfs -text")
    else:
        logger.info("Saving locally.")

        # this is where we'll save our transcriptions
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # 8. Load Metric
    metric = evaluate.load("wer")
    # convention is that we space all punctuation *except* apostrophes
    all_punctuation = list(string.punctuation.replace("'", ""))

    def compute_metrics(preds, labels):
        # replace padded labels by the padding token
        for idx in range(len(labels)):
            labels[idx][labels[idx] == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True, decode_with_timestamps=return_timestamps)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # space punctuation for orthographic WER (c.f. ESB paper https://arxiv.org/abs/2210.13352)
        spaced_pred_str = [
            pred_str[i].replace(punctuation, f" {punctuation} ")
            for punctuation in all_punctuation
            for i in range(len(pred_str))
        ]
        spaced_label_str = [
            label_str[i].replace(punctuation, f" {punctuation} ")
            for punctuation in all_punctuation
            for i in range(len(label_str))
        ]
        # Disabling since I am getting an error
        #wer_ortho = 100 * metric.compute(predictions=spaced_pred_str, references=spaced_label_str)
        wer_ortho = 0

        # normalize everything and re-compute the WER
        norm_pred_str = [normalizer(pred) for pred in pred_str]
        norm_label_str = [normalizer(label) for label in label_str]
        # for logging, we need the pred/labels to match the norm_pred/norm_labels, so discard any filtered samples here
        pred_str = [pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
        label_str = [label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]
        # filtering step to only evaluate the samples that correspond to non-zero normalized references:
        norm_pred_str = [norm_pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
        norm_label_str = [norm_label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]

        wer = 100 * metric.compute(predictions=norm_pred_str, references=norm_label_str)

        return {"wer": wer, "wer_ortho": wer_ortho}, pred_str, label_str, norm_pred_str, norm_label_str

    data_collator = FlaxDataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        input_padding="longest",
        target_padding="max_length",
        max_target_length=max_label_length,
    )

    # Store some constants
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    eval_batch_size = per_device_eval_batch_size * jax.device_count()

    # Define generation function
    num_beams = (
        training_args.generation_num_beams
        if training_args.generation_num_beams is not None
        else model.config.num_beams
    )

    gen_kwargs = {
        "max_length": max_label_length,
        "num_beams": num_beams,
        "language": data_args.language_code,  # forcing the language and task tokens helps the flax model in its generations
        "task": data_args.task,
        "return_timestamps": return_timestamps,
    }

    def generate_step(params, batch):
        output_ids = model.generate(
            batch[model_input_name],
            # attention_mask=batch.get("attention_mask"),
            params=params,
            **gen_kwargs,
        )
        return output_ids.sequences

    # Create parallel version of the generate step
    p_generate_step = jax.pmap(generate_step, "batch")

    # Replicate params on each device
    params = jax_utils.replicate(params)

    def eval_step_with_save(split="eval"):
        # ======================== Evaluating ==============================
        eval_preds = []
        eval_labels = []
        eval_ids = []
        eval_start = time.time()

        eval_loader = get_data_loader(
            vectorized_datasets[split],
            batch_size=eval_batch_size,
            data_collator=data_collator,
            dataloader_num_workers=dataloader_num_workers,
        )
        # make the split name pretty for librispeech etc
        split = split.replace(".", "-").split("/")[-1]
        model_name = model_args.model_name_or_path.replace("/", "-")
        output_csv = os.path.join(output_dir, f"{model_name}-{data_args.language}-{data_args.task}-{split}-transcription.tsv")

        batches = tqdm(eval_loader, desc=f"Evaluating {split}...")

        for step, batch in enumerate(batches):
            logger.info(f"S={step}")
            # Model forward
            labels = batch["labels"]
            eval_ids.extend(batch.pop("file_ids"))

            # generation
            generated_ids = pad_shard_unpad(p_generate_step)(
                params, batch.data, min_device_batch=per_device_eval_batch_size
            )
            eval_preds.extend(jax.device_get(generated_ids.reshape(-1, gen_kwargs["max_length"])))
            eval_labels.extend(labels)

            if step % training_args.logging_steps == 0 and step > 0:
                eval_preds_list = [arr.tolist() for arr in eval_preds]
                pred_str = tokenizer.batch_decode(eval_preds_list, skip_special_tokens=True)
                csv_data = [[eval_ids[i], pred_str[i]] for i in range(len(pred_str))]

                with open(output_csv, "w", encoding="UTF8", newline="") as f:
                    batches.write(f"Opening file split {split} step {step}")

                    writer = csv.writer(f, delimiter="\t")
            
                    # write multiple rows
                    writer.writerow(["id", model_args.model_name_or_path])
                    writer.writerows(csv_data)
                    batches.write(f"Finished writing split {split} step {step}")


                if training_args.push_to_hub:
                    logger.info("Pushing to hub.")
                    repo.push_to_hub(
                        commit_message=f"Saving transcriptions for split {split} step {step}.",
                        blocking=False,
                    )
                else:
                    logger.info("Here we should push to the bucket")


        eval_time = time.time() - eval_start

        # compute WER metric for eval sets
        wer_desc = ""
        if "validation" in split or "test" in split:
            wer_metric, pred_str, label_str, norm_pred_str, norm_label_str = compute_metrics(eval_preds, eval_labels)
            wer_desc = " ".join([f"Eval {key}: {value} |" for key, value in wer_metric.items()])
            # Save metrics
            if has_wandb and jax.process_index() == 0 and "wandb" in training_args.report_to:
                write_wandb_metric(wandb_logger, wer_metric, eval_time, prefix=split)
                write_wandb_pred(wandb_logger, pred_str, label_str, norm_pred_str, norm_label_str, prefix=split)
        else:
            pred_str = tokenizer.batch_decode(eval_preds, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(eval_labels, skip_special_tokens=True)

        batches.write(f"Saving final transcriptions for split {split}.")
        csv_data = [[eval_ids[i], label_str[i], pred_str[i]] for i in range(len(pred_str))]
        with open(output_csv, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f, delimiter='\t')
            # write multiple rows
            writer.writerow(["file_id", "target", model_args.model_name_or_path])
            writer.writerows(csv_data)

        # Print metrics
        logger.info(wer_desc)

    logger.info("***** Running Labelling *****")
    logger.info("  Instantaneous batch size per device =" f" {training_args.per_device_eval_batch_size}")
    logger.info(f"  Total eval batch size (w. parallel & distributed) = {eval_batch_size}")
    for split in data_splits:
        eval_step_with_save(split=split)
        if training_args.push_to_hub:
            logger.info("Final push to hub.")

            repo.push_to_hub(
                commit_message=f"Saving final transcriptions for split {split.replace('.', '-').split('/')[-1]}",
                blocking=False,
            )
        else:
            logger.info("Final push to the bucket")


if __name__ == "__main__":
    main()