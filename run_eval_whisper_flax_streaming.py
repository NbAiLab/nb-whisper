#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 National Library of Norway. All rights reserved.
#
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
Evaluating Whisper models using the Flax library and 🤗 Datasets.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import logging
from datasets import Dataset, DatasetDict, IterableDatasetDict, interleave_datasets, load_dataset
import jax
import jax.numpy as jnp
from flax.training.common_utils import shard
from flax.jax_utils import pad_shard_unpad, unreplicate
from tqdm.auto import tqdm
from transformers import FlaxAutoModelForSpeechSeq2Seq, AutoTokenizer, AutoProcessor, AutoFeatureExtractor
from typing import Any, Callable, Dict, Generator, List, Optional, Union
from flax.training.common_utils import get_metrics
import torch
import itertools
from functools import partial
import flax
import numpy as np


logger = logging.getLogger(__name__)

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
        max_input_length (:obj:`float`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_target_length (:obj:`int`, `optional`):
            Maximum length of the ``labels`` of the returned list and optionally padding length (see above).
        pad_input_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the input sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        pad_target_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the target sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Any
    decoder_start_token_id: int
    input_padding: Union[bool, str] = "longest"
    target_padding: Union[bool, str] = "max_length"
    max_input_length: Optional[float] = None
    max_target_length: Optional[int] = None
    pad_input_to_multiple_of: Optional[int] = None
    pad_target_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:
        model_input_name = self.processor.model_input_names[0]
        input_features = {model_input_name: features[model_input_name]}
        label_features = {"input_ids": features["labels"]}

        # reformat list to dict and set to pytorch format
        batch = self.processor.feature_extractor.pad(
            input_features,
            max_length=self.max_input_length,
            padding=self.input_padding,
            pad_to_multiple_of=self.pad_input_to_multiple_of,
            return_tensors="np",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            max_length=self.max_target_length,
            padding=self.target_padding,
            pad_to_multiple_of=self.pad_target_to_multiple_of,
            return_tensors="np",
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        labels = labels_batch["input_ids"]
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
            labels_batch.attention_mask = labels_batch.attention_mask[:, 1:]
        
        
            
        decoder_input_ids = shift_tokens_right(
            labels, self.decoder_start_token_id)

        # replace padding with -100 to ignore correctly when computing the loss
        labels = np.ma.array(labels, mask=np.not_equal(
            labels_batch.attention_mask, 1))
        labels = labels.filled(fill_value=-100)

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids
        batch["attention_mask"] = labels_batch.attention_mask  # Add attention_mask to the batch
        
        return batch


def load_maybe_streaming_dataset(dataset_name, dataset_config_name, split="train", streaming=True, **kwargs):
    """
    Utility function to load a dataset in streaming mode. For datasets with multiple splits,
    each split is loaded individually and then splits combined by taking alternating examples from
    each (interleaving).
    """
    if "+" in split:
        # load multiple splits separated by the `+` symbol with streaming mode
        dataset_splits = [
            load_dataset(dataset_name, dataset_config_name,
                         split=split_name, streaming=streaming, **kwargs)
            for split_name in split.split("+")
        ]
        # interleave multiple splits to form one dataset
        interleaved_dataset = interleave_datasets(dataset_splits)
        return interleaved_dataset
    else:
        # load a single split *with* streaming mode
        dataset = load_dataset(
            dataset_name, dataset_config_name, split=split, streaming=streaming, **kwargs)
        return dataset

def collate_batch(samples):
    return {key: [feature[key] for feature in samples] for key in samples[0]}

def data_loader(
    dataset: Dataset,
    batch_size: int,
    drop_last: bool=True,
    num_workers: int=0,
) -> Generator:
    """
    Returns batches of size `batch_size` from `dataset`. If `drop_last` is set to `False`, the final batch may be incomplete,
    and range in size from 1 to `batch_size`. Shuffle batches if `shuffle` is `True`.
    """
    data_loader_iterator = iter(torch.utils.data.DataLoader(
        batch_size=batch_size,
        dataset=dataset.with_format("torch"),
        num_workers=num_workers,
        collate_fn=collate_batch,
        drop_last=drop_last,
    ))
    return data_loader_iterator

def generate_step(params, batch):
    model.params = params
    
    attention_mask = batch.get("attention_mask")
    
    output_ids = model.generate(batch[model_input_name], attention_mask=attention_mask, **gen_kwargs)
    
    return output_ids.sequences

# Define eval fn
def eval_step(params, batch, label_smoothing_factor=0.0):
    labels = batch.pop("labels")
    logits = model(**batch, params=params, train=False)[0]

    loss, num_labels = loss_fn(logits, labels, label_smoothing_factor)
    num_labels = jax.lax.psum(num_labels, "batch")

    # True loss = total loss / total samples
    loss = jax.lax.psum(loss, "batch")
    loss = jax.tree_util.tree_map(lambda x: x / num_labels, loss)

    metrics = {"loss": loss}
    return metrics

def compute_metrics(pred_ids, label_ids, return_preds_labels=False):
    # Replace padded labels by the padding token
    for idx in range(len(label_ids)):
        label_ids[idx][label_ids[idx] == -100] = tokenizer.pad_token_id

    predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # We do not want to group tokens when computing the metrics
    labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if do_normalize_eval:
        pred_str = [normalizer(pred) for pred in predictions]
        label_str = [normalizer(label) for label in labels]
        # Filtering step to only evaluate the samples that correspond to non-zero references:
        pred_str = [pred_str[i]
                    for i in range(len(pred_str)) if len(label_str[i]) > 0]
        label_str = [label_str[i]
                     for i in range(len(label_str)) if len(label_str[i]) > 0]
    else:
        pred_str = predictions
        label_str = labels

    wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
    cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)

    if return_preds_labels:
        return {"wer": wer, "cer": cer}, predictions, labels
    else:
        return {"wer": wer, "cer": cer}
    

def evaluate(model_name, dataset_name, dataset_split_name, num_beams):
    #Default settings
    streaming = True
    dataset_config_name = None
    text_column_name = "text"
    audio_column_name = "audio"
    max_label_length = 256
    
    def prepare_dataset(batch):
        # Process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(
            sample["array"], sampling_rate=sample["sampling_rate"])
        # Process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]
        batch["input_length"] = len(sample["array"])

        # Process targets
        input_str = batch[text_column_name]
        
        
        batch["labels"] = tokenizer(input_str, truncation=True, max_length=max_label_length).input_ids
        return batch
    
    
    model = FlaxAutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name,use_auth_token=True)
    model_input_name = feature_extractor.model_input_names[0]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.set_prefix_tokens(language="Norwegian", task="transcribe")
    
    processor = AutoProcessor.from_pretrained(model_name)
    data_collator = FlaxDataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        input_padding="longest",
        target_padding="longest",
        max_target_length=256,
        pad_input_to_multiple_of=None,
        pad_target_to_multiple_of=None,
    )

    raw_datasets = IterableDatasetDict() if streaming else DatasetDict()
    raw_datasets["eval"] = load_maybe_streaming_dataset(
            dataset_name,
            dataset_config_name,
            split=dataset_split_name,
            streaming=streaming,
            use_auth_token=True,
        )
        
    raw_datasets_features = list(next(iter(raw_datasets.values())).features.keys())
    

    
    vectorized_datasets = raw_datasets.map(
        prepare_dataset,
        remove_columns=raw_datasets_features,
    )
    eval_dataset = vectorized_datasets["eval"]
    
    gen_kwargs = {"max_length": 256, "num_beams": num_beams}

    p_eval_step = jax.pmap(partial(
        eval_step, label_smoothing_factor=0.0), "batch")
    
    p_generate_step = jax.pmap(generate_step, "batch")

    max_eval_steps_iter = itertools.repeat(None)
    eval_loader = data_loader(eval_dataset, 4, drop_last=False)

    eval_metrics = []
    eval_preds = []
    eval_labels = []

    for _ in tqdm(max_eval_steps_iter, desc="Evaluating...", position=2, leave=False):
        # Model forward
        try:
            samples = next(eval_loader)
        except StopIteration:
            break
        batch = data_collator(samples)
        breakpoint()
        labels = batch["labels"]

        metrics = pad_shard_unpad(p_eval_step, static_return=True)(
            model.params, batch.data, min_device_batch=1)
        eval_metrics.append(metrics)

        # Generation
        generated_ids = pad_shard_unpad(
            p_generate_step)(model.params, batch.data)
        eval_preds.extend(jax.device_get(
            generated_ids.reshape(-1, gen_kwargs["max_length"])))
        eval_labels.extend(labels)
            
    # Normalize eval metrics
    eval_metrics = get_metrics(eval_metrics)
    eval_metrics = jax.tree_util.tree_map(jnp.mean, eval_metrics)

    # Compute metrics
    metric_desc = ""
    metric_values, pred_str, label_str = compute_metrics(
        eval_preds, eval_labels, return_preds_labels=True
    )
    eval_metrics.update(metric_values)
    metric_desc = " | ".join(
        [f"Eval {key}: {value}" for key, value in metric_values.items()])

    desc = f"Eval WER: {metric_values['wer']}"

    logger.info(desc)



def main(args):
    evaluate(args.model_id, args.dataset, args.split, args.num_beams)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mozilla-foundation/common_voice_11_0",
        help="Dataset name to evaluate the `model_id`. Should be loadable with 🤗 Datasets",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'test'`",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams for beam search in generation.",
    )

    args = parser.parse_args()

    main(args)

            
