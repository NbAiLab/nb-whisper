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


import argparse
import itertools
import json
import logging
from pathlib import Path
from datasets import load_metric
import jax
import jax.numpy as jnp
from flax.training.common_utils import shard
from flax.training.train_state import TrainState
from tqdm.auto import tqdm
from transformers import FlaxAutoModelForSpeechSeq2Seq, AutoTokenizer
from datasets import load_dataset

logger = logging.getLogger(__name__)

def data_loader(dataset, batch_size, drop_last=True):
    for i in range(0, len(dataset), batch_size):
        if drop_last and i + batch_size > len(dataset):
            break
        yield dataset[i:i + batch_size]

def evaluate(model_name, dataset_name, dataset_split_name, num_beams):
    model = FlaxAutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset(dataset_name, split=dataset_split_name)

    state = TrainState.create(apply_fn=model.__call__, params=model.params)

    data_collator = lambda samples: tokenizer(samples["speech"], truncation=True, padding="longest", return_tensors="jax")

    gen_kwargs = {"max_length": 256, "num_beams": num_beams, "early_stopping": True}

    p_generate_step = jax.pmap(
        model.generate,
        axis_name="batch",
        in_axes=(0, None),
        static_broadcasted_argnums=1,
        donate_argnums=(0,)
    )

    eval_metrics = []
    eval_preds = []
    eval_labels = []
    eval_loader = data_loader(dataset, 16, drop_last=False)
    for _ in tqdm(itertools.repeat(None), desc="Evaluating...", leave=False):
        try:
            samples = next(eval_loader)
        except StopIteration:
            break
        batch = data_collator(samples)

        labels = batch["labels"]

        generated_ids = p_generate_step(state.params, batch.data)
        eval_preds.extend(jax.device_get(generated_ids.reshape(-1, gen_kwargs["max_length"])))
        eval_labels.extend(labels)

    eval_preds_text = tokenizer.batch_decode(eval_preds, skip_special_tokens=True)
    eval_labels_text = tokenizer.batch_decode(eval_labels, skip_special_tokens=True)

    wer_metric = load_metric("wer")
    metric_value = wer_metric.compute(predictions=eval_preds_text, references=eval_labels_text)

    desc = f"Eval WER: {metric_value['wer']}"
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

            
