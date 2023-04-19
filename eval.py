import os
import sys
from unittest.mock import patch

from slugify import slugify

import run_flax_speech_recognition_seq2seq_streaming as cli


def get_run_name(model_name_or_path, dataset_name, dataset_config_name, num_beams, do_normalize_eval, **kwargs):
    run_name_parts = [
        slugify(model_name_or_path),
        slugify(dataset_name),
        slugify(dataset_config_name or "default"),
        f"{num_beams or 0}beams",
    ]
    if do_normalize_eval:
        run_name_parts.append("normalize")
    return "_".join(run_name_parts).lower()


def main():
    parser = cli.HfArgumentParser(
        (cli.ModelArguments, cli.DataTrainingArguments, cli.Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    sys_argv = sys.argv
    run_name_params = {**model_args.__dict__, **data_args.__dict__, **training_args.__dict__}
    if "--dataset_name" in sys_argv:
        dataset_pos = sys_argv.index("--dataset_name") + 1
        dataset = sys_argv[dataset_pos]
        if ":" in dataset:
            dataset_name, dataset_config_name = dataset.split(":")
            sys_argv[dataset_pos] = dataset_name
            sys_argv.extend(["--dataset_config_name", dataset_config_name])
            run_name_params["dataset_config_name"] = dataset_config_name
    if "--run_name" not in sys_argv:
        sys_argv.extend(["--run_name", get_run_name(**run_name_params)])
    with patch.object(sys, "argv", sys_argv):
        cli.main()


if __name__ == "__main__":
    main()
