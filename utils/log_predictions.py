import logging
import os
import re
import jax
import pandas as pd


logger = logging.getLogger(__name__)

# Setting log level to screen and making sure only one processor per machine outputs logs
logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)


def trim_bold(text):
    if text.startswith(" "):
        return f" {trim_bold(text[1:])}"
    elif text.endswith(" "):
        return f"{trim_bold(text[:-1])} "
    else:
        return f"**{text}**"


def format_diff(label_text, pred_text):
    label_words = re.split(r"\b", label_text)
    label_bag = set(word.strip() for word in label_words)
    pred_words = re.split(r"\b", pred_text)
    pred_bag = set(word.strip() for word in pred_words)

    formatted_label_text = "".join(
        [trim_bold(word)
         if word.strip() not in pred_bag else f"{word}" for word in label_words]
    )
    formatted_pred_text = "".join(
        [trim_bold(word)
         if word.strip() not in label_bag else f"{word}" for word in pred_words]
    )
    formatted_label_text = formatted_label_text.replace(
        "****", "").replace(".", r"\.").replace(",", r"\,").replace("-", r"\-")
    formatted_pred_text = formatted_pred_text.replace(
        "****", "").replace(".", r"\.").replace(",", r"\,").replace("-", r"\-")

    return formatted_label_text, formatted_pred_text


def write_predictions(
    summary_writer,
    train_metrics,
    eval_metrics,
    train_time,
    step,
    prediction_ids=None,
    predictions=None,
    label_ids=None,
    labels=None,
    model_args=None,
    training_args=None,
    data_args=None,
    eval_name=None,
):
    if not eval_name:
        predictions_folder_name = os.path.join(
            training_args.output_dir, "predictions")
    else:
        predictions_folder_name = os.path.join(
            training_args.output_dir, "predictions", eval_name)

    if not os.path.exists(predictions_folder_name):
        os.makedirs(predictions_folder_name)

    eval_table = f"| STEP| loss | wer |cer|\n| ---| --- | --- |--- |\n| **{step}**| {eval_metrics['loss']:.3f} | {eval_metrics['wer']:.3f} |{eval_metrics['cer']:.3f} |"

    # Put predictions into a table
    inference_df = pd.DataFrame(columns=['target', 'prediction'])
   

    for pred_text, label_text in zip(predictions, labels):
        formatted_label_text, formatted_pred_text = format_diff(
            label_text, pred_text)
        new_row = pd.DataFrame(
            {'target': formatted_label_text, 'prediction': formatted_pred_text}, index=[0])
        inference_df = pd.concat(
            [inference_df, new_row], ignore_index=True)
    
    # Create the prediction table of the first N rows
    inference_df = inference_df[['target', 'prediction']]
    predict_table = inference_df.to_markdown(index=False)

    # Build the markdown page
    markdown_str = f"{eval_table}\n\n{predict_table}"
    

    # Save the stats file
    stats_file_name = f"{predictions_folder_name}/step_{step}.md"
    with open(stats_file_name, "w") as f:
        f.write(markdown_str)

    # Create an header for all the files
    md_files = sorted(os.path.basename(file) for file in os.listdir(
        predictions_folder_name) if file.startswith("step_"))
    sorted_md_files = sorted(
        md_files, key=lambda x: int(x[0:-3].split("_")[1]))
    md_header = " | ".join(
        f"[Step {file[:-3].split('_')[1]}]({file})" for file in sorted_md_files)

    # Add this header to all the stats file in the folder
    for filename in os.listdir(predictions_folder_name):
        if filename.startswith("step_"):
            with open(os.path.join(predictions_folder_name, filename), "r+") as f:
                content = f.read()
                new_content = md_header + "\n\n" + \
                    content[content.index("| STEP| loss | wer"):]
                f.seek(0)
                f.write(new_content)
                f.truncate()
                

    logger.info(
        f"Created {stats_file_name} and updated the headers of the other stats files")
