import argparse
import uuid
from functools import partial

import gradio as gr
import librosa
import moviepy.editor as mpy
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline


max_duration = 360  # seconds
fps = 25
video_width = 640
video_height = 480
margin_left = 20
margin_right = 20
margin_top = 20
line_height = 44

background_image = Image.open("nb-background.png")
font = ImageFont.truetype("Lato-Regular.ttf", 40)
text_color = (255, 200, 200)
highlight_color = (255, 255, 255)

# checkpoint = "openai/whisper-tiny"
# checkpoint = "openai/whisper-base"
checkpoint = "NbAiLabBeta/nb-whisper-small-v0.2"


def load_pipe(checkpoint):
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        from transformers import (
            AutomaticSpeechRecognitionPipeline,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            checkpoint
        ).to("cuda")  #.half()
        processor = WhisperProcessor.from_pretrained(checkpoint)
        pipe = AutomaticSpeechRecognitionPipeline(
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            batch_size=1,
            # torch_dtype=torch.float16,
            device="cuda:0"
        )
    else:
        pipe = pipeline(model=checkpoint)

    # TODO: no longer need to set these manually once the models have been updated on the Hub
    # whisper-tiny
    # pipe.model.generation_config.alignment_heads = [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]
    # whisper-base
    # pipe.model.generation_config.alignment_heads = [[3, 1], [4, 2], [4, 3], [4, 7], [5, 1], [5, 2], [5, 4], [5, 6]]
    # whisper-small
    #pipe.model.generation_config.alignment_heads = [[5, 3], [5, 9], [8, 0], [8, 4], [8, 7], [8, 8], [9, 0], [9, 7], [9, 9], [10, 5]]
    print(pipe.model.generation_config.alignment_heads)
    return pipe


chunks = []

start_chunk = 0
last_draws = []
last_image = None

def make_frame(t):
    global chunks, start_chunk, last_draws, last_image

    image = background_image.copy()
    draw = ImageDraw.Draw(image)

    # for debugging: draw frame time
    #draw.text((20, 20), str(t), fill=text_color, font=font)

    space_length = draw.textlength(" ", font)
    x = margin_left
    y = margin_top

    # Create a list of drawing commands
    draws = []
    for i in range(start_chunk, len(chunks)):
        chunk = chunks[i]
        chunk_start = chunk["timestamp"][0]
        chunk_end = chunk["timestamp"][1]
        if chunk_start > t: break
        if chunk_end is None: chunk_end = max_duration

        word = chunk["text"]
        word_length = draw.textlength(word + " ", font) - space_length

        if x + word_length >= video_width - margin_right:
            x = margin_left
            y += line_height

            # restart page when end is reached
            if y >= margin_top + line_height * 7:
                start_chunk = i
                break

        highlight = (chunk_start <= t < chunk_end)
        draws.append([x, y, word, word_length, highlight])

        x += word_length + space_length

    # If the drawing commands didn't change, then reuse the last image,
    # otherwise draw a new image
    if draws != last_draws:
        for x, y, word, word_length, highlight in draws:
            if highlight:
                color = highlight_color
                draw.rectangle([
                    x, y + line_height, x + word_length, y + line_height + 4
                ], fill=color)
            else:
                color = text_color

            draw.text((x, y), word, fill=color, font=font)

        last_image = np.array(image)
        last_draws = draws
    if last_image is None:
        last_image = np.array(image)
    return last_image


def predict(audio_path, checkpoint, filename=None):
    global chunks, start_chunk, last_draws, last_image

    start_chunk = 0
    last_draws = []
    last_image = None

    audio_data, sr = librosa.load(audio_path, mono=True)
    duration = librosa.get_duration(y=audio_data, sr=sr)
    duration = min(max_duration, duration)
    audio_data = audio_data[:int(duration * sr)]

    # Get pipe
    pipe = load_pipe(checkpoint)

    # Run Whisper to get word-level timestamps.
    audio_inputs = librosa.resample(
        audio_data, orig_sr=sr, target_sr=pipe.feature_extractor.sampling_rate
    )
    output = pipe(
        audio_inputs,
        chunk_length_s=30,
        stride_length_s=[4, 2],
        return_timestamps="word",
        generate_kwargs=dict(language="<|no|>")
    )
    chunks = output["chunks"]
    print(output)

    # Create the video.
    clip = mpy.VideoClip(make_frame, duration=duration)
    audio_clip = mpy.AudioFileClip(audio_path).set_duration(duration)
    clip = clip.set_audio(audio_clip)
    if not filename:
        filename = f"{str(uuid.uuid4())}_video.mp4"
    clip.write_videofile(filename, fps=fps, codec="libx264", audio_codec="aac")
    return filename

title = "Word-level timestamps with Whisper"
description = f"""This demo shows Whisper <b>word-level timestamps</b> in action using Hugging Face Transformers. It creates a video showing subtitled audio with the current word highlighted. It can even do music lyrics!

<!-- This demo uses the <b>NbAiLab/scream_small_beta</b> checkpoint. -->

Since it's only a demo, the output is limited to the first {max_duration} seconds of audio.
"""
article = """<div style='margin:20px auto;'>
<p>Credits:<p>

<ul>
<li><a href="https://freesound.org/people/petrsvar/sounds/254085/">Norwegian folk song, recorded in traditional hut in Nordmarka near Oslo</a></li>
<li><a href="https://freesound.org/people/Engangskameraten/sounds/179028/">4-year old Jeppe singing a Norwegian lullaby about being nice to insects</a></li>
<li>Lato font by Łukasz Dziedzic (licensed under Open Font License)</li>
</ul>

</div>
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create a video named OUTPUT with word by word transcriptions '
                    'using FILE and MODEL.\nWith no params will launch a Gradio demo.')
    parser.add_argument('-f','--file', help='Input file name', required=False)
    parser.add_argument('-m','--model', help='Model name or path', required=False)
    parser.add_argument('-o','--output', help='Output file name', required=False)
    args = vars(parser.parse_args())

    if not args["file"] and not args["model"] and not args["output"]:
        examples = [
            ["examples/folk_song.mp3", checkpoint],
            ["examples/lullaby.mp3", checkpoint],
        ]
        gr.Interface(
            fn=predict,
            inputs=[
                gr.Audio(label="Upload Audio", source="upload", type="filepath"),
                gr.Textbox(label="Model", value=checkpoint),
            ],
            outputs=[
                gr.Video(label="Output Video"),
            ],
            title=title,
            description=description,
            article=article,
            examples=examples,
        ).launch(share=True)
    else:
        print(predict(args["file"], args["model"] or checkpoint, args["output"]))