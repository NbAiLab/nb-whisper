---
title: Whisper Word-Level Timestamps
emoji: 💭⏰
colorFrom: yellow
colorTo: indigo
sdk: gradio
sdk_version: 3.35.2
app_file: app.py
pinned: false
license: apache-2.0
---

Splash Video Creator
====================

Usage: app.py [-h] [-f FILE] [-m MODEL] [-o OUTPUT]

Create a video named OUTPUT with word by word transcriptions using FILE and MODEL. With no params will launch a Gradio demo.

Options:
-  -h, --help            show this help message and exit
-  -f FILE, --file FILE  Audio file
-  -m MODEL, --model MODEL
                        Model name or path
-  -o OUTPUT, --output OUTPUT
                        Output file name

Example
```bash
python app.py --file examples/lullaby.mp3 -m "openai/whisper-small" -o video.mp4
```

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
