---
license: apache-2.0
language:
- 'no'
- nb
- nn
- en
datasets:
- NbAiLab/ncc_speech
- NbAiLab/NST
- NbAiLab/NPSC
base_model: openai/whisper-#size#
tags:
- audio
- asr
- automatic-speech-recognition
- hf-asr-leaderboard
metrics:
- wer
- cer
library_name: transformers
pipeline_tag: automatic-speech-recognition
widget:
- src: https://datasets-server.huggingface.co/assets/google/fleurs/--/nb_no/train/1/audio/audio.mp3
  example_title: FLEURS sample 1
- src: https://datasets-server.huggingface.co/assets/google/fleurs/--/nb_no/train/4/audio/audio.mp3
  example_title: FLEURS sample 2
---
#Finetuned#

# NB-Whisper #Size# (Release Candidate)

**IMPORTANT:** These models are currently Release Candidates. We are in the final stages of testing. If everything proceeds smoothly, we plan to officially release the models later this month.

Introducing the **_Norwegian NB-Whisper #Size# model_**, proudly developed by the National Library of Norway. NB-Whisper is a cutting-edge series of models designed for automatic speech recognition (ASR) and speech translation. These models are based on the work of [OpenAI's Whisper](https://arxiv.org/abs/2212.04356). Each model in the series has been trained for 250,000 steps, utilizing a diverse dataset of 8 million samples. These samples consist of aligned audio clips, each 30 seconds long, culminating in a staggering 66,000 hours of speech. For an in-depth understanding of our training methodology and dataset composition, keep an eye out for our upcoming article.

| Model Size | Parameters | Model |
|------------|------------|------------|
| Tiny       | 39M        | [NB-Whisper Tiny](https://huggingface.co/NbAiLabBeta/nb-whisper-tiny) |
| Base       | 74M        | [NB-Whisper Base](https://huggingface.co/NbAiLabBeta/nb-whisper-base) |
| Small      | 244M       | [NB-Whisper Small](https://huggingface.co/NbAiLabBeta/nb-whisper-small) |
| Medium     | 769M       | [NB-Whisper Medium](https://huggingface.co/NbAiLabBeta/nb-whisper-medium) |
| Large      | 1550M      | [NB-Whisper Large](https://huggingface.co/NbAiLabBeta/nb-whisper-large) |



### Specialised Models
While the main models are suitable for most transcription task, we demonstrate how easy it is to change the output of the main model. The following models are trained 250 additional steps from the main models above, and might be suitable for more targetted use cases:
- **Verbatim version**: This lower-cased variant is more literal and suitable for tasks requiring detailed transcription, such as linguistic analysis.
- **Semantic version**: This variant focuses less on verbatim accuracy but captures the essence of content, ideal for meeting minutes and subtitling.


| Model Size | Parameters | Verbatim version | Semantic version |
|------------|------------|------------|------------------|
| Tiny       | 39M        | [Tiny - verbatim](https://huggingface.co/NbAiLabBeta/nb-whisper-tiny-verbatim) | [Tiny - semantic](https://huggingface.co/NbAiLabBeta/nb-whisper-tiny-semantic) |
| Base       | 74M        | [Base - verbatim](https://huggingface.co/NbAiLabBeta/nb-whisper-base-verbatim) | [Base - semantic](https://huggingface.co/NbAiLabBeta/nb-whisper-base-semantic) |
| Small      | 244M       | [Small - verbatim](https://huggingface.co/NbAiLabBeta/nb-whisper-small-verbatim) | [Small - semantic](https://huggingface.co/NbAiLabBeta/nb-whisper-small-semantic) |
| Medium     | 769M       | [Medium - verbatim](https://huggingface.co/NbAiLabBeta/nb-whisper-medium-verbatim) | [Medium - semantic](https://huggingface.co/NbAiLabBeta/nb-whisper-medium-semantic) |
| Large      | 1550M      | [Large - verbatim](https://huggingface.co/NbAiLabBeta/nb-whisper-large-verbatim) | [Large - semantic](https://huggingface.co/NbAiLabBeta/nb-whisper-large-semantic) |


### Model Description

- **Developed by:** [NB AI-Lab](https://ai.nb.no/)
- **Shared by:** [NB AI-Lab](https://ai.nb.no/)
- **Model type:** `whisper`
- **Language(s) (NLP):** Norwegian, Norwegian Bokmål, Norwegian Nynorsk, English
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Trained from model:** [openai/whisper-#size#](https://huggingface.co/openai/whisper-#size#)
- **Code Repository:** https://github.com/NbAiLab/nb-whisper/
- **Paper:** _Coming soon_
- **Demo:** _See Spaces on this page_

## Example transcription
<center>
  <figure>
    <video controls>
      <source src="https://huggingface.co/NbAiLab/nb-whisper-small-beta/resolve/main/king.mp4" type="video/mp4">
    Your browser does not support the video tag.
    </video>   
  <figcaption><a href="https://www.royalcourt.no/tale.html?tid=137662&sek=28409&scope=27248" target="_blank">Speech given by His Majesty The King of Norway at the garden party hosted by Their Majesties The King and Queen at the Palace Park on 1st of September 2016.</a> Transcribed using the Small model.</figcaption>
</figure> 
</center>


## How to Use the Models

### Online Demos
You can try the models directly through the HuggingFace Inference API, accessible on the right side of this page. Be aware that initially, the model needs to load and will run on limited CPU capacity, which might be slow. To enhance your experience, we are temporarily hosting some models on TPUs for a few days, significantly boosting their performance. Explore these under the **Spaces** section on the [Main Page](https://huggingface.co/NbAiLabBeta/).

### Local Setup with HuggingFace
Alternatively, you can run the models locally. The Tiny, Base, and Small models are optimized for CPU execution. For the Medium and Large models, we recommend a system equipped with a GPU to ensure efficient processing. Setting up and using these models with HuggingFace's Transformers is straightforward, provided you have [Python](https://www.python.org/downloads/) installed on your machine. For practical demonstrations, refer to examples using this [sample mp3 file](https://github.com/NbAiLab/nb-whisper/raw/main/audio/king.mp3).

```bash
# Download the sample file
$ wget -N https://github.com/NbAiLab/nb-whisper/raw/main/audio/king.mp3

# Install necessary libraries. 
$ pip install transformers>=4.35.2
```

After this is done, you should be able to run this in Python:

```python
from transformers import pipeline

# Load the model
asr = pipeline("automatic-speech-recognition", "#model_name#")

#transcribe
asr("king.mp3", generate_kwargs={'task': 'transcribe', 'language': 'no'})

```

<details>
<summary>Expected output</summary>

```json
{
  {'text': ' Nordmenn er nordlendinger, trøndere, sørlendinger og folk fra alle andre regioner. Nordmenn er også innvandret fra Afghanistan, Pakistan, Polen, Sverige, Somalia og Syria. Det er ikke alltid så lett å si hvor vi er fra, hvilken nasjonalitet vi er fra. Hvilken nasjonalitet vi er fra. Hvilken nasjonalitet vi er fra. Hvilken nasjonalitet vi er fra. Hvilken nasjonalitet vi er fra. Hvilken nasjonalitet vi er fra. Hvilken nasjonalitet vi er fra.'}
}
```
</details>

### Extended HuggingFace
Examining the output, we see that there are multiple repetitions in the end. This is because the default length is 30 seconds and the video is 1:25 minutes. By passing the ```chunk_lengt_s``` argument, we can transcribe longer file. The examples below also illustrates how to transcribe to English or Nynorsk, and how to get timestamps for sentences and words.

```python
# Long Transcripts
asr("king.mp3", chunk_length_s=30, generate_kwargs={'task': 'transcribe', 'language': 'no'})

# Return Timestamps
asr("king.mp3", chunk_length_s=30, return_timestamps=True, generate_kwargs={'task': 'transcribe', 'language': 'no'})

# Return Word Level Timestamps
asr("king.mp3", chunk_length_s=30, return_timestamps="word", generate_kwargs={'task': 'transcribe', 'language': 'no'})

# Transcribe to Nynorsk
asr("king.mp3", chunk_length_s=30, generate_kwargs={'task': 'transcribe', 'language': 'nn'})

# Transcribe to English
asr("king.mp3", chunk_length_s=30, generate_kwargs={'task': 'transcribe', 'language': 'en'})

```
<details>
<summary>Expected output</summary>

```json
Long transcripts:
{
{'text': ' Nordmenn er nordlendinger, trøndere, sørlendinger og folk fra alle andre regioner. Nordmenn er også innvandret fra Afghanistan, Pakistan, Polen, Sverige, Somalia og Syria. Det er ikke alltid så lett å si hvor vi er fra, hvilken nasjonalitet vi er fra. Hvilken nasjonalitet vi er fra. Hvilken nasjonalitet vi er fra. Hvilken nasjonalitet vi er fra. Hvilken nasjonalitet vi er fra. Hvilken nasjonalitet vi er fra, hvilken nasjonalitet vi tilhører. Det vi kaller hjem, er der hjertet vårt er, og det kan ikke alltid plasseres innenfor landegrenser. Nordmenn er jenter som er glad i jenter, gutter som er glad i gutter, og jenter og gutter som er glad i hverandre. Nordmenn trommer på Gud, Allah, Altet og ingenting. Nordmenn liker Grieg, Kygo, Helbilis og Kari Bremnes. Med andre ord, Norge er dere. Norge er oss. Mitt største håp for Norge er at vi skal klare å ta vare på hverandre, at vi skal bygge dette landet videre på tillit, fellesskap og raushet.'}
```

Timestamps:
```json
{
{'text': ' Nordmenn er nordlendinger, trøndere, sørlendinger og folk fra alle andre regioner. Nordmenn er også innvandret fra Afghanistan, Pakistan, Polen, Sverige, Somalia og Syria. Det er ikke alltid så lett å si hvor vi er fra, hvilken nasjonalitet vi er fra. Hvilken nasjonalitet vi er fra. hvilken nasjonalitet vi tilhører. Det vi kaller hjem, er der hjertet vårt er, og det kan ikke alltid plasseres innenfor landegrenser. Nordmenn er jenter som er glad i jenter, gutter som er glad i gutter, og jenter og gutter som er glad i hverandre. Nordmenn trommer på Gud, Allah, Altet og ingenting. Nordmenn liker Grieg, Kygo, Helbiles og Kari Bremnes. Med andre ord, Norge er dere. Norge er oss. Mitt største håp for Norge er at vi skal klare å ta vare på hverandre, at vi skal bygge dette landet videre på tillit, fellesskap og raushet.',
 'chunks': [{'timestamp': (0.0, 5.46),
   'text': ' Nordmenn er nordlendinger, trøndere, sørlendinger'},
  {'timestamp': (5.52, 8.68), 'text': ' og folk fra alle andre regioner.'},
  {'timestamp': (8.68, 16.64),
   'text': ' Nordmenn er også innvandret fra Afghanistan, Pakistan, Polen, Sverige, Somalia og Syria.'},
  {'timestamp': (16.64, 13.3),
   'text': ' Det er ikke alltid så lett å si hvor vi er fra, hvilken nasjonalitet vi er fra.'},
  {'timestamp': (13.32, 30.28),
   'text': ' Hvilken nasjonalitet vi er fra. hvilken nasjonalitet vi tilhører.'},
  {'timestamp': (32.52, 39.16),
   'text': ' Det vi kaller hjem, er der hjertet vårt er, og det kan ikke alltid plasseres'},
  {'timestamp': (39.16, 42.0), 'text': ' innenfor landegrenser.'},
  {'timestamp': (42.0, 46.74),
   'text': ' Nordmenn er jenter som er glad i jenter, gutter som er glad i gutter,'},
  {'timestamp': (46.74, 51.12),
   'text': ' og jenter og gutter som er glad i hverandre.'},
  {'timestamp': (51.16, 57.42),
   'text': ' Nordmenn trommer på Gud, Allah, Altet og ingenting.'},
  {'timestamp': (57.42, 64.3),
   'text': ' Nordmenn liker Grieg, Kygo, Helbiles og Kari Bremnes.'},
  {'timestamp': (64.34, 71.24),
   'text': ' Med andre ord, Norge er dere. Norge er oss.'},
  {'timestamp': (71.24, 78.04),
   'text': ' Mitt største håp for Norge er at vi skal klare å ta vare på hverandre,'},
  {'timestamp': (78.12, 84.68),
   'text': ' at vi skal bygge dette landet videre på tillit, fellesskap og raushet.'}]}
```
</details>


Word Level Timestamps:
```json
{
  "text": "Nordmenn er nordlendinger, trøndere, sørlendinger og folk fra alle andre regioner. Nordmenn er også innvandret fra Afghanistan, Pakistan, Polen, Sverige, Somalia og Syria. Det er ikke alltid så lett å si hvor vi er fra, hvilken nasjonalitet vi tilhører. Det vi kaller hjem, er der hjertet vårt er, og det kan ikke alltid plasseres innenfor landegrenser. Nordmenn er jenter som er glad i jenter, gutter som er glad i gutter, og jenter og gutter som er glad i hverandre. Nordmenn trommer på Gud, Allah, Altet og ingenting. Nordmenn liker Grieg, Kygo, Helbilis og Kari Bremnes. Med andre ord, Norge er dere. Norge er oss. Mitt største håp for Norge er at vi skal klare å ta vare på hverandre, at vi skal bygge dette landet videre på tillit, fellesskap og raushet.",
  "chunks": [
    {"text": "Nordmenn", "timestamp": [0.72, 1.42]},
    {"text": "er", "timestamp": [1.42, 1.74]},
    // ... more chunks ...
    {"text": "raushet.", "timestamp": [83.1, 84.88]}
  ]
}
```

Nynorsk:
```json
{
  "text": "Nordmenn er nordlendingar, trøndarar, sørlendingar og folk frå alle andre regionar. Nordmenn er også innvandra frå Afghanistan, Pakistan, Polen, Sverige, Somalia og Syria. Det er ikkje alltid så lett å seie kvar vi er frå, kva nasjonalitet vi tilhøyrer. Det vi kallar heim, er der hjartet vårt er, og det kan ikkje alltid plasserast innanfor landegrenser. Nordmenn er jenter som er glad i jenter, gutar som erade i gutar, og jenter og gutar som er glade i kvarandre. Nordmenn trommar på Gud, Allah, Altet og ingenting. Nordmenn liker Grieg, Kygo, Helbiles og Kari Bremnes. Med andre ord, Noreg er dere! Noreg er oss. Mitt største håp for Noreg er at vi skal klare å ta vare på kvarandre, at vi skal byggje dette landet vidare på tillit, fellesskap og raushet."
}
```

English:
```json
{
  "text": "Norwegians are Norwegians, trønders, southerners and people from all other regions. Norwegians are also invaded from Afghanistan, Pakistan, Poland, Sweden, Somalia and Suria. It is not always so easy to say where we are from, what nationality we belong to. What we call home is where our heart is, and it cannot always be placed within national borders. Norwegians are girls who like girls, boys who like boys, and girls and boys who like each other. Norwegians thrump on God, Allah, Altet and nothing. Norwegians like Grieg, Kygo, Helbilis and Kari Bremnes. In other words, Norway is you. Norway is us. My biggest hope for Norway is that we should be able to take care of each other, that we should build this country on trust, community and generosity."
}
```

</details>

### Whisper CPP
Whisper CPP is a C++ implementation of the Whisper model, offering the same functionalities with the added benefits of C++ efficiency and performance optimizations. This allows embedding any Whisper model into a binary file, facilitating the development of real applications. However, it requires some familiarity with compiling C++ programs. Their [homepage](https://github.com/ggerganov/whisper.cpp) provides examples of how to build applications, including real-time transcription. 

We have converted this model to the ggml-format model used by Whisper CPP binaries. The file can be downloaded [here](blob/main/ggml-model.bin), and a `q5_0` quantized version is also available [here](blob/main/ggml-model-q5_0.bin).

```bash
# We can download and compile whisper.cpp
$ git clone --depth 1 https://github.com/ggerganov/whisper.cpp --branch v1.5.1
$ cd whisper.cpp/
$ make

# We also need to convert the audio to WAV as that is the only format supported by whisper.cpp
$ wget -N https://github.com/NbAiLab/nb-whisper/raw/main/audio/king.mp3
$ ffmpeg -i king.mp3 -ar 16000 -ac 1 -c:a pcm_s16le king.wav                                        

# And run it with the f16 default model
$ ./main -m /path/to/ggml-model.bin king.wav

# Or the quantized version
$ ./main -m /path/to/ggml-model-q5_0.bin king.wav
```

### API
Instructions for accessing the models via a simple API are included in the demos under Spaces. Note that these demos are temporary and will only be available for a few weeks.

## Training Data
The training data originates from Språkbanken and the National Library of Norway's digital collection, including:

- NST Norwegian ASR Database (16 kHz) and its corresponding dataset
- Transcribed speeches from the Norwegian Parliament by Språkbanken
- TV broadcast (NRK) subtitles (NLN digital collection)
- Audiobooks (NLN digital collection)

## Downstream Use

The models, especially the smaller ones, may exhibit occasional hallucinations and may drop parts of the transcript. They are designed to convert spoken language into grammatically correct written sentences, which might not always be word-for-word translations. We have made two extra model variant for users that want a different transcription style. We encourage users to try the models themselves to get a better understanding.

## Bias, Risks, and Limitations

Using these models without adequate risk assessment and mitigation could be considered irresponsible. They may contain biases or other undesirable distortions. Users who deploy these models or integrate them into systems or services are responsible for mitigating risks and complying with applicable AI regulations. The National Library of Norway, as the model owner, disclaims liability for any outcomes resulting from third-party use of these models.

### Software
The model was trained using Jax/Flax and converted to PyTorch, Tensorflow, whisper.cpp, and ONXX formats. These are available under `Files and versions`. We welcome requests for conversion to other formats. All training code and scripts are released under the Apache License 2.0 in the GitHub repository [nb-whisper](https://github.com/NbAiLab/nb-whisper/).

## Citation & Contributors
The NB-Whisper #Size# model is a product of the NoSTram project led by Per Egil Kummervold ([@pere](https://huggingface.co/pere)) at the National Library of Norway. Key contributors include Javier de la Rosa ([@versae](https://huggingface.co/versae)), Freddy Wetjen ([@freddyw](https://huggingface.co/freddyw)), and Rolv-Arild Braaten ([@Rolv-Arild](https://huggingface.co/Rolv-Arild)). NB AI-Lab, under the direction of Svein Arne Brygfjeld ([@Brygfjeld](https://huggingface.co/Brygfjeld)), supported the project's successful completion. A detailed paper on our process and findings is forthcoming.

## Disclaimer

The models published in this repository are intended for a generalist purpose and are available to third parties. These models may have bias and/or any other undesirable distortions. When third parties, deploy or provide systems and/or services to other parties using any of these models (or using systems based on these models) or become users of the models, they should note that it is their responsibility to mitigate the risks arising from their use and, in any event, to comply with applicable regulations, including regulations regarding the use of artificial intelligence. In no event shall the owner of the models (The National Library of Norway) be liable for any results arising from the use made by third parties of these models.

## Acknowledgements

Our gratitude extends to [Google TPU Research Cloud](https://sites.research.google/trc/about/) for training resources, Google Cloud for translation credits, and HuggingFace's Sanchit Ghandi for technical support. A special thank you to Per Erik Solberg at Språkbanken for the collaboration on the Stortinget corpus.

## Contact
For feedback, technical concerns, or collaboration inquiries, please contact <a rel="noopener nofollow" href="mailto:ailab@nb.no">ailab@nb.no</a>. If you plan to include this model in your research, contact us for the latest information on our upcoming paper for citation purposes.
