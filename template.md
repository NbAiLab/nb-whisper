---
license: cc-by-4.0
language:
- 'no'
- nb
- nn
- en
datasets:
- NbAiLab/ncc_speech
- NbAiLab/NST
- NbAiLab/NPSC
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

# NB-Whisper #Size# (Release Candidate)
Please not that these are still only Release Candidates. We are now doing the final testing, and if all goes well, we will release models later this month.

This is the **_Norwegian NB-Whisper #Size# model_** released by the National Library of Norway. NB-Whisper is a series of models for automatic speech recognition (ASR) and speech translation, building upon the foundation laid by [OpenAI's Whisper](https://arxiv.org/abs/2212.04356). All models are trained for a total of 250.000 steps on a dataset consisting of 8M samples of aligned 30 second audio clips. This amounts to totally 66.000 hours of speech. For more details about the training set, please see our forthcoming article.

<center>
  <figure>
    <video controls>
      <source src="https://huggingface.co/NbAiLab/nb-whisper-small-beta/resolve/main/king.mp4" type="video/mp4">
    Your browser does not support the video tag.
    </video>   
  <figcaption><a href="https://www.royalcourt.no/tale.html?tid=137662&sek=28409&scope=27248" target="_blank">Speech given by His Majesty The King of Norway at the garden party hosted by Their Majesties The King and Queen at the Palace Park on 1 September 2016.</a>Transcribed using the Small model.</figcaption>
</figure> 
</center>


## Model Details

NB-Whisper models will be available in five different sizes:

| Model Size | Parameters | Main Model | Verbatim version |  Semantic version |
|------------|------------|--------------|--------------|--------------|
| tiny       | 39M        | [NB-Whisper Tiny](https://huggingface.co/NbAiLabBeta/nb-whisper-tiny) |[Tiny - verbatim](https://huggingface.co/NbAiLabBeta/nb-whisper-tiny-verbatim) |[Tiny - semantic](https://huggingface.co/NbAiLabBeta/nb-whisper-tiny-semantic) |
| base       | 74M        | [NB-Whisper Base](https://huggingface.co/NbAiLabBeta/nb-whisper-base) |[Base - verbatim](https://huggingface.co/NbAiLabBeta/nb-whisper-base-verbatim) |[Base - semantic](https://huggingface.co/NbAiLabBeta/nb-whisper-base-semantic) |
| small      | 244M       | [NB-Whisper Small](https://huggingface.co/NbAiLabBeta/nb-whisper-small) |[Small - verbatim](https://huggingface.co/NbAiLabBeta/nb-whisper-small-verbatim) |[Small - semantic](https://huggingface.co/NbAiLabBeta/nb-whisper-small-semantic) |
| medium     | 769M       | [NB-Whisper Medium](https://huggingface.co/NbAiLabBeta/nb-whisper-medium) |[Medium - verbatim](https://huggingface.co/NbAiLabBeta/nb-whisper-medium-verbatim) |[Medium - semantic](https://huggingface.co/NbAiLabBeta/nb-whisper-medium-semantic) |
| large      | 1550M      | [NB-Whisper Large](https://huggingface.co/NbAiLabBeta/nb-whisper-large) |[Large - verbatim](https://huggingface.co/NbAiLabBeta/nb-whisper-large-verbatim) |[Large - semantic](https://huggingface.co/NbAiLabBeta/nb-whisper-large-semantic) |


Please refer to the OpenAI Whisper model card for more details about the backbone model.

### Model Description

- **Developed by:** [NB AI-Lab](https://ai.nb.no/)
- **Shared by:** [NB AI-Lab](https://ai.nb.no/)
- **Model type:** `whisper`
- **Language(s) (NLP):** Norwegian, Norwegian Bokmål, Norwegian Nynorsk, English
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Finetuned from model:** [openai/whisper-small](https://huggingface.co/openai/whisper-small)

### Model Sources
- **Code Repository:** https://github.com/NbAiLab/nb-whisper/
- **Paper:** _Coming soon_
- **Demo:** _See Spaces on this page_

## How to use the models
### Online Demos
It is possible to run the models directly from the HuggingFace Inference API on the right side of this page. However, this means that the model will first need to load and then it will run on a very small CPU. It is very slow. For a few days, we will however host some of the models on TPUs. This is a lot faster. Take a look under **Spaces** on the [Main Page](https://huggingface.co/NbAiLabBeta/).

### HuggingFace
You can download the models, and run them locally. The Tiny, Base and Small models are well suited to run on CPU. For Medium and Large we recommend a computer/server with a graphical card (GPU). Using the models with Transformers from HuggingFace is trivial as long as you have [Python](https://www.python.org/downloads/) installed on your computer. The examples are using this [sample mp3-file]([audio/king.mp3](https://github.com/NbAiLab/nb-whisper/raw/main/audio/king.mp3)).

```bash
# Download the sample file
> wget -N https://github.com/NbAiLab/nb-whisper/raw/main/audio/king.mp3

# Install necessary libraries. 
> pip install transformers>=4.35.2
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

Examining the output, we see that there are multiple repetitions in the end. This is because the default length is 30 seconds and the video is 1:25 minutes. By passing the `chunk_lengt_s` argument, we can transcribe longer file.

```python
asr("king.mp3", chunk_length_s=30, generate_kwargs={'task': 'transcribe', 'language': 'no'})
```
<details>
<summary>Expected output</summary>

```json
{
{'text': ' Nordmenn er nordlendinger, trøndere, sørlendinger og folk fra alle andre regioner. Nordmenn er også innvandret fra Afghanistan, Pakistan, Polen, Sverige, Somalia og Syria. Det er ikke alltid så lett å si hvor vi er fra, hvilken nasjonalitet vi er fra. Hvilken nasjonalitet vi er fra. Hvilken nasjonalitet vi er fra. Hvilken nasjonalitet vi er fra. Hvilken nasjonalitet vi er fra. Hvilken nasjonalitet vi er fra, hvilken nasjonalitet vi tilhører. Det vi kaller hjem, er der hjertet vårt er, og det kan ikke alltid plasseres innenfor landegrenser. Nordmenn er jenter som er glad i jenter, gutter som er glad i gutter, og jenter og gutter som er glad i hverandre. Nordmenn trommer på Gud, Allah, Altet og ingenting. Nordmenn liker Grieg, Kygo, Helbilis og Kari Bremnes. Med andre ord, Norge er dere. Norge er oss. Mitt største håp for Norge er at vi skal klare å ta vare på hverandre, at vi skal bygge dette landet videre på tillit, fellesskap og raushet.'}
```
</details>

Here the output looks a lot better. We can also ask the model to output timestamps:
```python
asr("king.mp3", chunk_length_s=30, return_timestamps=True, generate_kwargs={'task': 'transcribe', 'language': 'no'})
```
<details>
<summary>Expected output</summary>

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

Some other cool features to look into:
```python
# Transcribe to Nynorsk
asr("king.mp3", chunk_length_s=30, generate_kwargs={'task': 'transcribe', 'language': 'nn'})
```
<details>
<summary>Expected output</summary>

```json
{
  "text": "Nordmenn er nordlendingar, trøndarar, sørlendingar og folk frå alle andre regionar. Nordmenn er også innvandra frå Afghanistan, Pakistan, Polen, Sverige, Somalia og Syria. Det er ikkje alltid så lett å seie kvar vi er frå, kva nasjonalitet vi tilhøyrer. Det vi kallar heim, er der hjartet vårt er, og det kan ikkje alltid plasserast innanfor landegrenser. Nordmenn er jenter som er glad i jenter, gutar som erade i gutar, og jenter og gutar som er glade i kvarandre. Nordmenn trommar på Gud, Allah, Altet og ingenting. Nordmenn liker Grieg, Kygo, Helbiles og Kari Bremnes. Med andre ord, Noreg er dere! Noreg er oss. Mitt største håp for Noreg er at vi skal klare å ta vare på kvarandre, at vi skal byggje dette landet vidare på tillit, fellesskap og raushet."
}
```
</details>

```python
# Transcribe to English
asr("king.mp3", chunk_length_s=30, generate_kwargs={'task': 'transcribe', 'language': 'en'})
```
<details>
<summary>Expected output</summary>

```json
{
  "text": "Norwegians are Norwegians, trønders, southerners and people from all other regions. Norwegians are also invaded from Afghanistan, Pakistan, Poland, Sweden, Somalia and Suria. It is not always so easy to say where we are from, what nationality we belong to. What we call home is where our heart is, and it cannot always be placed within national borders. Norwegians are girls who like girls, boys who like boys, and girls and boys who like each other. Norwegians thrump on God, Allah, Altet and nothing. Norwegians like Grieg, Kygo, Helbilis and Kari Bremnes. In other words, Norway is you. Norway is us. My biggest hope for Norway is that we should be able to take care of each other, that we should build this country on trust, community and generosity."
}
```
</details>

```python
# Return Word Level Timestamps
asr("king.mp3", chunk_length_s=30, return_timestamps="word", generate_kwargs={'task': 'transcribe', 'language': 'no'})
```

<details>
<summary>Expected output</summary>

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
</details>

### Whisper CPP
Whisper CPP is a C++ implementation of the Whisper model.  The C++ version aims to provide the same functionalities while leveraging the efficiency and performance optimizations inherent in C++. This allows you to embed any Whisper model into a binary file, and build real applications around this. It does however require some knowledge about compiling C++ programs. Their [homepage](https://github.com/ggerganov/whisper.cpp) gives you several examples of how to build applications, including real-time transription.

We have converted this model to the ggml-format needed to build Whisper CPP binaries. You can download the file [here](blob/main/ggml-model.bin).

### API
Check instructions within the demoes in Spaces for how to access the models through a simple API. Please note that the demoes are actually demoes, and will just be available a few weeks. 

## Training Data
Trained data comes from Språkbanken and the digital collection at the National Library of Norway. Training data includes:

- NST Norwegian ASR Database (16 kHz), and its corresponding dataset
- Transcribed speeches from the Norwegian Parliament produced by Språkbanken
- TV broadcast (NRK) subtitles (NLN digital collection)
- Audiobooks (NLN digital collection)


### Downstream Use

The models (and in particular the small sizes) are still known to show some hallucinations, as well as a tendency to drop part of the transcript from time to time. Please also note that the transcripts are typically not word by word. Spoken language and written language are often very different, and the model aims to "translate" spoken utterances into grammatically correct written sentences. We strongly believe that the best way to understand these models is to try them yourself.

A significant part of the training material comes from TV subtitles. Subtitles often shorten the content to make it easier to read. Typically, non-essential parts of the utterance can be also dropped. In some cases, this is a desired ability, in other cases, this is undesired. The final release of these model will provida a mechanism to control for this beaviour.



## Bias, Risks, and Limitations

Production use without adequate assessment of risks and mitigation may be considered irresponsible or harmful. These models may have bias and/or any other undesirable distortions. When third parties, deploy or provide systems and/or services to other parties using any of these models (or using systems based on these models) or become users of the models, they should note that it is their responsibility to mitigate the risks arising from their use and, in any event, to comply with applicable regulations, including regulations regarding the use of artificial intelligence. In no event shall the owner of the models (The National Library of Norway) be liable for any results arising from the use made by third parties of these models.


#### Software
The model is trained using Jax/Flax. The final model is then converted to Pytorch, Tensorflow, whisper.cpp and ONXX. You can find all these formats under `Files and versions`. Please tell us if you would like the models to be converted to other format. 

## Citation & Contributors
The development of this model was part of the contributors' professional roles at the National Library of Norway, under the _NoSTram_ project led by _Per Egil Kummervold (PEK)_. The Jax code, dataset loaders, and training scripts were collectively designed by _Javier de la Rosa (JdlR)_, _Freddy Wetjen (FW)_, _Rolv-Arild Braaten (RAB)_, and _PEK_. Primary dataset curation was handled by _FW_, _RAB_, and _PEK_, while _JdlR_ and _PEK_ crafted the documentation. The project was completed under the umbrella of AiLab, directed by _Svein Arne Brygfjeld_. 

All contributors played a part in shaping the optimal training strategy for the Norwegian ASR model based on the Whisper architecture. 

_A paper detailing our process and findings is underway!_


## Acknowledgements

Thanks to [Google TPU Research Cloud](https://sites.research.google/trc/about/) for supporting this project with extensive training resources. Thanks to Google Cloud for supporting us with credits for translating large parts of the corpus. A special thanks to [Sanchit Ghandi](https://huggingface.co/sanchit-gandhi) at HuggingFace for providing thorough technical advice in debugging and with the work of getting this to train on Google TPUs. A special thanks to Per Erik Solberg at Språkbanken for the collaboration with regard to the Stortinget corpus.

## Contact
Please do not hesitate to contact us with any experiences, insights, or suggestions that you may have. Your input is invaluable in helping us to improve the model and ensure that it effectively serves the needs of users. Whether you have technical concerns, usability suggestions, or ideas for future enhancements, we welcome your input. Thank you for participating in this critical stage of our model's development.

If you intend to incorporate this model into your research, we kindly request that you reach out to us. We can provide you with the most current status of our upcoming paper, which you can cite to acknowledge and provide context for the work done on this model. 


Please use this email as the main contact point, it is read by the entire team: <a rel="noopener nofollow" href="mailto:ailab@nb.no">ailab@nb.no</a>.
