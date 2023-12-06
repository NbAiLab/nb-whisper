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
You can download the models, and run them locally. The Tiny, Base and Small models are well suited to run on CPU. For Medium and Large we recommend a computer/server with a graphical card (GPU). Using the models with Transformers from HuggingFace is trivial as long as you have [Python](https://www.python.org/downloads/) installed on your computer. The examples are using this [sample mp3-file]([audio/king.mp3](https://github.com/NbAiLab/nb-whisper/raw/main/audio/king.mp3)),

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
asr = pipeline("automatic-speech-recognition","#model_name#")

#transcribe
asr("king.mp3")

```

<details>
<summary>Expected output</summary>

```json
{
  "text": "Så mange anga kører seg i så viktig sak, så vi får du kører det tilbake med. Om kabaret gudam i at vi skal hjælge. Kør seg vi gjør en uda? Nei noe skal å abelistera sonvorne skrifer. Det er sak, så kjent det bare handling i samtatsen til bargører. Trudet første lask. På den å først så å køre og en gange samme, og så får vi gjør å vorte vorte vorte når vi kjent dit."
}
```
<\details>

Timestamps can also be retrieved by passing in the right parameter.

```python
asr(
    "audio.mp3",
    generate_kwargs={'task': 'transcribe', 'language': 'no'},
    return_timestamps=True,
)
# {'text': ' at så mange angar til seg så viktig sak, så vi får jo kjølget klare tilbakemeldingen om hva valget dem gjør at vi skal gjøre. Hva skjer vi gjøre nå da? Nei, nå skal jo administrationen vår skrivferdige sak, så kjem til behandling i samfærdshetshøyvalget, tror det første 
#  r. Først så kan vi ta og henge dem kjemme, og så får vi gjøre vårt valget når vi kommer dit.',
#  'chunks': [{'timestamp': (0.0, 5.34),
#    'text': ' at så mange angar til seg så viktig sak, så vi får jo kjølget klare tilbakemeldingen om'},
#   {'timestamp': (5.34, 8.64),
#    'text': ' hva valget dem gjør at vi skal gjøre.'},
#   {'timestamp': (8.64, 10.64), 'text': ' Hva skjer vi gjøre nå da?'},
#   {'timestamp': (10.64, 17.44),
#    'text': ' Nei, nå skal jo administrationen vår skrivferdige sak, så kjem til behandling i samfærdshetshøyvalget,'},
#   {'timestamp': (17.44, 19.44), 'text': ' tror det første år.'},
#   {'timestamp': (19.44, 23.94),
#    'text': ' Først så kan vi ta og henge dem kjemme, og så får vi gjøre vårt valget når vi kommer dit.'}]}
```

### API
Check instructions within the demoes in Spaces for how to access these through a simple API. Please note that the demoes are actually demoes, and will just be available a few weeks. 

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

This is a public beta that is not intended for production. Production use without adequate assessment of risks and mitigation may be considered irresponsible or harmful. These models may have bias and/or any other undesirable distortions. When third parties, deploy or provide systems and/or services to other parties using any of these models (or using systems based on these models) or become users of the models, they should note that it is their responsibility to mitigate the risks arising from their use and, in any event, to comply with applicable regulations, including regulations regarding the use of artificial intelligence. In no event shall the owner of the models (The National Library of Norway) be liable for any results arising from the use made by third parties of these models.


#### Software
The model is trained using Jax/Flax. The final model is then converted to Pytorch, Tensorflow, whisper.cpp and ONXX. Please tell us if you would like future models to be converted to other format. 

## Citation & Contributors
The development of this model was part of the contributors' professional roles at the National Library of Norway, under the _NoSTram_ project led by _Per Egil Kummervold (PEK)_. The Jax code, dataset loaders, and training scripts were collectively designed by _Javier de la Rosa (JdlR)_, _Freddy Wetjen (FW)_, _Rolv-Arild Braaten (RAB)_, and _PEK_. Primary dataset curation was handled by _FW_, _RAB_, and _PEK_, while _JdlR_ and _PEK_ crafted the documentation. The project was completed under the umbrella of AiLab, directed by _Svein Arne Brygfjeld_. 

All contributors played a part in shaping the optimal training strategy for the Norwegian ASR model based on the Whisper architecture. 

_A paper detailing our process and findings is underway!_


## Acknowledgements

Thanks to [Google TPU Research Cloud](https://sites.research.google/trc/about/) for supporting this project with extensive training resources. Thanks to Google Cloud for supporting us with credits for translating large parts of the corpus. A special thanks to [Sanchit Ghandi](https://huggingface.co/sanchit-gandhi) for providing thorough technical advice in debugging and with the work of getting this to train on Google TPUs. A special thanks to Per Erik Solberg at Språkbanken for the collaboration with regard to the Stortinget corpus.

## Contact
By releasing this ASR Whisper model  to gather constructive feedback on its performance. Please do not hesitate to contact us with any experiences, insights, or suggestions that you may have. Your input is invaluable in helping us to improve the model and ensure that it effectively serves the needs of users. Whether you have technical concerns, usability suggestions, or ideas for future enhancements, we welcome your input. Thank you for participating in this critical stage of our model's development.

If you intend to incorporate this model into your research, we kindly request that you reach out to us. We can provide you with the most current status of our upcoming paper, which you can cite to acknowledge and provide context for the work done on this model. 


Please use this email as the main contact point, it is read by the entire team: <a rel="noopener nofollow" href="mailto:ailab@nb.no">ailab@nb.no</a>
