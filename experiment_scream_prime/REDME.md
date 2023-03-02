# Scream 
## Primordial Experiments

These are the first experiments on the new Norwegian Whisper Model ("Scream"). The first set of experiments consists of 8 models all trained from the pretrained Whisper models and all following the same learning rate as specified in the [Whisper article](https://cdn.openai.com/papers/whisper.pdf).

All models are trained on a single TPU Pod v4-8 with 128GB RAM. The tiny and small model is trained with a batch of 256 as in the Whisper article. The Small model is trained with a batch size of 144. 

|        | constant | linear |
|--------|----------|--------|
| tiny   | 1        | 2      |
| base   | 3, 7, 8  | 4      |
| small  | 5        | 6      |


* Exp1-6 is trained on the [NCC3NRK-corpus](https://huggingface.co/datasets/NbAiLab/NCC_S3_nrk)
* Exp7 is trained on the [NCC3-corpus](https://huggingface.co/datasets/NbAiLab/NCC_S3)
* Exp8 is trained on the [NCC-corpus](https://huggingface.co/datasets/NbAiLab/NCC_S)


For more details about the training parameters, see the tranings scripts.