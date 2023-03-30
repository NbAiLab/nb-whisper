# SCREAM 
## Septimus Experiments

These experiments constitute the seventh series of trials on the new Norwegian Whisper Model ("Scream"), based on the NCC-S corpus series. The objective of these experiments is to explore the scalability of the model on TPU pods, while also examining the impact of adding the NPSC corpus to the training for the first time.

The Tiny models are trained on a TPU Pod v4-32 with a batch size of 512, while the Large models are trained on a TPU Pod v4-64 with a batch size of 160. The Tiny models aim to compare the use of the nrk_v5 corpus to using all_v5 corpus. The all_v5 corpus includes the first version of NPSC+ corpus and the NST corpus in addition the the NRK subtitles.

The Large model experiments aim to investigate the impact of scaling and determine the optimal learning rate, using 2e6 and 8e6. All models use linear decay.


## Experiments
|        | NRK | All |
|--------|----------|--------|
| Tiny   | [Experiment 1](https://huggingface.co/NbAiLab/scream_tiny_sept_nrk)        | [Experiment 2](https://huggingface.co/NbAiLab/scream_tiny_sept_nrk)      |


|        | 2e6 | 8e6 |
|--------|----------|--------|
| Large  | [Experiment 3](https://huggingface.co/NbAiLab/scream_large_sept_2e6)        | [Experiment 4](https://huggingface.co/NbAiLab/scream_large_sept_8e6)      |


In addition a set of smaller tests are run on the tpu v4-8. The purpose is looking more closely at learning rate and scheduler.

|        | Experiment |
|--------|----------|
| Linear 1e5  | [Experiment 5](https://huggingface.co/NbAiLab/scream_tiny_sept_1e5_constant)        | 
| Linear 2e5  | [Experiment 6](https://huggingface.co/NbAiLab/scream_tiny_sept_2e5_linear)        | 
| Linear 3e5  | [Experiment 7](https://huggingface.co/NbAiLab/scream_tiny_sept_3e5_linear)        | 
| Constant 1e5  | [Experiment 8](https://huggingface.co/NbAiLab/scream_tiny_sept_1e5_constant)        | 



For more details about the training parameters, see the tranings scripts.