# SCREAM 
## Septimus Experiments

These are the sevent series of experiments on the new Norwegian Whisper Model ("Scream") based on the NCC-S corpus serie. This experiments tries to scale the work on the TPU pods. In addition it for the first time looks into the effect of adding the NPSC corpus to the training. 

The Tiny models here are trained on a TPU Pod v4-32 with a batch size of 512, while the Large models are trained on a TPU Pod v4-64 with a batch size of 160. The purpose of the Tiny models is to investigate the diffrence between using the nrk_v5 corpus compared to using all_v5 that also contains the first verison of NPSC+ and NST. 

The purpose of the Large model experiments are to investigate the effect on scaling, in addition to trying to get an idea of what is the best learning rate. The learning rates 2e6 and 8e6 are used. All models are using linear decay.


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
| Linear 1e5  | [Experiment 4](https://huggingface.co/NbAiLab/scream_tiny_sept_1e5_constant)        | 
| Linear 2e5  | [Experiment 5](https://huggingface.co/NbAiLab/scream_tiny_sept_2e5_linear)        | 
| Linear 3e5  | [Experiment 6](https://huggingface.co/NbAiLab/scream_tiny_sept_3e5_linear)        | 
| Constant 1e5  | [Experiment 7](https://huggingface.co/NbAiLab/scream_tiny_sept_1e5_constant)        | 



For more details about the training parameters, see the tranings scripts.