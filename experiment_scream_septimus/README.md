# SCREAM 
## Septimus Experiments

These experiments constitute the seventh series of trials on the new Norwegian Whisper Model ("Scream"), based on the NCC-S corpus series. The objective of these experiments is to explore the scalability of the model on TPU pods, while also examining the impact of adding the NPSC corpus to the training for the first time.

The Tiny models are trained on a TPU Pod v4-32 with a batch size of 512, while the Large models are trained on a TPU Pod v4-64 with a batch size of 160. The Tiny models aim to compare the use of the nrk_v5 corpus to using all_v5 corpus. The all_v5 corpus includes the first version of NPSC+ corpus and the NST corpus in addition the the NRK subtitles.

The Large model experiments aim to investigate the impact of scaling and determine the optimal learning rate, using 2e6 and 8e6. All models use linear decay.


## Experiments
|        | NRK | All | 
|--------|----------|--------|
| Tiny   | [Experiment 1](https://huggingface.co/NbAiLab/scream_tiny_sept_nrk)        | [Experiment 2](https://huggingface.co/NbAiLab/scream_tiny_sept_all)      |


|        | 2e6 | 8e6 |
|--------|----------|--------|
| Large  | [Experiment 3](https://huggingface.co/NbAiLab/scream_large_sept_2e6)        | [Experiment 4](https://huggingface.co/NbAiLab/scream_large_sept_8e6)      |


In addition we are running a single experiment on a v4-32 where the batch size is scaled to 64x4x4=1024. The learning rate on this experiment is scaled using the liear scaling rule to 2.828 (2e-5 * sqrt(512/256)). We have adusted down the number of training steps and doubled the log reporting so that it should train and report after the same number of experiments as Experiment 2.

|        | Experiment |
|--------|----------|
| Linear 2.828-e5 1024bs| [Experiment 5](https://huggingface.co/NbAiLab/scream_tiny_sept_all_1024bs)        | 


There is a defined set of smaller experiments on v4-8 for hyperparameter search (learning rate). This sett is not started yet. 


For more details about the training parameters, see the tranings scripts.