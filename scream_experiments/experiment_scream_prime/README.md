# SCREAM 
## Primordial Experiments

These are the first experiments on the new Norwegian Whisper Model ("Scream") based on the NCC-S corpus serie. The first set of experiments consists of 8 models all trained from the pretrained Whisper models and all following the same learning rate as specified in the [Whisper article](https://cdn.openai.com/papers/whisper.pdf).

All models are trained on a single TPU Pod v4-8 with 128GB RAM. The tiny and small model is trained with a batch of 256 as in the Whisper article. The Small model is trained with a batch size of 144. 

## Experiments
|        | Constant LR | Linear LR |
|--------|----------|--------|
| Tiny   | [Experiment 1](https://huggingface.co/NbAiLab/scream_prime_e1_ncc3nrk_linearlr_tiny)        | [Experiment 2](https://huggingface.co/NbAiLab/scream_prime_e2_ncc3nrk_constantlr_tiny)      |
| Base   | [Experiment 3](https://huggingface.co/NbAiLab/scream_prime_e3_ncc3nrk_linearlr_base), [Experiment 7](https://huggingface.co/NbAiLab/scream_prime_e7_nccs3_constantlr_base), [Experiment 8](https://huggingface.co/NbAiLab/scream_prime_e8_ncc_constantlr_base)  | [Experiment 4](https://huggingface.co/NbAiLab/scream_prime_e4_ncc3nrk_constantlr_base)      |
| Small  | [Experiment 5](https://huggingface.co/NbAiLab/scream_prime_e5_ncc3nrk_linearlr_small)        | [Experiment 6](https://huggingface.co/NbAiLab/scream_prime_e6_ncc3nrk_constantlr_small)      |


* Experiment 1-6 is trained on the [NCC3NRK-corpus](https://huggingface.co/datasets/NbAiLab/NCC_S3_nrk)
* Experiment 7 is trained on the [NCC3-corpus](https://huggingface.co/datasets/NbAiLab/NCC_S3)
* Experiment 8 is trained on the [NCC-corpus](https://huggingface.co/datasets/NbAiLab/NCC_S)


For more details about the training parameters, see the tranings scripts. For more information about the corporas see the dataset cards.