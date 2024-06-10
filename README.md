# Llama 3

## Installation

This task was run on CUDA 12.1, Pytorch 1.12.1, and a nightly build of torchtune. To run this task, create an environment with the following command:
```bash
conda env create -f llama3_env.yml
```

Make sure you download the weights for Meta-Llama-3-8B-Instruct. this must be in a directory called Meta-Llama-3-8B-Instruct/.

## Finetuning

To finetune using the default Llama3 weights, run:
```bash
tune run lora_finetune_single_device --config daic_8B_qlora_style.yaml
```

## Inference

If you want to generate text using your own finetuned model from the previous step, replace the element in checkpoint_files to your desired file.

To run the inference task on each conversation, use DAIC_Llama_Chain-finetune.ipynb. To run it on the entire corpus for style transfer, use DAIC_Llama_Chain-style.ipynb.