# Llama 3

We are unlocking the power of large language models. Our latest version of Llama is now accessible to individuals, creators, researchers, and businesses of all sizes so that they can experiment, innovate, and scale their ideas responsibly.

This release includes model weights and starting code for pre-trained and instruction-tuned Llama 3 language models — including sizes of 8B to 70B parameters.

This repository is a minimal example of loading Llama 3 models and running inference. For more detailed examples, see [llama-recipes](https://github.com/facebookresearch/llama-recipes/).


## Installation

This task was run on CUDA 12.1, Pytorch 1.12.1, and a nightly build of torchtune. To run this task, create an environment with the following command:
```bash
conda env create -f llama3_env.yml
```

## Finetuning

If you want to use the provided finetuned model, skip this step.

To finetune using the default Llama3 weights, run:
```bash
tune run lora_finetune_single_device --config daic_8B_qlora_style.yaml
```

## Inference

If you want to generate text using your own finetuned model from the previous step, replace the element in checkpoint_files to your desired file.

To run the inference task on each conversation, use DAIC_Llama_Chain-finetune.ipynb. To run it on the entire corpus for style transfer, use DAIC_Llama_Chain-style.ipynb.