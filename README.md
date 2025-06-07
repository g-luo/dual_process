# 🐢 Dual-Process Image Generation
**Grace Luo, Jonathan Granskog, Aleksander Holynski, Trevor Darrell**

This repository contains the PyTorch implementation of Dual-Process Image Generation.

[[`Project Page`](https://dual-process.github.io)][[`arXiv`](https://arxiv.org/abs/2506.01955)]

## Setup
This code was tested with Python 3.10. To install the necessary packages, please run:
```
conda env create -f environment.yaml
conda activate dig
```

## Gradio Demo
You can also run our method as a gradio app using the following command.
If you set the environment variable `OPENAI_API_KEY`, you can use the extra prompt and question expansion features we have provided. You can also leave this variable unset, which will disable these features.
```
conda activate dig
export OPENAI_API_KEY=<your_openai_api_key>
python3 launch_gradio.py
```

## Generation Scripts
You can also mass optimize LoRAs and visualize their samples using the following script.
```
conda activate dig
./run_dual_process.sh
```

## Compute Requirements
This code was tested on a single 80GB Nvidia A100 GPU. 
However, depending on the configuration of Image Generator and VLM,
you can run this codebase on as little as two 24GB Nvidia RTX 4090 GPUs.
You can find the VRAM requirements for each supported model in the table below (as measured with `torch.cuda.max_memory_reserved`).
We also indicate whether each model is officially supported (experimental models have not been extensively tested).

Model                                              | Model Size | VRAM (GB) | Officially Supported
---------------------------------------------------|------------|-----------|---------
**Image Generator**                               |            |           |         
[Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) | 860M | 6.6 | ❌
[Sana](https://huggingface.co/Efficient-Large-Model/Sana_1600M_512px_diffusers) | 1.6B | 15.0 | ❌
[Flux Schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)        | 12B  | 44.3 | ✅
[Flux Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)                | 12B  | 44.3 | ✅
**VLM**                                            |            |           |         
[Gemma3](https://huggingface.co/google/gemma-3-4b-it)                          | 4B   | 12.1 | ❌
[LLaVA v1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf)                  | 7B   | 16.7 | ❌
[Idefics2](https://huggingface.co/HuggingFaceM4/idefics2-8b)                   | 8B   | 17.3 | ✅
[Qwen2.5 VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)               | 7B   | 19.6 | ✅
[Pixtral](https://huggingface.co/mistral-community/pixtral-12b)                | 12B  | 34.7 | ❌

## Citing
```
@article{luo2025dualprocess,
  title={Dual-Process Image Generation},
  author={Grace Luo and Jonathan Granskog and Aleksander Holynski and Trevor Darrell},
  journal={arXiv preprint arXiv:2506.01955},
  year={2025}
}
```
