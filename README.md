# Lvar

## 📖 Introduction
Learn visual autoregressive


## 🔥 Supported Models

### Class-condition Generation Models
- VAR (NeurIPS'2024 **Best Paper Award**)

### Text-to-Image Generation Models
- Infinity (CVPR'2025 Oral)
- FastVAR (ICCV'2025)
- ScaleKV
- SkipVAR


## ⚙️Installation

### Model Zoo

#### Infinity

**Download [flan-t5-xl](https://huggingface.co/google/flan-t5-xl).**

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
```
These three lines will download flan-t5-xl to your ~/.cache/huggingface directory.

or 

```python
mkdir pretrained_models/infinity/flan-t5-xl
cd pretrained_models/infinity/flan-t5-xl

huggingface-cli download google/flan-t5-xl --local-dir ./
```
---

**Download <a href='https://huggingface.co/FoundationVision/infinity'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20weights-FoundationVision/Infinity-yellow'></a>**

If you want to download all the weights at once, please refer to
```python
mkdir pretrained_models/infinity/Infinity
cd pretrained_models/infinity/Infinity

huggingface-cli download FoundationVision/Infinity --local-dir ./
```

Download the commonly used weights, please refer to
```python
mkdir pretrained_models/infinity/Infinity
cd pretrained_models/infinity/Infinity

huggingface-cli download FoundationVision/Infinity --include="infinity_vae_d32reg.pth" --local-dir ./
huggingface-cli download FoundationVision/Infinity --include="infinity_2b_reg.pth" --local-dir ./
```

For more models, please refer to the `readme` of each model in the `pretrained_models/` directory.

- [LlamaGen](./pretrained_models/llamagen/readme.md)
- 

## 🍭 Evaluation

We provide code and corresponding scripts for various benchmarks.
Please refer to the following `readme` for different benchmarks.

- [evaluation/gen_eval](./evaluation/gen_eval/)
- [evaluation/dpg_bench](./evaluation/dpg_bench/readme.md)
- [evaluation/hpsv2](./evaluation/hpsv2/readme.md)
- [evaluation/image_reward](./evaluation/image_reward/readme.md)


## Reference
The **Lvar** codebase is adapted from [VAR](https://github.com/FoundationVision/VAR) and [Infinity](https://github.com/FoundationVision/Infinity). Special thanks to their excellent works! 