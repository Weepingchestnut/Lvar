# Lvar

## 📖 Introduction

**Lvar** is a learning- and research-oriented codebase for **Visual AutoRegressive** (**VAR**) generation architectures.
**`L`** carries a dual meaning: **Learn**, for making visual autoregressive models easier to study and understand from the ground up, and **Light/Lighting**, for integrating lightweight design, efficient inference, and system-level acceleration techniques.
<!-- Built around a unified, modular, and extensible framework, Lvar aims to provide implementations of model architectures, training recipes, inference optimizations, caching strategies, and infrastructure components for advancing next-generation visual generation systems. -->


## 🔥 Supported Models

### Class-condition Generation Models
- [VAR](https://arxiv.org/abs/2404.02905) (NeurIPS'2024 **Best Paper Award**)

Acceleration Methods

---

### Text-to-Image Generation Models

Baseline Models
- [Infinity](https://arxiv.org/abs/2412.04431) (CVPR'2025 Oral)
- [HART](https://arxiv.org/abs/2410.10812) (ICLR'2025)

Acceleration Methods
- FastVAR (ICCV'2025)
- SparseVAR (ICCV'2025)
- ScaleKV (NeurIPS'2025)
- SkipVAR (arXiv'2025)
<!-- - SparVAR (CVPR'2026) -->

### Video Generation Models

- [InfinityStar](https://arxiv.org/abs/2511.04675) (NeurIPS'2025 Oral)

### Other Visual AutoRegressive Method

- BitDance (arXiv:)
- GRN (arXiv:)


## ⚙️ Installation

### 1. Basic Env.
- Some customized Kernels are written for **Hopper** GPUs, and depend on optimizations specific to CUDA Toolkit version ≥ 12.8 (recommend `12.8.1`!).
- For PyTorch, the recommended version is `2.7.1` or later.

```bash
conda create -n torch271 python=3.12

# for CUDA 12.8
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
# flash-attention
MAX_JOBS=16 pip install flash-attn --no-build-isolation

cd Lvar
pip install -r requirements.txt

# Since dev, the Python path should be set manually
vim ~/.bashrc
export PYTHONPATH=$PYTHONPATH:{your-path}/Lvar
```
<!-- export PYTHONPATH=$PYTHONPATH:/share/project/wangning/158/zekun/wkspace/Lvar -->

### 2. Kernel Compile

#### 2.1 SparVAR

The installation of SparVAR Cross-Scale Self-Similar Sparse Attention ($CS^4A$) is rather complicated; please refer to [Install SparVAR $CS^4A$ Kernels](./kernels/chipmunk/readme.md).
<!-- SparVAR的跨尺度自相似稀疏注意力的安装比较复杂，请参考A [Install SparVAR $CS^4A$ Kernels](./kernels/chipmunk/readme.md) -->

#### 2.2 HART
```bash
cd models/hart/kernels
bash install.sh
```

---

## Model Zoo

### Infinity weights

**Download [flan-t5-xl](https://huggingface.co/google/flan-t5-xl).**

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
```
These three lines will download flan-t5-xl to your ~/.cache/huggingface directory.

or 

```python
cd pretrained_models/infinity

bash hf_down.sh
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

For more models, please refer to the `hf_down.sh` of each model in the `pretrained_models/` directory.

e.g.
<!-- - [LlamaGen](./pretrained_models/llamagen/readme.md) -->
- [Infinity](./pretrained_models/infinity/hf_down.sh)

## 🍭 Evaluation

We provide code and corresponding scripts for various benchmarks.
Please refer to the following `readme` for different benchmarks.

- [evaluation/gen_eval](./evaluation/gen_eval/)
- [evaluation/dpg_bench](./evaluation/dpg_bench/readme.md)
- [evaluation/hpsv2](./evaluation/hpsv2/readme.md)
- [evaluation/image_reward](./evaluation/image_reward/readme.md)


## Reference
The **Lvar** codebase is adapted from [VAR](https://github.com/FoundationVision/VAR) and [Infinity](https://github.com/FoundationVision/Infinity). Special thanks to their excellent works! 