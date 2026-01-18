# Lvar

## 📖 Introduction
Learn visual autoregressive


## 🔥 Supported Models

### Class-condition Generation Models
- VAR (NeurIPS'2024 **Best Paper Award**)

### Text-to-Image Generation Models

Baseline Models
- Infinity (CVPR'2025 Oral)

Acceleration Method
- FastVAR (ICCV'2025)
- SparseVAR (ICCV'2025)
- ScaleKV (NeurIPS'2025)
- SkipVAR


## ⚙️ Installation

### Basic Env.
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

### Kernel Compile
Our SparVAR
```bash

```

HART
```bash
cd models/hart/kernels
bash install.sh
```

---

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