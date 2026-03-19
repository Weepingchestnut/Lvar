# ImageReward

## ImageReward

[ImageReward](https://github.com/THUDM/ImageReward) 是一种用于评估生成图像的人类偏好分数的指标，其通过使用 137K人类排序的图像对 来微调CLIP模型，从而学习人类的偏好。

| Methods   | # Params | Avg Image Reward $\uparrow$ | Avg CLIP Scale $\uparrow$ |
| --------- | -------- | --------------------------- | ------------------------- |
| Infinity  | 2.20B    | 0.9430                      | 0.2694                    |
| + ScaleKV | 2.20B    | 0.9300                      | 0.2698                    |
| FastVAR   | 2.20B    | 0.8869                      | 0.2707                    |

---

## How to eval 如何验证

（推荐）新建一个虚拟环境 `imgrewd` 用于ImageReward验证，PyTorch版本采用 `2.5.1` 或 `2.6.0`。

```python
# (recommend) conda create -n imgrewd python=3.12
# (recommend) pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

pip install image-reward pytorch_lightning
pip install openai==1.34.0
pip install git+https://github.com/openai/CLIP.git ftfy
# pip install git+https://kkgithub.com/openai/CLIP.git ftfy
MAX_JOBS=16 pip install flash-attn==2.7.1.post4 --no-build-isolation
# pip install -U timm diffusers
# pip install diffusers==0.16.0 or diffusers==0.30.0
pip install httpx==0.23.0
# pip install httpx==0.20.0

wget https://github.com/THUDM/ImageReward/blob/main/benchmark/benchmark-prompts.json
```

运行脚本开始测试：
```shell
bash scripts/image_reward_infer4eval.sh
```


# Some Installation Bugs 一些安装Bug

注意：对于较新版本的 `transformers`，`image-reward` 可能会存在 `import` 错误，若出现相关错误，请修改如下代码
```python
# .py file: .../anaconda3/envs/torch271/lib/python3.12/site-packages/ImageReward/models/BLIP/med.py
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
# -->
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
```

如果使用的是 `flash_attn` 的较新版本（如 `2.8.3`），可能会遇到以下错误：
```python
RuntimeError: Failed to import diffusers.models.autoencoders.autoencoder_kl because of the following error (look up to see its traceback):
Requires Flash-Attention version >=2.7.1,<=2.8.0 but got 2.8.3.
```

降低 `Flash Attention` 版本或者创建一个新的虚拟环境都是可行的解决方案。
```bash
MAX_JOBS=16 pip install flash-attn==2.7.1.post4 --no-build-isolation
```
