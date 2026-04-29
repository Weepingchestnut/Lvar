# ImageReward

## ImageReward

[ImageReward](https://github.com/THUDM/ImageReward) is a metric for evaluating the human preference score of generated images. It learns human preference through fine-tuning CLIP model with 137K human ranked image pairs.

| Methods   | # Params | Avg Image Reward $\uparrow$ | Avg CLIP Scale $\uparrow$ |
| --------- | -------- | --------------------------- | ------------------------- |
| Infinity  | 2.20B    | 0.9430                      | 0.2694                    |
| + ScaleKV | 2.20B    | 0.9300                      | 0.2698                    |
| FastVAR   | 2.20B    | 0.8869                      | 0.2707                    |

---

## How to eval

(Recommended) Use the same virtual environment `modelscope` as GenEval and DPG-Bench.

```python
# (recommend) conda activate modelscope

pip install image-reward pytorch_lightning
pip install openai==1.34.0
pip install git+https://github.com/openai/CLIP.git ftfy
# pip install git+https://kkgithub.com/openai/CLIP.git ftfy
pip install diffusers==0.30.0 huggingface_hub==0.23.2
MAX_JOBS=8 pip install flash-attn==2.7.1.post4 --no-build-isolation
# pip install httpx==0.20.0

wget https://github.com/THUDM/ImageReward/blob/main/benchmark/benchmark-prompts.json
```

Run bash:
```shell
bash scripts/image_reward_infer4eval.sh
```


# Some Installation Bugs

Note: For newer versions of `transformers`, `image-reward` may have `import` errors. If such errors occur, please modify the following code.
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

If you have the latest version of `flash_attn` (e.g. `2.8.3`), you may encounter the following error:
```python
RuntimeError: Failed to import diffusers.models.autoencoders.autoencoder_kl because of the following error (look up to see its traceback):
Requires Flash-Attention version >=2.7.1,<=2.8.0 but got 2.8.3.
```
Lowering the Flash Attention version or creating a new virtual environment are both viable solutions.
```bash
MAX_JOBS=16 pip install flash-attn==2.7.1.post4 --no-build-isolation
```
