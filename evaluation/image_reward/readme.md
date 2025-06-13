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

```python
pip install image-reward pytorch_lightning
pip install -U timm diffusers
pip install openai==1.34.0
pip install httpx==0.20.0
pip install diffusers==0.16.0
pip install git+https://github.com/openai/CLIP.git ftfy

wget https://github.com/THUDM/ImageReward/blob/main/benchmark/benchmark-prompts.json
```

运行脚本
```shell
bash scripts/image_reward_infer4eval.sh
```
