# Human Preference Score v2.1

## HPS v2.1 benchmark

[HPSv2.1](https://github.com/tgxs002/HPSv2) is a metric for evaluating the human preference score of generated images. It learns human preference through fine-tuning CLIP model with 798K human ranked image pairs. The human ranked image pairs are from human experts.

| Methods   | # Params | Painting | Photo | Animation | Concept-art | Average $\uparrow$ |
| --------- | -------- | -------- | ----- | --------- | ----------- | ------------------ |
| Infinity  | 2.20B    | 30.45    | 29.33 | 31.66     | 30.41       | 30.46              |
| + ScaleKV | 2.20B    | 30.32    | 29.26 | 31.54     | 30.21       | 30.33              |
| FastVAR   | 2.20B    | 28.47    | 27.61 | 29.77     | 25.58       | 28.61              |

---

<!-- | Methods   | # Params | GPU Mem  |
| --------- | -------- | -------- |
| I-------y | 2------B | undefined------ |
| + ScaleKV | 2.20B    |          |
| FastVAR   | 2.20B    | 23269MiB |

--- -->

## How to eval

```python
pip install hpsv2
pip install -U diffusers
sudo apt install python3-tk
wget https://dl.fbaipublicfiles.com/mmf/clip/bpe_simple_vocab_16e6.txt.gz
# mv bpe_simple_vocab_16e6.txt.gz /home/tiger/.local/lib/python3.9/site-packages/hpsv2/src/open_clip
mv bpe_simple_vocab_16e6.txt.gz ~/anaconda3/envs/torch260/lib/python3.12/site-packages/hpsv2/src/open_clip
```

由于需要从Huggingface中下载文件，因此需先设置好 `tools/conf.py` 中的huggingface token。

运行脚本
```shell
bash scripts/hpsv2_infer4eval.sh
```

- 测试结果可能与论文中不同，其他人也存在类似的问题，详情请见 https://github.com/FoundationVision/Infinity/issues/42
