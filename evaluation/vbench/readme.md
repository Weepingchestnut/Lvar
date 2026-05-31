
## VBench 环境配置

- https://github.com/Vchitect/VBench

```bash
conda create -n vbench python=3.10
# Please install PyTorch with 11.6<=CUDA<=12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# install detectron2 from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
python -m pip install -e . --no-build-isolation

# install vbench
git clone https://github.com/Vchitect/VBench.git
pip install -e . --no-build-isolation
```


## Fix Seed

为什么要固定seed？
- 完整推理一遍 Vbench 耗时久，因此在快速迭代算法时通常先关注一到两个维度的 Vbench 分数来进行简单评估
- 评估加速后模型相比于Baseline模型的生成质量时，通常测试一些 low-level 指标（如PSNR），因此需要固定 seed 以尽可能确保生成视频的一致性

为防止每次验证时 seed 不统一的问题，我们先对 Vbench 共 946 个 prompts 随机生成对应的 seed，并写入到新的 `VBench_rewrited_prompt_fixed_seed.json` 文件中，
这样保证了每次评估时每个prompt都有一个不变的 seed，使得复现结果、测试low-Level指标等更加方便。

执行如下命令来创建新的 `VBench_rewrited_prompt_fixed_seed.json`
```bash
python evaluation/vbench/add_fixed_seeds.py
```

其次为保证同一个 prompt 的不同采样生成的视频不同，我们在推理脚本中基于每个prompt固定的seed进行一定程度的偏移
```python
seed = prompt_seed + n_samples * sample_idx
```
- `prompt_seed` 是之前已经写入到 `VBench_rewrited_prompt_fixed_seed.json` 中的 seed
- `n_samples * sample_idx` 是基于当前采样数量进行简单的偏移


## How to use

### VBench Evaluation

输出目录结构：
```
work_dir/evaluation/vbench/infinitystar_720p/fps16_5s_enlarge2captain1/
├── videos/
├── videos_by_dimension/
│   └── subject_consistency/
│       ├── a bear catching a salmon in its powerful jaws-0.mp4
│       ├── ...
│
└── frames_by_dimension/
    └── subject_consistency/
        ├── a bear catching a salmon in its powerful jaws-0/
        │   ├── 000000.png
        │   ├── 000001.png
        │   └── ...
```


### Low-level metrics

`evaluation/vbench/eval_low_level_metrics.py` 针对已生成的 VBench 视频进行 low-level 指标的评估。

```bash
# mutil-GPUs
torchrun --nproc_per_node=8 evaluation/vbench/eval_low_level_metrics.py \
  --baseline-root /path/to/baseline/fps16_5s_enlarge2captain1 \
  --candidate-root /path/to/accelerated/fps16_5s_enlarge2captain1
  --preferred-source png

# single GPU
python3 evaluation/vbench/eval_low_level_metrics.py \
  --baseline-root /path/to/baseline/fps16_5s_enlarge2captain1 \
  --candidate-root /path/to/accelerated/fps16_5s_enlarge2captain1
  --preferred-source png
```

支持：
- 从 videos/ 统计整个 VBench benchmark 的 overall PSNR / SSIM / LPIPS
- `--preferred-source {auto,npy,png,video}`，默认`auto`，评估时会优先读取`videos/*.npy`，其次读`frames_by_dimension/<dim>/<name>`或`videos/<physical_stem>/`下的`.png`帧目录，最后才回退`.mp4`解码


### Bugs

```
ModuleNotFoundError: No module named 'pkg_resources'
```

```bash
python -m pip install "setuptools<82.0.0" --force-reinstall
python -c "import pkg_resources"
```

