
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


### Why Fix Seed?

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
- 支持从 videos/ 统计整个 VBench benchmark 的 overall PSNR / SSIM / LPIPS

```bash
# mutil-GPUs
torchrun --nproc_per_node=8 evaluation/vbench/eval_low_level_metrics.py \
  --baseline-root /path/to/baseline/fps16_5s_enlarge2captain1 \
  --candidate-root /path/to/accelerated/fps16_5s_enlarge2captain1

# single GPU
python3 evaluation/vbench/eval_low_level_metrics.py \
  --baseline-root /path/to/baseline/fps16_5s_enlarge2captain1 \
  --candidate-root /path/to/accelerated/fps16_5s_enlarge2captain1
```

Other useful parameters:

- `--input-layout {auto, vbench, flat}` (default: `auto`): `vbench` expects the VBench structure (videos/ + videos_by_dimension/). `flat` treats both roots as plain directories of videos matched by filename (a root/videos subdir is also accepted); --dimensions is ignored in flat mode. `auto` detects the layout from each root.
<!-- auto 对两个 root 各自探测——能解析出 videos/ + videos_by_dimension/ 结构就走 VBench 模式，否则只要目录里（或其 videos/ 子目录里）有视频文件就走 flat 模式；两边探测结果不一致会直接报错提示需要显式指定. -->
- `--decode-backend {auto, torchcodec, decord, opencv}` (default: `auto`): Video decoding backend. `auto` prefers **[torchcodec](https://github.com/meta-pytorch/torchcodec)**, then decord, then opencv.
  - `--decode-device {auto, cuda, cpu}` (default: `auto`): torchcodec only. `cuda` decodes via NVDEC on the metric GPU; `auto` tries NVDEC and permanently falls back to `cpu` decoding on the first failure.
  - `--decode-threads (int)` (default: `0`): FFmpeg thread count per torchcodec decoder. 0 keeps FFmpeg's default.

- `--prefetch-depth (int)` (default: `2`): How many video pairs to decode ahead in background threads (overlaps decoding with GPU metric computation). `0` disables prefetching.

- `--frame-batch-size (int)` (default: `16`): How many frames to evaluate per GPU batch. The adjustment can be made appropriately based on the GPU memory.

- `--lpips-net-type {alex, vgg, squeeze}` (default: `vgg`): Backbone used by torchmetrics LPIPS.

- `--preferred-source {auto,npy,png,video}` (default: `auto`): Metric input source. 'auto' prefers npy, then PNG frame directories, then encoded videos.
<!-- 默认`auto`，评估时会优先读取`videos/*.npy`，其次读`frames_by_dimension/<dim>/<name>`或`videos/<physical_stem>/`下的`.png`帧目录，最后才回退`.mp4`解码 -->
- `--include-first-frame {0, 1}` (default: `1`): Whether to include frame 0 in metric computation. Set to 0 to evaluate only later video frames.

- `--collect-per-video (store_true)`: Gather per-video metric values to rank 0; adds a per_video_metrics section and std fields to the report json.

#### Use TorchCodec

TorchCodec is a Python library for decoding video and audio data into PyTorch tensors, on CPU and CUDA GPU.

```bash
# 1. Install FFmpeg
conda install "ffmpeg"
# or
conda install "ffmpeg" -c conda-forge

# Install PyTorch and TorchCodec:
# torch >= 2.11
pip install torchcodec
```

Check it after installation
```bash
# 1) 共享库在不在（Linux or conda env）
python -c "import ctypes.util; print(ctypes.util.find_library('avcodec'))"
# 2) TorchCodec 能否真正解码
# libopenvino 找 libstdc++ 时先命中 env 里更新版本的 libstdc++
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python -c "from torchcodec.decoders import VideoDecoder; d=VideoDecoder('xxx.mp4'); print(d[:].shape)"
# 3) 确认 imageio 没受影响（仍指向它自己捆绑的二进制）
python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
```

---


### Bugs

```
ModuleNotFoundError: No module named 'pkg_resources'
```

```bash
python -m pip install "setuptools<82.0.0" --force-reinstall
python -c "import pkg_resources"
```

