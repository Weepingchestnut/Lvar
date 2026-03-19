
# Evaluation on the GenEval

Evaluation on the GenEval benchmark. * result is with prompt rewriting.

| Methods   | # Params | Single Obj. | Two Obj. | Count. | Colors | Position | Color Attri. | Overall $\uparrow$ |
| --------- | -------- | ----------- | -------- | ------ | ------ | -------- | ------------ | ------------------ |
| Infinity  | 2.20B    | 0.9938      | 0.7879   | 0.6688 | 0.8271 | 0.2525   | 0.6100       | 0.6900             |
| Infinity* | 2.20B    | 1.0000      | 0.8586   | 0.6594 | 0.8537 | 0.4450   | 0.5675       | 0.7307             |
| +ScaleKV* | 2.20B    | 1.0000      | 0.8510   | 0.6750 | 0.8484 | 0.4375   | 0.5400       | 0.7253             |
| FastVAR*  | 2.20B    | 1.0000      | 0.8283   | 0.6406 | 0.8271 | 0.4350   | 0.4950       | 0.7043             |



Infinity: 32903MiB / 40536MiB

ScaleKV: 22705MiB / 40536MiB


# How to eval

```python
# Download detection model weight
mkdir pretrained_models/mask2former
wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth -O "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"


pip install -U openmim
mim install mmengine mmcv-full==1.7.2
pip install mmdet==2.28.2 pytorch_lightning clip_benchmark open-clip-torch==2.20.0
# pip install -U diffusers
apt install libgl1
# pip install openai
# pip install httpx==0.20.0
```


# Some Installation Bugs

## Bug1: module 'pkgutil' has no attribute 'ImpImporter'

如果在执行 `mim install mmengine mmcv-full==1.7.2` 时报错如下，请卸载并重新安装 `setuptools`
```python
Traceback (most recent call last):
  File ".../anaconda3/envs/torch260/bin/mim", line 5, in <module>
    from mim.cli import cli
  File ".../anaconda3/envs/torch260/lib/python3.12/site-packages/mim/__init__.py", line 10, in <module>
    import setuptools  # noqa: F401
    ^^^^^^^^^^^^^^^^^
  File ".../anaconda3/envs/torch260/lib/python3.12/site-packages/setuptools/__init__.py", line 16, in <module>
    import setuptools.version
  File ".../anaconda3/envs/torch260/lib/python3.12/site-packages/setuptools/version.py", line 1, in <module>
    import pkg_resources
  File ".../anaconda3/envs/torch260/lib/python3.12/site-packages/pkg_resources/__init__.py", line 2172, in <module>
    register_finder(pkgutil.ImpImporter, find_on_path)
                    ^^^^^^^^^^^^^^^^^^^
AttributeError: module 'pkgutil' has no attribute 'ImpImporter'. Did you mean: 'zipimporter'?

# reinstall setuptools
pip uninstall -y setuptools
pip install setuptools
```

## Bug2: No module named 'mmcv._ext'

对于H系列GPU，上述安装过程可能在较新版本的Python(e.g. 3.12.x)和PyTorch(e.g. 2.6.0)环境中存在问题，建议使用 [evaluation/dpg_bench](../dpg_bench/readme.md) 所创建的版本较低的`modelscope`虚拟环境，重新安装即可。
<!-- ```python
pip uninstall mmcv mmcv-full

MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST="9.0" python setup.py develop
``` -->

如果遇到bug如下
```bash
ModuleNotFoundError: No module named 'mmcv._ext'
```
表示`mmcv-full 1.7.2`中的 CUDA Kernel 没有编译成功，可以参考如下链接从源码进行编译

- https://github.com/ULMEvalKit/ULMEvalKit/blob/main/docs/envs/benchmarks/GenEval.md
- https://mmcv.readthedocs.io/en/latest/get_started/build.html

```bash
# 没有 ninja 可能编译不成功
pip install ninja psutil

git clone https://github.com/open-mmlab/mmcv.git
cd mmcv; git checkout 1.x

# 使用 pip 安装，可能并不会编译 CUDA Kernel
# MMCV_WITH_OPS=1 MMCV_CUDA_ARGS="-arch=sm_90" pip install -v -e .

# 推荐使用 setup.py 直接安装：
MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST="9.0" python setup.py develop
# or
# MMCV_WITH_OPS=1 MMCV_CUDA_ARGS="-arch=sm_90" python setup.py build_ext --inplace
# MMCV_WITH_OPS=1 MMCV_CUDA_ARGS="-arch=sm_90" python setup.py install

# 验证 CUDA Kernel 是否编译成功
python .dev_scripts/check_installation.py
```

输出如下表示`mmcv-full`真正安装成功
```bash
Start checking the installation of mmcv-full ...
CPU ops were compiled successfully.
CUDA ops were compiled successfully.
mmcv-full has been installed successfully.

Environment information:
------------------------------------------------------------
...

TorchVision: 0.20.1+cu124
OpenCV: 4.13.0
MMCV: 1.7.2
MMCV Compiler: GCC 11.4
MMCV CUDA Compiler: 12.8
------------------------------------------------------------
```

若在验证时报错如下：
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

需安装安装缺失的 libGL 库
```bash
apt install libgl1
```






