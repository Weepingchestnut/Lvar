# Install SparVAR $CS^4A$ Kernels

## 1. Install dependencies

```bash
cd kernels/chipmunk

# Clone third-party libraries
git clone https://github.com/HazyResearch/ThunderKittens submodules/ThunderKittens
cd submodules/ThunderKittens
# Switch to the specified version
git checkout 0c44d2c7262fdd94273a18420a5861bc96a335e3

pip install -r requirements.txt         # in kernels/chipmunk dir
# -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. Install chipmunk

### 2.1 Conda env

If the system-level CUDA version does not match the requirement of 12.8, you may try installing CUDA 12.8 in a conda virtual environment and use the CUDA within the virtual environment to compile the Kernel.

```bash
# install cuda in virtual env
conda install -c nvidia cuda-toolkit=12.8



# Make the CUDA config take effect in the current session
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# check current CUDA version
nvcc -V
# --------------- Expected output ---------------
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2025 NVIDIA Corporation
# Built on Fri_Feb_21_20:23:50_PST_2025
# Cuda compilation tools, release 12.8, V12.8.93
# Build cuda_12.8.r12.8/compiler.35583870_0
# -----------------------------------------------
which nvcc
# --------------- Expected output ---------------
# like: /home/xxx/.conda/envs/{vir_env_name}/bin/nvcc
# -----------------------------------------------
```

If you have bug like this
```bash
RuntimeError: The current installed version of /home/xxx/.conda/envs/torch2110/bin/x86_64-conda-linux-gnu-c++ (14.3.0) is greater than the maximum required version by CUDA 12.8. Please make sure to use an adequate version of /home/xxx/.conda/envs/torch2110/bin/x86_64-conda-linux-gnu-c++ (>=6.0.0, <14.0).
```

You need to downgrade gcc/g++ in this environment to version 13

```bash
conda install -c conda-forge "gxx_linux-64=13.*" "gcc_linux-64=13.*"

# Reactivate the environment after installation, and let the conda activation script redirect `CC`/`CXX` to the new version 13 wrapper again:
x86_64-conda-linux-gnu-c++ --version
# --------------- Expected output ---------------
# x86_64-conda-linux-gnu-c++ (conda-forge gcc 13.4.0-7) 13.4.0
# Copyright (C) 2023 Free Software Foundation, Inc.
# This is free software; see the source for copying conditions.  There is NO
# warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------
echo $CXX
# --------------- Expected output ---------------
# /home/xxx/.conda/envs/torch2110/bin/x86_64-conda-linux-gnu-c++
# -----------------------------------------------
```

### 2.2 Compile Kernel

```bash
pip install -e . --no-build-isolation
```

> Note: If the C++ Kernel code has been modified, the above installation command needs to be re-executed to compile the new `.so` file.
<!-- > 注意：若修改了 C++ Kernel 代码，需要重新执行上述安装命令，编译新的`.so`文件 -->

### 2.3 Kernel Test
```bash
cd Lvar/kernels
python bench_cross_attn.py
```

## Other High-Performance Kernels Installation



```bash

```
