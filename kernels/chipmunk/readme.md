# Install SparVAR $CS^4A$ Kernels

## 1. Install dependencies

```bash
cd kernels/chipmunk

# Clone third-party libraries 安装第三方库
git clone https://github.com/HazyResearch/ThunderKittens submodules/ThunderKittens
cd submodules/ThunderKittens
# Switch to the specified version 切换到指定版本
git checkout 0c44d2c7262fdd94273a18420a5861bc96a335e3

pip install -r requirements.txt         # in kernels/chipmunk dir
# -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. Install chipmunk
```bash
pip install -e . --no-build-isolation
```

> Note: If the C++ Kernel code has been modified, the above installation command needs to be re-executed to compile the new `.so` file.
<!-- > 注意：若修改了 C++ Kernel 代码，需要重新执行上述安装命令，编译新的`.so`文件 -->

## 3. Kernel Test
```bash
cd Lvar/kernels
python bench_cross_attn.py
```
