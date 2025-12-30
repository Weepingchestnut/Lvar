


Install dependencies

```bash
cd kernels/chipmunk
git clone https://github.com/HazyResearch/ThunderKittens submodules/ThunderKittens
cd submodules/ThunderKittens
git checkout 0c44d2c7262fdd94273a18420a5861bc96a335e3
```

install chipmunk
```bash
pip install -e . --no-build-isolation
```

注意：若修改了 C++ Kernel 代码，需要重新执行上述安装命令，编译新的`.so`文件


