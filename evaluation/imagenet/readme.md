# c2i 推理

核心推理代码分别涉及如下两个文件

- `.../Lvar/evaluation/imagenet/class_cond_sample_ddp.py`，参考[DiT](https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py)中的推理代码改写而来，类别生成顺序随机；
- `.../Lvar/evaluation/imagenet/class_cond_sample_ddp_seq.py`，相比于上述文件，固定类别生成顺序，即从`class 0`到`class 999`，每个类别生成50张图像。



执行脚本`scripts/c2i_sample_ddp.sh`，可改写脚本中的推理代码文件为上述一种（推荐固定类别生成顺序，生成质量指标更高）

```bash
bash scripts/c2i_sample_ddp.sh
```

- `--model-depth`：选择模型大小，可设置为[16, 20, 24, 30]
- `--per-proc-batch-size`：每卡的batch size，可根据显存调整，默认为32
  - 对于固定类别生成顺序，会根据显卡数量自定义，无需设定



跑通后会将生成的图像数据存储到路径`.../Lvar/work_dir`下，同时生成对应的`.npz`文件，用于后续测试指标。



# c2i 指标计算

## OpenAI TensorFlow 脚本

参考VAR官方Repo，主要还是使用openai提供的[验证工具](https://github.com/openai/guided-diffusion/tree/main/evaluations)（基于TensorFlow框架）。

具体运行文件位于`.../Lvar/evaluation/imagenet/openai_evaluator.py`



需要先下载ImageNet的参考`.npz`数据集，

- ImageNet 256x256: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz)（常用）
- ImageNet 512x512: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz)

将下载好的参考`.npz`数据集放置到路径`.../Lvar/evaluation/imagenet/VIRTUAL_imagenet256_labeled.npz`。



直接运行如下命令即可：

```bash
python openai_evaluator.py VIRTUAL_imagenet256_labeled.npz <path_to_our_generation/xxx.npz>
```



## PyTorch 脚本

新增基于PyTorch的验证脚本，参考[MAR](https://github.com/LTH14/mar/blob/main/engine_mar.py)中的实现。

首先需安装经过自定义的`torch-fidelty`，安装命令如下：

```bash
# pip install -e git+https://github.com/LTH14/torch-fidelity.git@master#egg=torch-fidelity

# --src: set git download path
pip install -e git+https://github.com/LTH14/torch-fidelity.git@master#egg=torch-fidelity --src evaluation/imagenet
```

此外，需下载ImageNet的统计参考`.npz`数据集，

- ImageNet 256x256: [adm_in256_stats.npz](https://github.com/LTH14/mar/blob/main/fid_stats/adm_in256_stats.npz)

将下载好的统计参考`.npz`数据集放置到路径`.../Lvar/evaluation/imagenet/adm_in256_stats.npz`。



c2i推理脚本会默认进行FID和IS的计算，也可以通过代码`torch_evaluator.py`进行指标计算；

- `img_save_folder`：存储生成图像的路径。







