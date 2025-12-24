


## Fix Seed

为什么要固定seed？
- 完整推理一遍 Vbench 耗时久，因此在快速迭代算法时通常先关注一到两个维度的 Vbench 分数来进行简单评估
- 评估加速后模型相比于Baseline模型的生成质量时，通常测试一些 low-Level 指标（如PSNR），因此需要固定 seed 以尽可能确保生成视频的一致性

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


# Network Problem

