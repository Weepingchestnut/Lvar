import os

import torch

import matplotlib.pyplot as plt


class VisualAttnMap:
    def __init__(
            self,
            cur_scale: int = 1,
            cur_block: int = 1,
            save_dir: str = "work_dir/visual_attn_map"):

        self.cur_scale = cur_scale
        self.cur_block = cur_block

        self.save_dir = save_dir
    
    def set_cur_scale(self, new_scale):
        self.cur_scale = new_scale
    
    def set_cur_block(self, new_blcok):
        self.cur_block = new_blcok

    def visual_attn_map(self, attn_matrix: torch.Tensor):
        # sample_idx = 0
        # head_idx = 0

        attn_map_save_dir = os.path.join(
            self.save_dir,
            f"scale-{self.cur_scale}_block-{self.cur_block}")
        
        if not os.path.exists(attn_map_save_dir):
            os.makedirs(attn_map_save_dir)

        for sample_idx in range(attn_matrix.shape[0]):  # 遍历所有样本
            for head_idx in range(attn_matrix.shape[1]):  # 遍历所有注意力头

                attn_head = attn_matrix[sample_idx, head_idx, :, :].cpu().detach().numpy()

                # 可视化并保存
                plt.figure(figsize=(8, 8))
                plt.matshow(attn_head, cmap='viridis')
                plt.colorbar()
                plt.title(f"Attention Map - Sample {sample_idx}, Head {head_idx}")
                plt.xlabel("All Scale Tokens")
                plt.ylabel("Current Scale Tokens")

                # 保存为 PNG 文件
                plt.savefig(os.path.join(
                    attn_map_save_dir, f"attention_map_sample-{sample_idx}_head-{head_idx}.png"),
                    dpi=300,
                    bbox_inches='tight')
                plt.close()
