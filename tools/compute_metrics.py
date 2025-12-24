import argparse
import os

import numpy as np
import torch
# from cleanfid import fid
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import (LearnedPerceptualImagePatchSimilarity,
                                PeakSignalNoiseRatio,
                                StructuralSimilarityIndexMeasure)
from torchvision.transforms import Resize
from tqdm import tqdm


def read_image(path: str):
    """
    input: path
    output: tensor (C, H, W)
    """
    img = np.asarray(Image.open(path)).copy()
    if len(img.shape) == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img


# class MultiImageDataset(Dataset):
#     def __init__(self, root0, root1, is_gt=False):
#         super().__init__()
#         self.root0 = root0
#         self.root1 = root1
#         file_names0 = os.listdir(root0)
#         file_names1 = os.listdir(root1)

#         self.image_names0 = sorted([name for name in file_names0 if name.endswith(".png") or name.endswith(".jpg")])
#         self.image_names1 = sorted([name for name in file_names1 if name.endswith(".png") or name.endswith(".jpg")])
#         self.is_gt = is_gt
#         assert len(self.image_names0) == len(self.image_names1)

#     def __len__(self):
#         return len(self.image_names0)

#     def __getitem__(self, idx):
#         img0 = read_image(os.path.join(self.root0, self.image_names0[idx]))
#         if self.is_gt:
#             # resize to 1024 x 1024
#             img0 = Resize((1024, 1024))(img0)
#         img1 = read_image(os.path.join(self.root1, self.image_names1[idx]))

#         batch_list = [img0, img1]
#         return batch_list


# GenEval
class MultiImageDataset(Dataset):
    def __init__(self, root_preds, root_gts, is_gt=False):
        super().__init__()
        self.root_preds = root_preds
        self.root_gts = root_gts
        self.is_gt = is_gt

        self.image_preds = []
        self.image_gts = []

        for dirpath, _, filenames in os.walk(self.root_preds):
            for filename in filenames:
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    path_pred = os.path.join(dirpath, filename)

                    # 计算该文件相对于 root_preds 的路径
                    relative_path = os.path.relpath(path_pred, self.root_preds)

                    # 构建在 root_gts 中对应的文件完整路径
                    path_gt = os.path.join(self.root_gts, relative_path)

                    # 检查对应的文件是否存在
                    if os.path.exists(path_gt):
                        self.image_preds.append(path_pred)
                        self.image_gts.append(path_gt)

        # 排序以确保文件对的顺序是确定性的
        self.image_preds.sort()
        self.image_gts.sort()

        print(f"成功找到 {len(self.image_preds)} 对匹配的图像。")
        assert len(self.image_preds) > 0, "错误：在两个路径下没有找到任何匹配的图像文件对。"
        assert len(self.image_preds) == len(self.image_gts)

    def __len__(self):
        return len(self.image_preds)

    def __getitem__(self, idx):
        # for test
        if idx == 0:
            print(f"\n{self.image_gts[idx]=}")
            print(f"{self.image_preds[idx]=}")
        
        # 直接使用已经存储的完整路径读取图像
        img_gt = read_image(self.image_gts[idx])
        if self.is_gt:
            # resize to 1024 x 1024
            img_gt = Resize((1024, 1024))(img_gt)
        
        # 直接使用完整路径读取图像
        img_pred = read_image(self.image_preds[idx])

        batch_list = [img_pred, img_gt]

        return batch_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--is_gt", action="store_true")
    parser.add_argument("--input_root_preds", type=str, required=True)
    parser.add_argument("--input_root_gts", type=str, required=True)
    args = parser.parse_args()

    psnr = PeakSignalNoiseRatio(data_range=(0, 1), reduction="elementwise_mean", dim=(1, 2, 3)).to("cuda")
    ssim = StructuralSimilarityIndexMeasure(data_range=(0, 1)).to("cuda")
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to("cuda")

    dataset = MultiImageDataset(args.input_root_preds, args.input_root_gts, is_gt=args.is_gt)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    progress_bar = tqdm(dataloader)
    with torch.inference_mode():
        for i, batch in enumerate(progress_bar):
            batch = [img.to("cuda") / 255 for img in batch]
            batch_size = batch[0].shape[0]
            psnr.update(batch[0], batch[1])
            lpips.update(batch[0], batch[1])
            ssim.update(batch[0], batch[1])
    # fid_score = fid.compute_fid(args.input_root0, args.input_root1)

    print("\nPSNR:", psnr.compute().item())
    print("SSIM:", ssim.compute().item())
    print("LPIPS:", lpips.compute().item())
    # print("FID:", fid_score)
