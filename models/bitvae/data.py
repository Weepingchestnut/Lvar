# Copyright (c) Foundationvision, Inc. All Rights Reserved

import os
import os.path as osp
import random
import argparse
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
import torch.utils.data as data
import torch.distributed as dist
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from models.bitvae.ms_bsq import dynamic_resolution_h_w


DATASET_DICT = {
    "imagenet": {
        "train_label": "labels/imagenet/train.txt", 
        "val_label": "labels/imagenet/val.txt",
        "dataset_path":  "",
        "data_type": "image"
    },
    "openimages": {
        "train_label": "labels/openimages/train.txt", # 9M+ images
        "val_label": "labels/openimages/train.txt",
        "dataset_path":  "",
        "data_type": "image"
    },
    # fake data for debug
    "fake": {
        "train_label": "",
        "val_label": "",
        "dataset_path": "",
        "data_type": "image"
    }
}


def _pil_interp(method):
    if method == 'bicubic':
        return InterpolationMode.BICUBIC
    elif method == 'lanczos':
        return InterpolationMode.LANCZOS
    elif method == 'hamming':
        return InterpolationMode.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return InterpolationMode.BILINEAR

import timm.data.transforms as timm_transforms
timm_transforms._pil_interp = _pil_interp

def get_parent_dir(path):
    return osp.basename(osp.dirname(path))

# Only used during inference (for benchmarks like tuchong, which allows different resolutions)
class DynamicAspectRatioGroupedDataset(data.Dataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, 
        dataset, batch_size, 
        debug=False, seed=0, train=True, random_bucket_ratio=0.
    ):
        """
        Args:
            dataset: an iterable. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            batch_size (int):
        """
        self.train = train
        self._idx = 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.random_bucket_ratio = random_bucket_ratio
        self.aspect_ratio = list(dynamic_resolution_h_w.keys())
        num_buckets = len(self.aspect_ratio) * 3 # in each aspect-ratio, we have three scales
        self._buckets = [[] for _ in range(num_buckets)]
        self.debug = debug
        self.seed = seed
        if type(dataset) == ImageDataset:
            self.batch_factor = 0.6 # A100 config
        else:
            raise NotImplementedError
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def closest_id(self, v, type="width"):
        if type == "width":
            dist = np.array([abs(v - self.width[i]) for i in range(len(self.width))])
            return np.argmin(dist)
        else:
            dist = np.array([abs(v - self.aspect_ratio[i]) for i in range(len(self.aspect_ratio))])
            return np.argmin(dist)
    
    def __len__(self):
        return len(self.dataset) // self.batch_size ### an approximate value


    def collate_func(self, batch_list):
        batch = batch_list[0]

        return {
            "image": torch.stack([d["image"] for d in batch], dim=0),
            "label": [d["label"] for d in batch],
            "path": [d["path"] for d in batch],
            "height": [d["height"] for d in batch],
            "width": [d["width"] for d in batch],
            "type": [d["type"] for d in batch],
        }

    def get_batch_size(self, w_id, ar_id):
        sel_ar = self.aspect_ratio[ar_id]
        sel_w = self.width[w_id]
        dynamic_batch_size = int(self.batch_size / sel_ar / ((sel_w / self.max_width)**2) / self.batch_factor)
        return max(dynamic_batch_size, 1)
    
    def get_ar_w_new(self, w, ar, strategy="closest"):
        if strategy == "closest":
            # get new ar
            dist = np.array([abs(ar - self.aspect_ratio[i]) for i in range(len(self.aspect_ratio))])
            ar_id = np.argmin(dist)
            ar_new = self.aspect_ratio[ar_id]
            # get new w
            w_list = list(dynamic_resolution_h_w[ar_new].keys())
            dist = np.array([abs(w - w_list[i]) for i in range(len(w_list))])
            w_id = np.argmin(dist)
            w_new = w_list[w_id]
            h_new = dynamic_resolution_h_w[ar_new][w_new]["pixel"][0]
            return w_new, h_new, w_id, ar_id
        elif strategy == "random":
            raise NotImplementedError

    def get_aug(self, w, h, strategy="closest"):
        sel_h, sel_w = h, w
        assert sel_h % 8 == 0 and sel_w % 8 == 0
        aug_shape = (sel_h, sel_w)
        if strategy == "closest":
            aug = transforms.Resize(aug_shape) # resize to the closest size
        elif strategy == "random":
            min_edge = min(sel_w, sel_h)
            aug = transforms.Compose([
                transforms.Resize(min_edge),
                transforms.CenterCrop((sel_h, sel_w)),
            ])
        return aug

    def __getitem__(self, idx):
        d = self.dataset.__getitem__(idx)
        w, h = d["width"], d["height"]
        ar = h / w
        strategy = "random" if random.random() < self.random_bucket_ratio else "closest"
        w_new, h_new, w_id, ar_id = self.get_ar_w_new(w, ar, strategy=strategy)
        aug = self.get_aug(w_new, h_new, strategy=strategy)
        images = d["image"]
        assert images.ndim == 3
        d["image"] = aug(images)
        assert (d["image"].shape[1] % 8 == 0) and (d["image"].shape[2] % 8 == 0)
        
        return [d]
    
    def __iter__(self):
        # if not self.debug:
        while True:
            if self.train:
                idx = random.randint(0, self.dataset.__len__()-1)
            else:
                idx = self._idx
                self._idx = (self._idx + 1) % self.dataset.__len__()

            d = self.dataset.__getitem__(idx)
            w, h = d["width"], d["height"]
            ar = h / w
            strategy = "random" if random.random() < self.random_bucket_ratio else "closest"
            w_new, h_new, w_id, ar_id = self.get_ar_w_new(w, ar, strategy=strategy)
            aug = self.get_aug(w_new, h_new, strategy=strategy)
            images = d["image"]
            assert images.ndim == 3
            
            d["image"] = aug(images)
            assert (d["image"].shape[1] % 8 == 0) and (d["image"].shape[2] % 8 == 0)

            bucket_id = ar_id * 3 + w_id # TODO: fix this hardcode 3
            bucket = self._buckets[bucket_id]
            bucket.append(d)
            target_batch_size = self.get_batch_size(w_id, ar_id) if self.train else self.batch_size
            if len(bucket) == target_batch_size:
                data = bucket[:]
                # Clear bucket first, because code after yield is not
                # guaranteed to execute
                del bucket[:]
                yield data


class AspectRatioGroupedDataset(data.IterableDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, 
        dataset, batch_size, 
        width=[256, 320, 384, 448, 512], # A100 config
        aspect_ratio=[4/16, 6/16, 8/16, 9/16, 10/16, 12/16, 14/16, 1, 16/14, 16/12, 16/10, 16/9, 16/8, 16/6, 16/4], # A100 config
        max_resolution=512*512, # A100 config
        debug=False, seed=0, train=True, random_bucket_ratio=0.
    ):
        """
        Args:
            dataset: an iterable. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            batch_size (int):
        """
        self.train = train
        self._idx = 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.random_bucket_ratio = random_bucket_ratio
        num_buckets = len(width) * len(aspect_ratio)
        self.width = width
        self.max_width = max(width)
        self.aspect_ratio = aspect_ratio
        self._buckets = [[] for _ in range(num_buckets)]
        self.debug = debug
        self.seed = seed
        self.max_resolution = max_resolution
        if type(dataset) == ImageDataset:
            self.batch_factor = 0.6 # A100 config
        else:
            raise NotImplementedError
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def closest_id(self, v, type="width"):
        if type == "width":
            dist = np.array([abs(v - self.width[i]) for i in range(len(self.width))])
            return np.argmin(dist)
        else:
            dist = np.array([abs(v - self.aspect_ratio[i]) for i in range(len(self.aspect_ratio))])
            return np.argmin(dist)
    
    def __len__(self):
        return len(self.dataset) // self.batch_size ### an approximate value


    def collate_func(self, batch_list):
        batch = batch_list[0]

        return {
            "image": torch.stack([d["image"] for d in batch], dim=0),
            "label": [d["label"] for d in batch],
            "path": [d["path"] for d in batch],
            "height": [d["height"] for d in batch],
            "width": [d["width"] for d in batch],
            "type": [d["type"] for d in batch],
        }

    def get_batch_size(self, w_id, ar_id):
        sel_ar = self.aspect_ratio[ar_id]
        sel_w = self.width[w_id]
        dynamic_batch_size = int(self.batch_size / sel_ar / ((sel_w / self.max_width)**2) / self.batch_factor)
        return max(dynamic_batch_size, 1)

    def memory_safty_guard(self, w_id, ar_id):
        while True:
            sel_ar = self.aspect_ratio[ar_id]
            sel_w = self.width[w_id]
            if self.max_resolution < 0 or (sel_ar * sel_w) * sel_w <= self.max_resolution:
                break
            else:
                w_id = w_id - 1
        return w_id, ar_id
    
    def get_ar_w_id(self, w, ar, strategy="closest"):
        if strategy == "closest":
            w_id = self.closest_id(w, type="width")
            ar_id = self.closest_id(ar, type="aspect_ratio")   
        elif strategy == "random":
            h = w * ar
            ws = [_w for _w in self.width if _w <= w]
            _w = random.choice(ws) if len(ws) > 0 else self.width[0]
            w_id = self.width.index(_w)
            ars = [_ar for _ar in self.aspect_ratio if _ar * w < h]
            _ar = random.choice(ars) if len(ars) > 0 else self.aspect_ratio[0]
            ar_id = self.aspect_ratio.index(_ar)
        return self.memory_safty_guard(w_id, ar_id)

    def get_aug(self, w_id, ar_id, strategy="closest"):
        sel_ar = self.aspect_ratio[ar_id]
        sel_w = self.width[w_id]
        sel_h = int(sel_w * sel_ar)
        sel_h = (sel_h+4) - ((sel_h+4) % 8) # round by 8
        aug_shape = (sel_h, int(sel_w))
        if strategy == "closest":
            aug = transforms.Resize(aug_shape) # resize to the closest size
        elif strategy == "random":
            min_edge = min(sel_w, sel_h)
            aug = transforms.Compose([
                transforms.Resize(min_edge),
                transforms.CenterCrop((sel_h, sel_w)),
            ])
        return aug

    def __iter__(self):
        # if not self.debug:
        while True:
            if self.train:
                idx = random.randint(0, self.dataset.__len__()-1)
            else:
                idx = self._idx
                self._idx = (self._idx + 1) % self.dataset.__len__()

            d = self.dataset.__getitem__(idx)
            w, h = d["width"], d["height"]
            ar = h / w
            strategy = "random" if random.random() < self.random_bucket_ratio else "closest"
            w_id, ar_id = self.get_ar_w_id(w, ar, strategy=strategy)
            aug = self.get_aug(w_id, ar_id, strategy=strategy)
            images = d["image"]
            assert images.ndim == 3
            
            d["image"] = aug(images)
            assert (d["image"].shape[1] % 8 == 0) and (d["image"].shape[2] % 8 == 0)

            bucket_id = ar_id * len(self.width) + w_id
            bucket = self._buckets[bucket_id]
            bucket.append(d)
            target_batch_size = self.get_batch_size(w_id, ar_id) if self.train else self.batch_size
            if len(bucket) == target_batch_size:
                data = bucket[:]
                # Clear bucket first, because code after yield is not
                # guaranteed to execute
                del bucket[:]
                yield data


class FakeImageDataset(data.Dataset):
    """ 生成随机噪声数据用于 Debug """
    def __init__(self, resolution=256, length=100, train=True):
        self.resolution = resolution
        self.length = length # 假定数据集长度
        self.train = train
        
        # 模拟 ImageNet 的 1000 类
        self.n_classes = 1000 

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. 生成随机图像张量 [C, H, W]
        # BitVAE 通常期望输入范围在 [-1, 1]
        # 如果 args.resolution 是列表 [H, W]，这里取 resolution[0]
        res = self.resolution[0] if isinstance(self.resolution, list) else self.resolution
        
        # 生成随机噪声
        img = torch.randn(3, res, res)
        
        # 截断到 [-1, 1] 模拟归一化后的图像
        img = img.clamp(-1.0, 1.0)

        # 2. 返回符合 Pipeline 要求的字典
        return {
            "image": img,
            "label": random.randint(0, self.n_classes - 1),
            "path": f"fake_data/img_{idx}.jpg", # 假路径
            "height": res,
            "width": res,
            "type": "image" # 必须包含 type
        }
    
    # 兼容 AspectRatioGroupedDataset 需要的 collate_func 接口
    # 即使不开启 multi_resolution，有时代码也会检查这个属性
    def collate_func(self, batch_list):
        return {
            "image": torch.stack([d["image"] for d in batch_list], dim=0),
            "label": [d["label"] for d in batch_list],
            "path": [d["path"] for d in batch_list],
            "height": [d["height"] for d in batch_list],
            "width": [d["width"] for d in batch_list],
            "type": [d["type"] for d in batch_list],
        }


class ImageDataset(data.Dataset):
    """ Generic dataset for Images files stored in folders
    Returns BCHW Images in the range [-0.5, 0.5] """

    def __init__(self, data_folder, data_list, train=True, resolution=64, aug="resize"):
        """
        Args:
            data_folder: path to the folder with images. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding images stored
        """
        super().__init__()
        self.train = train
        self.data_folder = data_folder
        self.data_list = data_list
        self.resolution = resolution

        with open(self.data_list) as f:
            self.annotations = f.readlines()
        
        total_classes = 1000
        classes = []
        # from imagenet_stubs.imagenet_2012_labels import label_to_name
        # for i in range(total_classes):
        #     classes.append(label_to_name(i))
        
        self.classes = classes
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}
        self.label_to_class = {i: c for i, c in enumerate(self.classes)}

        crop_function = transforms.RandomCrop(resolution) if train else transforms.CenterCrop(resolution)
        flip_function = transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x) # flip if train else no op
        if aug == "resizecrop":
            augmentations = transforms.Compose([
                    transforms.Resize(min(resolution), interpolation=_pil_interp("bicubic")),
                    crop_function,
                    flip_function,
                    transforms.ToTensor(),
                ])
        elif aug == "crop":
            augmentations = transforms.Compose([
                    crop_function,
                    flip_function,
                    transforms.ToTensor(),
                ])
        elif aug == "keep":
            augmentations = transforms.Compose([
                    flip_function, 
                    transforms.ToTensor(),
                ])
        else:
            raise NotImplementedError
        
        self.aug = aug
        self.augmentations = augmentations
       

    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            ann = self.annotations[idx].strip()
            try:
                img_path, height, width = ann.split()
                img_label = -1
            
            except:
                img_path, img_label = ann.split()
            
            full_img_path = os.path.join(self.data_folder, img_path)
            
            img = Image.open(full_img_path).convert('RGB')
            h, w = img.height, img.width

            img = self.augmentations(img) * 2.0 - 1.0
            if self.aug != "keep":
                assert img.shape[1] == self.resolution[0] and img.shape[2] == self.resolution[1]

            return {"image": img, "label": int(img_label), "path": img_path, "height": h, "width": w, "type": "image"}
        except Exception as e:
            print(f"Error in dataloader {e}")
            return self.__getitem__((idx+1) % self.__len__())


class ImageData():

    def __init__(self, args, shuffle=True):
        super().__init__()
        self.args = args
        self.shuffle = shuffle

    @property
    def n_classes(self):
        dataset = self._dataset(True)
        return dataset[0].n_classes

    def _dataset(self, train):
        datasets = []
        for dataset, batch_size in zip(self.args.dataset_list, self.args.batch_size):
            dataset_path = DATASET_DICT[dataset]["dataset_path"]
            data_type = DATASET_DICT[dataset]["data_type"]
            train_label = DATASET_DICT[dataset]["train_label"]
            val_label = DATASET_DICT[dataset]["val_label"]
            # for debug
            if dataset == "fake":
                dataset = FakeImageDataset(
                    resolution=self.args.resolution, 
                    length=1000 if train else 100, # 训练集 1000 张，验证集 100 张
                    train=train
                )
            elif data_type == "image":
                dataset = ImageDataset(
                    dataset_path, train_label if train else val_label, train=train, resolution=self.args.resolution, aug=self.args.dataaug
                )
            
            if self.args.multi_resolution:
                # assert len(self.args.data_path) == 1
                if train:
                    dataset = AspectRatioGroupedDataset(
                        dataset, batch_size=batch_size, debug=self.args.debug, train=train, random_bucket_ratio=self.args.random_bucket_ratio
                    )
                else:
                    dataset = DynamicAspectRatioGroupedDataset(
                        dataset, batch_size=batch_size, debug=self.args.debug, train=train, random_bucket_ratio=self.args.random_bucket_ratio
                    )
            datasets.append(dataset)
        return datasets

    def _dataloader(self, train):
        dataset = self._dataset(train)
        # print(self.args.batch_size)
        if isinstance(self.args.batch_size, int):
            self.args.batch_size = [self.args.batch_size]
        
        assert len(dataset) == len(self.args.batch_size)
        dataloaders = []
        for dset, d_batch_size in zip(dataset, self.args.batch_size):
            if dist.is_initialized():
                sampler = data.distributed.DistributedSampler(
                    dset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
                )
                global_rank = dist.get_rank()
            else:
                sampler = None
                global_rank = None
            
            def seed_worker(worker_id):
                if global_rank:
                    seed = self.args.num_workers * global_rank + worker_id
                else:
                    seed = worker_id
                # print(f"Setting dataloader worker {worker_id} on GPU {global_rank} as seed {seed}")
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

            dataloader = data.DataLoader(
                dset,
                batch_size=d_batch_size if not self.args.multi_resolution else 1,
                num_workers=self.args.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                sampler=sampler if not isinstance(dset, data.IterableDataset) else None,
                collate_fn=dset.collate_func if hasattr(dset, "collate_func") else None,
                shuffle=sampler is None and train,
                drop_last=True,
            )

            dataloaders.append(dataloader)
        
        return dataloaders

    def train_dataloader(self):
        return self._dataloader(True)

    def val_dataloader(self):
        return self._dataloader(False)[0]

    def test_dataloader(self):
        return self.val_dataloader()


    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_path', type=str, nargs="+", default=[""])
        parser.add_argument('--data_type', type=str, nargs="+", default=[""])
        parser.add_argument('--dataset_list', type=str, nargs="+", default=[''])
        parser.add_argument('--dataaug', type=str, choices=["resize", "resizecrop", "crop", "keep"])
        parser.add_argument('--multi_resolution', action="store_true")
        parser.add_argument('--random_bucket_ratio', type=float, default=0.)

        parser.add_argument('--resolution', type=int, nargs="+", default=[512])
        parser.add_argument('--batch_size', type=int, nargs="+", default=[32])
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--image_channels', type=int, default=3)

        return parser
