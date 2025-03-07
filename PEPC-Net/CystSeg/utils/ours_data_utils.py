

import math
import os
import numpy as np
import torch
from monai import data, transforms
from monai.data import load_decathlon_datalist
from CystSeg.utils.transform import RandStimulateLowResolutionTransformd


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank: self.total_size: self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank: self.total_size: self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label", "edge"]),
            transforms.EnsureChannelFirstd(keys=["image", "label", "edge"]),
            transforms.Orientationd(keys=["image", "label", "edge"], axcodes="RAS"),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label", "edge"], source_key="image"),
            transforms.SpatialPadd(keys=["image", "label", "edge"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label", "edge"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=2,
                neg=1,
                num_samples=args.sw_batch_size,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label", "edge"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label", "edge"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label", "edge"], prob=0.5, spatial_axis=2),

            transforms.RandZoomd(keys=["image", "label", "edge"], min_zoom=0.7, max_zoom=1.4, prob=0.2,
                                 mode=["trilinear", "nearest", "trilinear"]),
            transforms.RandRotated(keys=["image", "label", "edge"], range_x=30 / 360 * 2 * np.pi,
                                   range_y=30 / 360 * 2 * np.pi, range_z=30 / 360 * 2 * np.pi, prob=0.2,
                                   mode=["trilinear", "nearest", "trilinear"]),

            transforms.RandGaussianNoiseD(keys=["image"], std=0.1, prob=0.1),
            transforms.RandGaussianSmoothD(keys=["image"], sigma_x=(0.5, 1), sigma_y=(0.5, 1), sigma_z=(0.5, 1.5),
                                           prob=0.2),
            transforms.RandScaleIntensityd(keys="image", factors=(-0.25, 0.25), prob=0.15),
            transforms.ScaleIntensityd(keys="image", minv=0, maxv=1),
            RandStimulateLowResolutionTransformd(keys="image", prob=0.25, factors=(0.5, 1)),
            transforms.RandAdjustContrastd(keys="image", gamma=(0.7, 1.5), prob=0.15),
            transforms.ToTensord(keys=["image", "label", "edge"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "test", base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=test_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
            )
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader
