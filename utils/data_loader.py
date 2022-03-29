"""
PuriDivER
Copyright 2022-present NAVER Corp.
GPLv3
"""
import logging.config
import os
from typing import List

import PIL
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger()


class ImageDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, dataset_path: str, transform=None):
        self.data_frame = data_frame
        self.dataset_path = dataset_path
        self.transform = transform
        self.prob = None

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_frame.iloc[idx]["file_name"]
        label = self.data_frame.iloc[idx].get("label", -1)
        true_label = self.data_frame.iloc[idx].get("true_label", -1)  # for noisy label

        img_path = os.path.join(self.dataset_path, img_name)
        image = PIL.Image.open(img_path).convert("RGB")
        if type(self.transform) == list:
            origin = self.transform[0](image)
            sample["origin_img"] = origin
            sample["test_img"] = self.transform[2](image)
            image = self.transform[1](image)
        elif self.transform is not None:
            image = self.transform(image)

        sample["image"] = image
        sample["label"] = int(label)
        sample["true_label"] = int(true_label) if true_label != -1 else int(label)
        sample["image_name"] = img_name
        if "logit" in self.data_frame.keys():
            sample["logit"] = self.data_frame.iloc[idx]["logit"]
        if self.prob is not None:
            sample["prob"] = self.prob[idx, :]
        return sample

    def set_prob(self, prob):
        assert prob.shape[0] == len(self.data_frame), \
            f"Expected prob shape {len(self.data_frame)} x -1, but {prob.shape[0]} x -1"
        self.prob = prob


def get_train_datalist(args, cur_iter: int, seed: int) -> List:
    # Download data collections from github
    collections_path = "tasks"
    collection_name = get_train_collection_name(
        dataset=args.dataset,
        exp=args.exp_name,
        rnd=seed,
        n_cls=args.n_cls_a_task,
        iter=cur_iter,
    )
    json_path = f"{collections_path}/{args.dataset}/{collection_name}.json"
    assert os.path.exists(json_path), f"json file not found: {json_path}"
    datalist = pd.read_json(json_path).to_dict(orient="records")
    logger.info(f"[Train] Get datalist from {collection_name}.json")
    return datalist


def get_train_collection_name(dataset, exp, rnd, n_cls, iter):
    collection_name = "{dataset}_train_{exp}_rand{rnd}_cls{n_cls}_task{iter}".format(
        dataset=dataset, exp=exp, rnd=rnd, n_cls=n_cls, iter=iter
    )
    return collection_name


def get_test_datalist(args, exp_name: str, cur_iter: int, seed: int) -> List:
    # Download data collections from github
    collections_path = "tasks"

    if exp_name is None:
        exp_name = args.exp_name

    if "disjoint" in exp_name:
        # merge current and all previous tasks
        tasks = list(range(cur_iter + 1))
    else:
        # merge over all tasks
        tasks = list(range(args.n_tasks))

    datalist = []
    for iter_ in tasks:
        collection_name = "{dataset}_test_rand{rnd}_cls{n_cls}_task{iter}".format(
            dataset=args.dataset, rnd=seed, n_cls=args.n_cls_a_task, iter=iter_
        )
        datalist += pd.read_json(
            f"{collections_path}/{args.dataset}/{collection_name}.json"
        ).to_dict(orient="records")
        logger.info(f"[Test ] Get datalist from {collection_name}.json")

    return datalist


def get_statistics(dataset: str):
    """
    Returns statistics of the dataset given a string of dataset name. To add new dataset, please add required statistics here
    """
    assert dataset in [
        "cifar10",
        "cifar100",
        "WebVision-V1-2",
        "Food-101N"
    ]
    mean = {
        "cifar10": (0.4914, 0.4822, 0.4465),
        "cifar100": (0.5071, 0.4867, 0.4408),
        "WebVision-V1-2": (0.485, 0.456, 0.406),
        "Food-101N": (0.0, 0.0, 0.0)
    }

    std = {
        "cifar10": (0.2023, 0.1994, 0.2010),
        "cifar100": (0.2675, 0.2565, 0.2761),
        "WebVision-V1-2": (0.229, 0.224, 0.225),
        "Food-101N": (1.0, 1.0, 1.0)
    }

    classes = {
        "cifar10": 10,
        "cifar100": 100,
        "WebVision-V1-2": 50,
        "Food-101N": 101
    }

    in_channels = {
        "cifar10": 3,
        "cifar100": 3,
        "WebVision-V1-2": 3,
        "Food-101N": 3
    }

    inp_size = {
        "cifar10": 32,
        "cifar100": 32,
        "WebVision-V1-2": 224,
        "Food-101N": 224
    }
    return (
        mean[dataset],
        std[dataset],
        classes[dataset],
        inp_size[dataset],
        in_channels[dataset],
    )


# from https://github.com/drimpossible/GDumb/blob/74a5e814afd89b19476cd0ea4287d09a7df3c7a8/src/utils.py#L102:5
def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
