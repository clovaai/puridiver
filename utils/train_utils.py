"""
PuriDivER
Copyright 2022-present NAVER Corp.
GPLv3
"""
import torch
from easydict import EasyDict as edict

from models import cifar, imagenet


def select_model(model_name, dataset, num_classes=None):
    opt = edict(
        {
            "depth": 18,
            "num_classes": num_classes,
            "in_channels": 3,
            "bn": True,
            "normtype": "BatchNorm",
            "activetype": "ReLU",
            "pooltype": "MaxPool2d",
            "preact": False,
            "affine_bn": True,
            "bn_eps": 1e-6,
            "compression": 0.5,
        }
    )

    if "cifar" in dataset:
        model_class = getattr(cifar, "ResNet")
    elif dataset in ["WebVision-V1-2", "Food-101N"]:
        model_class = getattr(imagenet, "ResNet")
    else:
        raise NotImplementedError(
            "Please select the appropriate datasets (cifar10, cifar100, WebVision-V1-2, Food-101N)"
        )

    if model_name == "resnet18":
        opt["depth"] = 18
    elif model_name == "resnet32":
        opt["depth"] = 32
    elif model_name == "resnet34":
        opt["depth"] = 34
    else:
        raise NotImplementedError(
            "Please choose the model name in [resnet18, resnet32, resnet34]"
        )

    model = model_class(opt)

    return model

# This function is from ...
# https://github.com/pytorch/pytorch/issues/11959
def soft_cross_entropy_loss(input, target, reduction='mean'):
    """
    :param input: (batch, *)
    :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
    """
    logprobs = torch.nn.functional.log_softmax(input.view(input.shape[0], -1), dim=1)
    batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
    if reduction == 'none':
        return batchloss
    elif reduction == 'mean':
        return torch.mean(batchloss)
    elif reduction == 'sum':
        return torch.sum(batchloss)
    else:
        raise NotImplementedError('Unsupported reduction mode.')