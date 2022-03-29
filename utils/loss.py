"""
PuriDivER
Copyright 2022-present NAVER Corp.
GPLv3
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def coteaching_loss(y1, y2, label, r_t, cutmix=False, label_b=None, lam=None):
    criterion = nn.CrossEntropyLoss(reduction='none').to(label.device)
    if cutmix:
        assert label_b is not None and lam is not None
        loss_1 = lam * criterion(y1, label) + (1 - lam) * criterion(y1, label_b)
        loss_2 = lam * criterion(y2, label) + (1 - lam) * criterion(y2, label_b)
    else:
        loss_1 = criterion(y1, label)
        loss_2 = criterion(y2, label)
    num_to_use = math.ceil(r_t * len(label))
    ind_to_use_2 = torch.argsort(loss_1)[:num_to_use]
    ind_to_use_1 = torch.argsort(loss_2)[:num_to_use]
    loss = torch.mean(loss_1[ind_to_use_1]) + torch.mean(loss_2[ind_to_use_2])
    return loss


def linear_rampup(current, warm_up, lambda_u=25, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return lambda_u * float(current)


def dividemix_loss(outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
    probs_u = torch.softmax(outputs_u, dim=1)

    Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
    Lu = torch.mean((probs_u - targets_u) ** 2)

    return Lx, Lu, linear_rampup(epoch, warm_up)


def neg_entropy_loss(outputs):
    probs = torch.softmax(outputs, dim=1)
    return torch.mean(torch.sum(probs.log() * probs, dim=1))
