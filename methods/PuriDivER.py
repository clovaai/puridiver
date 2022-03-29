"""
PuriDivER
Copyright 2022-present NAVER Corp.
GPLv3
"""
import logging
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from utils.augment import Cutout, Solarize, Invert
from utils.data_loader import cutmix_data, ImageDataset
from utils.loss import coteaching_loss, dividemix_loss, neg_entropy_loss
from utils.train_utils import select_model, soft_cross_entropy_loss

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class PuriDivER:
    def __init__(
            self, criterion, device, train_transform, test_transform, n_classes, additional_trans, **kwargs
    ):
        self.num_learned_class = 0
        self.num_learning_class = kwargs["n_init_cls"]
        self.n_classes = n_classes
        self.learned_classes = []
        self.class_mean = [None] * n_classes
        self.exposed_classes = []
        self.seen = 1
        self.topk = kwargs["topk"]
        self.label_map = {}

        self.device = device
        self.criterion = criterion
        self.dataset = kwargs["dataset"]
        self.dataset_path = kwargs["dataset_path"]
        self.model_name = kwargs["model_name"]
        self.lr = kwargs["lr"]

        self.train_transform = train_transform
        self.cutmix = "cutmix" in kwargs["transforms"]
        self.test_transform = test_transform

        self.prev_streamed_list = []
        self.streamed_list = []
        self.test_list = []
        self.memory_list = []
        self.memory_size = kwargs["memory_size"]
        self.mem_manage = kwargs["mem_manage"]

        self.model = select_model(self.model_name, self.dataset, kwargs["n_init_cls"])
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        self.already_mem_update = False

        np.random.seed(1)
        self.batch_size = kwargs["batchsize"]
        self.n_worker = kwargs["n_worker"]
        self.origin_coeff = kwargs["coeff"]
        self.coeff = kwargs["coeff"]
        self.exp_name = kwargs["exp_name"]

        self.robust_type = kwargs["robust_type"]

        self.seen_per_cls = {}

        if self.robust_type in ['CoTeaching', 'DivideMix']:
            self.model_2 = select_model(self.model_name, self.dataset, kwargs["n_init_cls"])
            self.model_2 = self.model_2.to(self.device)
        self.noise_rate = kwargs["noise_rate"]
        self.warmup = kwargs["warmup"]
        self.weak_transform = additional_trans

        # For getting result consistently.
        np.random.seed(1)

    def train(self, cur_iter, n_epoch, batch_size, n_worker):
        if len(self.memory_list) > 0:
            mem_dataset = ImageDataset(
                pd.DataFrame(self.memory_list),
                dataset_path=self.dataset_path,
                transform=self.train_transform,
            )
            if self.robust_type == "PuriDivER":
                mem_dataset = ImageDataset(
                    pd.DataFrame(self.memory_list),
                    dataset_path=self.dataset_path,
                    transform=[self.weak_transform, self.train_transform, self.test_transform],
                )
                split_dataset = ImageDataset(
                    pd.DataFrame(self.memory_list),
                    dataset_path=self.dataset_path,
                    transform=self.test_transform,
                )
            elif self.robust_type == 'DivideMix':
                mem_dataset = ImageDataset(
                    pd.DataFrame(self.memory_list),
                    dataset_path=self.dataset_path,
                    transform=[self.weak_transform, self.weak_transform, self.test_transform],
                )
                split_dataset = ImageDataset(
                    pd.DataFrame(self.memory_list),
                    dataset_path=self.dataset_path,
                    transform=self.test_transform,
                )
            memory_loader = DataLoader(
                mem_dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=n_worker,
            )
            stream_batch_size = batch_size
        else:
            memory_loader = None
            stream_batch_size = batch_size

        # train_list == streamed_list in RM
        train_list = self.streamed_list
        test_list = self.test_list

        # Configuring a batch with streamed and memory data equally.
        train_loader, test_loader = self.get_dataloader(
            stream_batch_size, n_worker, train_list, test_list
        )

        logger.info(f"Streamed samples: {len(self.streamed_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Train samples: {len(train_list) + len(self.memory_list)}")
        logger.info(f"Test samples: {len(test_list)}")

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr
        if self.robust_type == 'DivideMix':
            for param_group in self.optimizer_2.param_groups:
                param_group["lr"] = self.lr

        # TRAIN
        best_acc = 0.0
        eval_dict = dict()
        cnt_consistency = dict()
        self.model = self.model.to(self.device)

        for epoch in range(n_epoch):
            if self.robust_type == "SELFIE" and epoch >= self.warmup:
                train_loss, train_acc, cnt_consistency = self.selfie(memory_loader=memory_loader,
                                                                     optimizer=self.optimizer, criterion=self.criterion,
                                                                     cnt_consistency=cnt_consistency)
            elif self.robust_type == "CoTeaching":
                train_loss, train_acc = self.coteaching(memory_loader=memory_loader, optimizer=self.optimizer,
                                                        r_t=1 - min([epoch / 10 * (self.noise_rate), self.noise_rate]))
            elif self.robust_type == "DivideMix" and epoch >= self.warmup:
                label_loader, unlabel_loader = self.split_data(dataset=mem_dataset, test_dataset=split_dataset, n=2,
                                                               model=self.model_2)
                self.dividemix(epoch, self.model, self.model_2, self.optimizer, label_loader, unlabel_loader,
                               warm_up=self.warmup)
                label_loader, unlabel_loader = self.split_data(dataset=mem_dataset, test_dataset=split_dataset, n=2,
                                                               model=self.model)
                self.dividemix(epoch, self.model_2, self.model, self.optimizer_2, label_loader, unlabel_loader,
                               warm_up=self.warmup)

            elif self.robust_type == "PuriDivER" and epoch >= self.warmup:
                correct_loader, ambiguous_loader, incorrect_loader = self.puridiver_split(epoch, dataset=mem_dataset, n=2)
                if ambiguous_loader is not None and incorrect_loader is not None:
                    train_loss, train_acc = self.puridiver(correct_loader, ambiguous_loader,
                                                           incorrect_loader,
                                                           optimizer=self.optimizer)
                else:
                    train_loss, train_acc = self._train(memory_loader=memory_loader,
                                                        optimizer=self.optimizer, criterion=self.criterion)
            else:
                train_loss, train_acc = self._train(memory_loader=memory_loader,
                                                    optimizer=self.optimizer, criterion=self.criterion)
            eval_dict = self.evaluation(
                model=self.model, test_loader=test_loader, criterion=self.criterion
            )

            writer.add_scalar(f"task{cur_iter}/train/loss", train_loss, epoch)
            writer.add_scalar(f"task{cur_iter}/train/acc", train_acc, epoch)
            writer.add_scalar(f"task{cur_iter}/test/loss", eval_dict["avg_loss"], epoch)
            writer.add_scalar(f"task{cur_iter}/test/acc", eval_dict["avg_acc"], epoch)
            writer.add_scalar(f"task{cur_iter}/train/lr", self.optimizer.param_groups[0]["lr"], epoch)

            logger.info(
                f"Task {cur_iter} | Epoch {epoch + 1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )

            self.scheduler.step()
            if self.robust_type == 'DivideMix':
                self.scheduler_2.step()

            best_acc = max(best_acc, eval_dict["avg_acc"])

        return best_acc, eval_dict

    def split_data(self, dataset, test_dataset, n, model=None):
        assert n in [2, 3], "N should be 2 or 3"
        if model is None:
            model = self.model

        CE = nn.CrossEntropyLoss(reduction='none')
        model.eval()
        loader = DataLoader(test_dataset,
                            shuffle=False,
                            batch_size=64,
                            num_workers=2,
                            )
        losses = torch.tensor([])
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                inputs = data["image"]
                targets = data["label"]
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = CE(outputs, targets)
                losses = torch.cat([losses, loss.detach().cpu()])
        losses = (losses - losses.min()) / (losses.max() - losses.min())
        input_loss = losses.reshape(-1, 1)

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=n, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        mean_index = np.argsort(gmm.means_, axis=0)
        prob = prob[:, mean_index]
        dataset.set_prob(prob.squeeze(axis=-1))
        pred = prob.argmax(axis=1)

        idx = np.where(pred == 0)[0]
        correct_size = len(idx)
        if correct_size == 0:
            return [None for _ in range(n)]

        dataloader_correct = DataLoader(torch.utils.data.Subset(dataset, idx),
                                        shuffle=True,
                                        batch_size=self.batch_size,
                                        num_workers=2
                                        )

        idx = np.where(pred == 1)[0]
        amb_size = len(idx)
        batch_size = int(amb_size / correct_size * self.batch_size)
        if batch_size < 2:
            batch_size = 2

        if amb_size <= 2:
            dataloader_ambiguous = None
        else:
            dataloader_ambiguous = DataLoader(torch.utils.data.Subset(dataset, idx),
                                              shuffle=True,
                                              batch_size=batch_size,
                                              num_workers=2
                                              )

        if n == 3:
            idx = np.where(pred == 2)[0]
            incorrect_size = len(idx)
            batch_size = int(incorrect_size / correct_size * self.batch_size)
            if batch_size < 2:
                batch_size = 2
            if incorrect_size <= 2:
                dataloader_incorrect = None
            else:
                dataloader_incorrect = DataLoader(torch.utils.data.Subset(dataset, idx),
                                                  shuffle=True,
                                                  batch_size=batch_size,
                                                  num_workers=2
                                                  )

            logger.info(f"n_correct: {correct_size}\tn_ambiguous: {amb_size}\tn_incorrect: {incorrect_size}")
            return dataloader_correct, dataloader_ambiguous, dataloader_incorrect
        logger.info(f"n_correct: {correct_size}\tn_ambiguous: {amb_size}")
        return dataloader_correct, dataloader_ambiguous

    def puridiver_split(self, epoch, dataset, n, plot_gmm=False):
        assert n in [2], "N should be 2"
        dataloader_correct, dataloader_ambiguous, dataloader_incorrect = None, None, None
        CE = nn.CrossEntropyLoss(reduction='none')
        SM = torch.nn.Softmax(dim=1)
        self.model.eval()
        loader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=64,
                            num_workers=2,
                            )
        losses = torch.tensor([])
        uncertainties = torch.tensor([])
        if plot_gmm:
            clean_noises = torch.tensor([], dtype=torch.bool)  # for plot
            cert_uncerts = torch.tensor([], dtype=torch.bool)  # for plot
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                inputs = data["test_img"]
                targets = data["label"]
                y_true = data["true_label"]  # for plotting.
                inputs = inputs.cuda()
                outputs = self.model(inputs)
                logits = SM(outputs)
                uncerts = 1 - torch.max(logits, 1)[0]
                if plot_gmm:
                    clean_noises = torch.cat([clean_noises, (targets == y_true)])  # for plot
                    cert_uncerts = torch.cat(
                        [cert_uncerts, (outputs.detach().cpu().argmax(axis=1) == y_true)])  # for plot
                    # true_targets = torch.cat([true_targets, y_true])  # for plot
                    # pred_targets = torch.cat([pred_targets, outputs.detach().cpu()])
                targets = targets.cuda()
                loss = CE(outputs, targets)
                losses = torch.cat([losses, loss.detach().cpu()])
                uncertainties = torch.cat([uncertainties, uncerts.detach().cpu()])

        losses = (losses - losses.min()) / (losses.max() - losses.min())
        input_loss = losses.reshape(-1, 1)
        uncertainties = uncertainties.reshape(-1, 1)

        # fit a two-component GMM to the loss
        gmm_loss = GaussianMixture(n_components=n, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm_loss.fit(input_loss)
        gmm_loss_means = gmm_loss.means_
        if gmm_loss_means[0] <= gmm_loss_means[1]:
            small_loss_idx = 0
            large_loss_idx = 1
        else:
            small_loss_idx = 1
            large_loss_idx = 0

        prob = gmm_loss.predict_proba(input_loss)
        dataset.set_prob(prob)
        pred = prob.argmax(axis=1)

        idx = np.where(pred == small_loss_idx)[0]
        correct_size = len(idx)
        if correct_size == 0:
            return None, None, None

        dataloader_correct = DataLoader(torch.utils.data.Subset(dataset, idx),
                                        shuffle=True,
                                        batch_size=self.batch_size,
                                        num_workers=2,
                                        )
        # 2nd GMM using large loss datasets
        idx = np.where(pred == large_loss_idx)[0]
        high_loss_size = len(idx)
        batch_size = int(high_loss_size / correct_size * self.batch_size)
        if batch_size < 2:
            batch_size = 2

        if high_loss_size <= 2:
            dataloader_ambiguous = None
            dataloader_incorrect = None
        else:
            # fit a two-component GMM to the loss
            gmm_uncert = GaussianMixture(n_components=n, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm_uncert.fit(uncertainties[idx])
            prob_uncert = gmm_uncert.predict_proba(uncertainties[idx])
            pred_uncert = prob_uncert.argmax(axis=1)

            if gmm_uncert.means_[0] <= gmm_uncert.means_[1]:
                small_loss_idx = 0
                large_loss_idx = 1
            else:
                small_loss_idx = 1
                large_loss_idx = 0

            idx_uncert = np.where(pred_uncert == small_loss_idx)[0]
            amb_size = len(idx_uncert)
            batch_size = int(amb_size / correct_size * self.batch_size)
            if batch_size < 2:
                batch_size = 2

            if amb_size <= 2:
                dataloader_ambiguous = None
            else:
                dataloader_ambiguous = DataLoader(torch.utils.data.Subset(dataset, idx[idx_uncert]),
                                                  shuffle=True,
                                                  batch_size=batch_size,
                                                  num_workers=2,
                                                  )

            idx_uncert = np.where(pred_uncert == large_loss_idx)[0]
            incorrect_size = len(idx_uncert)
            batch_size = int(incorrect_size / correct_size * self.batch_size)
            if batch_size < 2:
                batch_size = 2
            if incorrect_size <= 2:
                dataloader_incorrect = None
            else:
                dataloader_incorrect = DataLoader(torch.utils.data.Subset(dataset, idx[idx_uncert]),
                                                  shuffle=True,
                                                  batch_size=batch_size,
                                                  num_workers=2,
                                                  )
            logger.info(f"n_correct: {correct_size}\tn_ambiguous: {amb_size}\tn_incorrect: {incorrect_size}")
        logger.info(f"n_correct: {correct_size}\tn_high_loss: {high_loss_size}")

        return dataloader_correct, dataloader_ambiguous, dataloader_incorrect

    def update_model(self, x, y, criterion, optimizer):
        optimizer.zero_grad()
        if self.robust_type == 'DivideMix':
            self.optimizer_2.zero_grad()

        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            logit = self.model(x)
            loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(
                logit, labels_b
            )
            if self.robust_type in ['CoTeaching', 'DivideMix']:
                logit_2 = self.model_2(x)
                loss += lam * criterion(logit_2, labels_a) + (1 - lam) * criterion(
                    logit_2, labels_b
                )
        else:
            logit = self.model(x)
            loss = criterion(logit, y)
            if self.robust_type in ['CoTeaching', 'DivideMix']:
                logit_2 = self.model_2(x)
                loss += criterion(logit_2, y)
                if self.robust_type == 'DivideMix' and 'asymN' in self.exp_name:
                    loss += neg_entropy_loss(logit) + neg_entropy_loss(logit_2)

        _, preds = logit.topk(self.topk, 1, True, True)

        loss.backward()
        optimizer.step()
        if self.robust_type == 'DivideMix':
            self.optimizer_2.step()

        return loss.item(), torch.sum(preds == y.unsqueeze(1)).item(), y.size(0)

    def puridiver(self, loader_L, loader_U, loader_R, optimizer):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        criterion_L = torch.nn.CrossEntropyLoss()
        criterion_U = torch.nn.MSELoss()

        unlabeled_train_iter = iter(loader_U)
        relabeled_train_iter = iter(loader_R)

        self.model.train()
        for data in loader_L:
            x_l = data["image"]
            y_l = data["label"]
            try:
                data_r = relabeled_train_iter.next()
            except:
                relabeled_train_iter = iter(loader_R)
                data_r = relabeled_train_iter.next()
            try:
                data_u = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(loader_U)
                data_u = unlabeled_train_iter.next()

            x_pseudo = data_r["origin_img"]
            x_r = data_r["image"]
            y_r = data_r["label"]
            y_r = torch.nn.functional.one_hot(y_r, num_classes=self.n_classes)
            correct_prob = data_r["prob"][:, 0]
            correct_prob = correct_prob.unsqueeze(axis=1).expand(-1, self.n_classes)

            x_u_weak = data_u["origin_img"]
            x_u_strong = data_u["image"]

            y_l = y_l.to(self.device)

            optimizer.zero_grad()

            do_cutmix = self.cutmix and np.random.rand(1) < 0.5
            if do_cutmix:
                x_cutmix, labels_a, labels_b, lam = cutmix_data(x=x_l, y=y_l, alpha=1.0)
                x_concat = torch.cat([x_pseudo, x_r, x_u_weak, x_u_strong, x_cutmix])
                x_concat = x_concat.to(self.device)
                logit = self.model(x_concat)
                r_size = x_pseudo.size(0)
                u_size = x_u_weak.size(0)
                logit_pseudo, logit_r, logit_u_weak, logit_u_strong, logit_cutmix = \
                    logit[:r_size], logit[r_size:2 * r_size], logit[2 * r_size: 2 * r_size + u_size], \
                    logit[2 * r_size + u_size:2 * r_size + 2 * u_size], logit[2 * r_size + 2 * u_size:]

                logit_pseudo_softmax = torch.nn.functional.softmax(logit_pseudo, dim=1)

                loss_L = lam * criterion_L(logit_cutmix, labels_a) + (1 - lam) * criterion_L(
                    logit_cutmix, labels_b
                )
                soft_pseudo = correct_prob * y_r + (1 - correct_prob) * logit_pseudo_softmax.detach().cpu()
                soft_pseudo = soft_pseudo.to(self.device)

                loss_R = soft_cross_entropy_loss(logit_r, soft_pseudo)
                loss_U = criterion_U(logit_u_strong, logit_u_weak)
                loss = (y_l.size(0) / (y_l.size(0) + y_r.size(0) + u_size)) * loss_L + \
                       (y_r.size(0) / (y_l.size(0) + y_r.size(0) + u_size)) * loss_R + \
                       (u_size / (y_l.size(0) + y_r.size(0) + u_size)) * loss_U

                print(f"Loss L: {loss_L.item()} | Loss R: {loss_R.item()} | Loss U: {loss_U.item()}")
                _, preds = logit_cutmix.topk(self.topk, 1, True, True)
            else:
                x_concat = torch.cat([x_pseudo, x_r, x_u_weak, x_u_strong, x_l])
                x_concat = x_concat.to(self.device)
                logit = self.model(x_concat)
                r_size = x_pseudo.size(0)
                u_size = x_u_weak.size(0)
                logit_pseudo, logit_r, logit_u_weak, logit_u_strong, logit_l = \
                    logit[:r_size], logit[r_size:2 * r_size], logit[2 * r_size:2 * r_size + u_size], \
                    logit[2 * r_size + u_size:2 * r_size + 2 * u_size], logit[2 * r_size + 2 * u_size:]

                logit_pseudo_softmax = torch.nn.functional.softmax(logit_pseudo, dim=1)

                soft_pseudo = correct_prob * y_r + (1 - correct_prob) * logit_pseudo_softmax.detach().cpu()
                soft_pseudo = soft_pseudo.to(self.device)

                loss = (y_l.size(0) / (y_l.size(0) + y_r.size(0) + u_size)) * criterion_L(logit_l, y_l) + \
                       (y_r.size(0) / (y_l.size(0) + y_r.size(0) + u_size)) * soft_cross_entropy_loss(logit_r,
                                                                                                      soft_pseudo) + \
                       (u_size / (y_l.size(0) + y_r.size(0) + u_size)) * criterion_U(logit_u_weak, logit_u_strong)

                _, preds = logit_l.topk(self.topk, 1, True, True)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += torch.sum(preds == y_l.unsqueeze(1)).item()
            num_data += y_l.size(0)

        n_batches = len(loader_L)
        return total_loss / n_batches, correct / num_data

    def selfie(self, memory_loader, optimizer, criterion, cnt_consistency):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        ths = 5
        self.model.train()
        for data in memory_loader:
            x = data["image"]
            y = data["label"]
            y_true = data["true_label"]
            img_names = data["image_name"]

            for idx, name in enumerate(img_names):
                if name not in cnt_consistency.keys():
                    cnt_consistency[name] = [0, 0]

                if cnt_consistency[name][1] >= ths:
                    y[idx] = cnt_consistency[name][0]

            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()
            do_cutmix = self.cutmix and np.random.rand(1) < 0.5
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                logit = self.model(x)

                loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(
                    logit, labels_b
                )
            else:
                logit = self.model(x)
                loss = criterion(logit, y)

            _, preds = logit.topk(self.topk, 1, True, True)
            preds = preds.detach().cpu()
            y = y.detach().cpu()

            for idx, name in enumerate(img_names):
                if cnt_consistency[name][0] == preds[idx]:
                    cnt_consistency[name][1] = cnt_consistency[name][1] + 1 if cnt_consistency[name][1] < ths else ths
                else:
                    cnt_consistency[name][0] = preds[idx].item()
                    cnt_consistency[name][1] = 1

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += torch.sum(y == y_true).item()
            num_data += y.size(0)

        n_batches = len(memory_loader)

        return total_loss / n_batches, correct / num_data, cnt_consistency

    def coteaching(
            self, memory_loader, optimizer, r_t
    ):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        self.model.train()
        self.model_2.train()
        for data in memory_loader:
            x = data["image"]
            y = data["label"]

            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()

            do_cutmix = self.cutmix and np.random.rand(1) < 0.5
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                logit = self.model(x)
                logit_2 = self.model_2(x)

                loss = coteaching_loss(logit, logit_2, labels_a, r_t, cutmix=True,
                                       label_b=labels_b, lam=lam)
            else:
                logit = self.model(x)
                logit_2 = self.model_2(x)

                loss = coteaching_loss(logit, logit_2, y, r_t, cutmix=False)

            _, preds = logit.topk(self.topk, 1, True, True)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        n_batches = len(memory_loader)

        return total_loss / n_batches, correct / num_data

    def dividemix(self, epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, warm_up=10, T=0.5,
                  alpha=4):
        net.train()
        net2.eval()  # fix one network and train the other

        num_iter = 0 if labeled_trainloader is None else len(labeled_trainloader)
        use_unlabeled = unlabeled_trainloader is not None

        if num_iter > 0:
            if use_unlabeled:
                unlabeled_train_iter = iter(unlabeled_trainloader)
            for batch_idx, data in enumerate(labeled_trainloader):
                inputs_x = data['image']
                inputs_x2 = data['origin_img']
                labels_x = data['label']
                if len(labels_x) <= 1:
                    continue
                w_x = data['prob'][:, 0]
                if use_unlabeled:
                    try:
                        data = unlabeled_train_iter.next()
                        inputs_u = data['image']
                        inputs_u2 = data['origin_img']
                    except:
                        unlabeled_train_iter = iter(unlabeled_trainloader)
                        data = unlabeled_train_iter.next()
                        inputs_u = data['image']
                        inputs_u2 = data['origin_img']
                    if inputs_u.size(0) <= 1:
                        continue

                batch_size = inputs_x.size(0)

                # Transform label to one-hot
                labels_x = torch.zeros(batch_size, self.num_learning_class).scatter_(1, labels_x.view(-1, 1), 1)
                w_x = w_x.view(-1, 1).type(torch.FloatTensor)

                inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
                if use_unlabeled:
                    inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

                with torch.no_grad():
                    # label co-guessing of unlabeled samples
                    if use_unlabeled:
                        outputs_u11 = net(inputs_u)
                        outputs_u12 = net(inputs_u2)
                        outputs_u21 = net2(inputs_u)
                        outputs_u22 = net2(inputs_u2)

                        pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(
                            outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
                        ptu = pu ** (1 / T)  # temparature sharpening

                        targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                        targets_u = targets_u.detach()

                    # label refinement of labeled samples
                    outputs_x = net(inputs_x)
                    outputs_x2 = net(inputs_x2)

                    px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                    px = w_x * labels_x + (1 - w_x) * px
                    ptx = px ** (1 / T)  # temparature sharpening

                    targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                    targets_x = targets_x.detach()

                    # mixmatch
                l = np.random.beta(alpha, alpha)
                l = max(l, 1 - l)

                if use_unlabeled:
                    all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
                    all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
                else:
                    all_inputs = torch.cat([inputs_x, inputs_x2], dim=0)
                    all_targets = torch.cat([targets_x, targets_x], dim=0)

                idx = torch.randperm(all_inputs.size(0))

                input_a, input_b = all_inputs, all_inputs[idx]
                target_a, target_b = all_targets, all_targets[idx]

                mixed_input = l * input_a + (1 - l) * input_b
                mixed_target = l * target_a + (1 - l) * target_b

                logits = net(mixed_input)
                logits_x = logits[:batch_size * 2]
                if use_unlabeled:
                    logits_u = logits[batch_size * 2:]
                    Lx, Lu, lamb = dividemix_loss(logits_x, mixed_target[:batch_size * 2], logits_u,
                                                  mixed_target[batch_size * 2:],
                                                  epoch + batch_idx / num_iter, warm_up)

                else:
                    Lx, Lu, lamb = dividemix_loss(logits_x, mixed_target[:batch_size * 2], torch.Tensor([[0]]),
                                                  torch.Tensor([[0]]),
                                                  epoch + batch_idx / num_iter, warm_up)

                # regularization
                prior = torch.ones(self.num_learning_class) / self.num_learning_class
                prior = prior.cuda()
                pred_mean = torch.softmax(logits, dim=1).mean(0)
                penalty = torch.sum(prior * torch.log(prior / pred_mean))

                loss = Lx + lamb * Lu + penalty
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _train(
            self, memory_loader, optimizer, criterion
    ):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        self.model.train()
        if self.robust_type == 'DivideMix':
            self.model_2.train()
        for data in memory_loader:
            x = data["image"]
            y = data["label"]

            x = x.to(self.device)
            y = y.to(self.device)

            l, c, d = self.update_model(x, y, criterion, optimizer)
            total_loss += l
            correct += c
            num_data += d

        n_batches = len(memory_loader)

        return total_loss / n_batches, correct / num_data

    def allocate_batch_size(self, n_old_class, n_new_class):
        new_batch_size = int(
            self.batch_size * n_new_class / (n_old_class + n_new_class)
        )
        old_batch_size = self.batch_size - new_batch_size
        return new_batch_size, old_batch_size

    def before_task(self, datalist, init_model=False, init_opt=True):
        logger.info("Apply before_task")
        incoming_df = pd.DataFrame(datalist)
        incoming_classes = incoming_df["klass"].unique().tolist()
        self.exposed_classes = list(set(self.learned_classes + incoming_classes))
        self.num_learning_class = max(
            len(self.exposed_classes), self.num_learning_class
        )

        for klass in incoming_classes:
            if klass not in self.label_map:
                if "true_label" in incoming_df:
                    label = incoming_df[incoming_df['klass'] == klass].iloc[0]['true_label']
                else:
                    label = incoming_df[incoming_df['klass'] == klass].iloc[0]['label']
                self.label_map[klass] = label

        in_features = self.model.fc.in_features
        out_features = self.model.fc.out_features
        # To care the case of decreasing head
        new_out_features = max(out_features, self.num_learning_class)
        if init_model:
            # init model parameters in every iteration
            logger.info("Reset model parameters")
            self.model = select_model(self.model_name, self.dataset, new_out_features)
        elif out_features != self.num_learning_class:
            self.model.fc = nn.Linear(in_features, new_out_features)
        else:
            logger.info("Blurry! There is no modification on model fc layer!")

        self.params = {
            n: p for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad
        }  # For regularzation methods
        self.model = self.model.to(self.device)

        if init_opt:
            # reinitialize the optimizer and scheduler
            logger.info("Reset the optimizer and scheduler states")
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=0.9, nesterov=True, weight_decay=1e-4
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=1, T_mult=2, eta_min=self.lr * 0.01
            )

        logger.info(f"Increasing the head of fc {out_features} -> {new_out_features}")

        self.already_mem_update = False

        if self.robust_type in ['CoTeaching', 'DivideMix']:
            in_features = self.model_2.fc.in_features
            out_features = self.model_2.fc.out_features
            # To care the case of decreasing head
            new_out_features = max(out_features, self.num_learning_class)
            if init_model:
                # init model parameters in every iteration
                self.model_2 = select_model(self.model_name, self.dataset, new_out_features)
            elif out_features != self.num_learning_class:
                self.model_2.fc = nn.Linear(in_features, new_out_features)

            self.model_2 = self.model_2.to(self.device)
            if init_opt:
                if self.robust_type == 'DivideMix':
                    self.optimizer_2 = optim.SGD(
                        self.model_2.parameters(), lr=self.lr, momentum=0.9, nesterov=True, weight_decay=1e-4
                    )
                    self.scheduler_2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        self.optimizer, T_0=1, T_mult=2, eta_min=self.lr * 0.01
                    )
                else:
                    self.optimizer.add_param_group({'params': self.model_2.parameters()})

    def online_train(self, cur_iter, batch_size, n_worker, seed):
        dataset = ImageDataset(
            pd.DataFrame(self.streamed_list),
            dataset_path=self.dataset_path,
            transform=self.test_transform,
        )
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=n_worker,
            drop_last=True,
        )
        testset = ImageDataset(
            pd.DataFrame(self.test_list),
            dataset_path=self.dataset_path,
            transform=self.test_transform,
        )
        test_loader = DataLoader(
            testset,
            shuffle=False,
            batch_size=128,
            num_workers=n_worker,
        )

        self.model = self.model.to(self.device)

        self.model.train()
        if self.robust_type in ['CoTeaching', 'DivideMix']:
            self.model_2.to(self.device)
            self.model_2.train()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = 0.001
        if self.robust_type == 'DivideMix':
            for param_group in self.optimizer_2.param_groups:
                param_group["lr"] = 0.001

        coeff = 0.0
        for batch_idx, data in enumerate(dataloader):
            start = time.time()
            x = data["image"]
            y = data["label"]
            if cur_iter > 0:
                # Create Memory Set.
                memset = ImageDataset(
                    pd.DataFrame(self.memory_list),
                    dataset_path=self.dataset_path,
                    transform=self.train_transform,
                )
                idx_list = list(range(len(memset)))
                random.shuffle(idx_list)

                idx = idx_list[:batch_size]
                mx = torch.cat([memset[i]["image"].unsqueeze(0) for i in idx])
                my = torch.tensor([memset[i]["label"] for i in idx])
                x = torch.cat([x, mx])
                y = torch.cat([y, my])

            x = x.to(self.device)
            y = y.to(self.device)
            l, _, _ = self.update_model(x, y, self.criterion, self.optimizer)
            # set coeff that is robust to various noise setup.
            self.coeff = self.origin_coeff / l
            if self.coeff > self.origin_coeff:
                self.coeff = self.origin_coeff

            coeff += self.coeff

            self.online_update_memory(self.streamed_list[batch_idx * batch_size:(batch_idx + 1) * batch_size])
            logger.info(f"[{batch_idx}/{len(dataloader)}] loss: {l}, diversity coeff: {self.coeff}, "
                        f"{1000 * (time.time() - start)} msec")

        eval_dict = self.evaluation(
            model=self.model, test_loader=test_loader, criterion=self.criterion
        )

        writer.add_scalar("OnlineModel_Acc", eval_dict["avg_acc"], cur_iter)
        writer.add_scalar("alpha", coeff / len(dataloader), cur_iter)

        logger.info("Memory statistic")
        memory_df = pd.DataFrame(self.memory_list)
        logger.info(f"\n{memory_df.klass.value_counts(sort=True)}")

    def online_update_memory(self, cur_data_list):
        if self.mem_manage == "RSV":
            self.reservoir_sampling(cur_data_list)
        elif self.mem_manage == "GBS":
            self.greedy_balanced_sampling(cur_data_list)
        elif len(self.memory_list) + len(cur_data_list) <= self.memory_size:
            self.memory_list = self.memory_list + cur_data_list
        elif self.mem_manage == "PuriDivER":
            self.memory_list = self.puridiver_sampling(cur_data_list)
        elif self.mem_manage == "RM":
            self.memory_list = self.rainbow_memory_sampling(cur_data_list + self.memory_list, num_class=self.n_classes)
        else:
            raise NotImplementedError()

    def calculate_loss_and_feature(self, df, get_loss=True, get_feature=True, test_batchsize=256):
        dataset = ImageDataset(
            df, dataset_path=self.dataset_path, transform=self.test_transform
        )
        dataloader = DataLoader(dataset, batch_size=min(test_batchsize, len(dataset)), shuffle=False)

        criterion = nn.CrossEntropyLoss(reduction='none')
        criterion = criterion.to(self.device)
        self.model.eval()

        with torch.no_grad():
            logits = []
            features = []
            labels = []
            for batch_idx, data in enumerate(dataloader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit, feature = self.model(x, get_feature=True)
                logits.append(logit)
                features.append(feature)
                labels.append(y)
            logits = torch.cat(logits, dim=0)
            features = torch.cat(features, dim=0)
            labels = torch.cat(labels, dim=0)

            if get_loss:
                loss = criterion(logits, labels)
                loss = loss.detach().cpu()
                loss = loss.tolist()
                df["loss"] = loss
            if get_feature:
                features = features.detach().cpu()
                features = features.tolist()
                df["feature"] = features
        return df

    def get_cls_cos_score(self, df):
        cls_features = np.array(df["feature"].tolist())

        weights = self.model.fc.block[0].weight.data.detach().cpu()
        clslist = df["label"].unique().tolist()
        assert len(clslist) == 1
        cls = clslist[0]
        relevant_idx = weights[cls, :] > torch.mean(weights, dim=0)

        cls_features = cls_features[:, relevant_idx]

        sim_matrix = cosine_similarity(cls_features)
        sim_score = sim_matrix.mean(axis=1)

        df['similarity'] = sim_score
        return df

    def puridiver_sampling(self, cur_data_list):
        cand_df = pd.DataFrame(self.memory_list + cur_data_list)
        cand_df = self.calculate_loss_and_feature(cand_df)
        while len(cand_df) > self.memory_size:
            cls_cnt = cand_df["label"].value_counts()
            cls_to_drop = cls_cnt[cls_cnt == cls_cnt.max()].sample().index[0]  # argmax w/ random tie-breaking
            cls_cand_df = cand_df[cand_df["label"] == cls_to_drop].copy()
            cls_cand_df = self.get_cls_cos_score(cls_cand_df)
            cls_loss = cls_cand_df["loss"].to_numpy()
            cls_loss = (cls_loss - cls_loss.mean()) / cls_loss.std()
            sim_score = cls_cand_df["similarity"].to_numpy()
            sim_score = (sim_score - sim_score.mean()) / sim_score.std()
            score = (1 - self.coeff) * cls_loss + self.coeff * sim_score
            drop_idx = np.argmax(score)
            cand_df = cand_df.drop(cls_cand_df.index.values[drop_idx])

        cand_df = cand_df.drop('feature', 1)
        return cand_df.to_dict(orient='records')

    def greedy_balanced_sampling(self, cur_data_list):
        for sample in cur_data_list:
            if sample["label"] in self.seen_per_cls:
                self.seen_per_cls[sample["label"]] += 1
            else:
                self.seen_per_cls[sample["label"]] = 1
            if len(self.memory_list) < self.memory_size:
                self.memory_list += [sample]
            else:
                cand_df = pd.DataFrame(self.memory_list + [sample])
                cls_cnt = cand_df["label"].value_counts()
                if np.random.rand() < cls_cnt[sample["label"]] / self.seen_per_cls[sample["label"]]:
                    if cls_cnt[sample["label"]] == cls_cnt.max():
                        cls_to_drop = sample["label"]
                    else:
                        cls_to_drop = cls_cnt[cls_cnt == cls_cnt.max()].sample().index[
                            0]  # argmax w/ random tie-breaking
                    cand_df = cand_df.drop(cand_df[cand_df["label"] == cls_to_drop].sample().index)
                    self.memory_list = cand_df.to_dict(orient='records')

    def _compute_uncert(self, infer_list, infer_transform, uncert_name):
        batch_size = 32
        infer_df = pd.DataFrame(infer_list)
        infer_dataset = ImageDataset(
            infer_df, dataset_path=self.dataset_path, transform=infer_transform
        )
        infer_loader = DataLoader(
            infer_dataset, shuffle=False, batch_size=batch_size, num_workers=2
        )

        self.model.eval()
        with torch.no_grad():
            for n_batch, data in enumerate(infer_loader):
                x = data["image"]
                x = x.to(self.device)
                logit = self.model(x)
                logit = logit.detach().cpu()

                for i, cert_value in enumerate(logit):
                    sample = infer_list[batch_size * n_batch + i]
                    sample[uncert_name] = 1 - cert_value

    def montecarlo(self, candidates):
        logger.info(f"Compute uncertainty!")

        transform_cands = [
            Cutout(size=8),
            Cutout(size=16),
            Cutout(size=24),
            Cutout(size=32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.RandomRotation(90),
            Invert(),
            Solarize(v=128),
            Solarize(v=64),
            Solarize(v=32),
        ]

        n_transforms = len(transform_cands)

        for idx, tr in enumerate(transform_cands):
            _tr = transforms.Compose([tr] + self.test_transform.transforms)
            self._compute_uncert(candidates, _tr, uncert_name=f"uncert_{str(idx)}")

        for sample in candidates:
            self.variance_ratio(sample, n_transforms)

    def variance_ratio(self, sample, cand_length):
        vote_counter = torch.zeros(sample["uncert_0"].size(0))
        for i in range(cand_length):
            top_class = int(torch.argmin(sample[f"uncert_{i}"]))  # uncert argmin.
            vote_counter[top_class] += 1
        assert vote_counter.sum() == cand_length
        sample["uncertainty"] = (1 - vote_counter.max() / cand_length).item()

    def rainbow_memory_sampling(self, samples, num_class):
        self.montecarlo(samples)

        sample_df = pd.DataFrame(samples)
        mem_per_cls = self.memory_size // num_class

        ret = []
        for i in range(num_class):
            cls_df = sample_df[sample_df["label"] == i]
            if len(cls_df) <= mem_per_cls:
                ret += cls_df.to_dict(orient="records")
            else:
                jump_idx = len(cls_df) // mem_per_cls
                uncertain_samples = cls_df.sort_values(by="uncertainty")[::jump_idx]
                ret += uncertain_samples[:mem_per_cls].to_dict(orient="records")

        num_rest_slots = self.memory_size - len(ret)
        if num_rest_slots > 0:
            logger.warning("Fill the unused slots by breaking the equilibrium.")
            ret += (
                sample_df[~sample_df.file_name.isin(pd.DataFrame(ret).file_name)]
                    .sample(n=num_rest_slots)
                    .to_dict(orient="records")
            )

        num_dups = pd.DataFrame(ret).file_name.duplicated().sum()
        if num_dups > 0:
            logger.warning(f"Duplicated samples in memory: {num_dups}")

        return ret

    def after_task(self):
        logger.info("Apply after_task")
        self.learned_classes = self.exposed_classes
        self.num_learned_class = self.num_learning_class

    def set_current_dataset(self, train_datalist, test_datalist):
        np.random.shuffle(train_datalist)
        self.prev_streamed_list = self.streamed_list
        self.streamed_list = train_datalist
        self.test_list = test_datalist

    def get_dataloader(self, batch_size, n_worker, train_list, test_list):
        # Loader
        train_loader = None
        test_loader = None
        if train_list is not None and len(train_list) > 0:
            train_dataset = ImageDataset(
                pd.DataFrame(train_list),
                dataset_path=self.dataset_path,
                transform=self.train_transform,
            )

            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=n_worker,
                drop_last=True,
            )

        if test_list is not None:
            test_dataset = ImageDataset(
                pd.DataFrame(test_list),
                dataset_path=self.dataset_path,
                transform=self.test_transform,
            )
            test_loader = DataLoader(
                test_dataset, shuffle=False, batch_size=batch_size, num_workers=n_worker
            )

        return train_loader, test_loader

    def evaluation(self, model, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit = model(x)

                loss = criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)

                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

        return ret

    def reservoir_sampling(self, samples):
        for sample in samples:
            if len(self.memory_list) < self.memory_size:
                self.memory_list += [sample]
            else:
                j = np.random.randint(0, self.seen)
                if j < self.memory_size:
                    self.memory_list[j] = sample
            self.seen += 1

    def _interpret_pred(self, y, pred):
        # xlabel is batch
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects
