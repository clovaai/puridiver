"""
PuriDivER
Copyright 2022-present NAVER Corp.
GPLv3
"""
import logging.config
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from configuration import config
from configuration.config import get_args_text
from methods.PuriDivER import PuriDivER
from utils.augment import Cutout, select_autoaugment
from utils.data_loader import get_train_datalist, get_test_datalist, get_statistics

writer = SummaryWriter("tensorboard")


def main():
    args = config.parse_args()

    def run_seed(s):
        s = int(s)
        # setup logger
        logging.config.fileConfig("./configuration/logging.conf")
        logger = logging.getLogger()
        log_save_path = f"{args.dataset}/{args.exp_name}_{args.mem_manage}_{args.robust_type}_msz{args.memory_size}_rnd{s}"
        os.makedirs(f"logs/{args.dataset}", exist_ok=True)
        fileHandler = logging.FileHandler("logs/{}.log".format(log_save_path), mode="w")
        formatter = logging.Formatter("[%(levelname)s] %(filename)s:%(lineno)d > %(message)s")
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        logger.info("###############################")
        logger.info(f"##### Random Seed {s} Start #####")
        logger.info("###############################")

        # print args
        logger.info(get_args_text(args))

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info(f"Set the device ({device})")

        torch.manual_seed(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(s)
        random.seed(s)

        # Transform Definition
        mean, std, n_classes, inp_size, _ = get_statistics(dataset=args.dataset)
        train_transform = []
        if 'cutout' in args.transforms:
            train_transform.append(Cutout(size=16))
        if 'autoaug' in args.transforms:
            train_transform.append(select_autoaugment())

        train_transform = transforms.Compose(
            [
                transforms.Resize((inp_size, inp_size)),
                transforms.RandomCrop(inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                *train_transform,
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        logger.info(f"Using train-transforms {train_transform}")
        weak_transform = transforms.Compose(
            [
                transforms.Resize((inp_size, inp_size)),
                transforms.RandomCrop(inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        logger.info(f"[1] Select a CIL method ({args.mem_manage}/{args.robust_type})")
        criterion = nn.CrossEntropyLoss(reduction="mean")
        kwargs = vars(args)
        method = PuriDivER(criterion=criterion,
                           device=device,
                           train_transform=train_transform,
                           test_transform=test_transform,
                           n_classes=n_classes,
                           additional_trans=weak_transform,
                           **kwargs)

        logger.info(f"[2] Incrementally training {args.n_tasks} tasks")
        task_records = defaultdict(list)

        for cur_iter in range(args.n_tasks):

            logger.info("#" * 50)
            logger.info(f"# Task {cur_iter} iteration")
            logger.info("#" * 50)
            logger.info("[2-1] Prepare a datalist for the current task")

            task_acc = 0.0
            eval_dict = dict()

            # get datalist
            cur_train_datalist = get_train_datalist(args, cur_iter, seed=s)
            cur_test_datalist = get_test_datalist(args, args.exp_name, cur_iter, seed=s)

            # Reduce datalist in Debug mode
            if args.debug:
                cur_train_datalist = cur_train_datalist[:200]
                print("=====current train datalist=====")
                for elem in cur_train_datalist:
                    print(elem)
                cur_test_datalist = cur_test_datalist[:200]

            logger.info("[2-2] Set environment for the current task")
            method.set_current_dataset(cur_train_datalist, cur_test_datalist)
            # Increment known class for current task iteration.
            method.before_task(cur_train_datalist, args.init_model, args.init_opt)

            # The way to handle streamed samles
            logger.info(f"[2-3] Start to online train")

            # Online Train
            method.online_train(cur_iter=cur_iter, batch_size=args.batchsize, n_worker=args.n_worker, seed=s)

            # No streamed training data, train with only memory_list
            method.set_current_dataset([], cur_test_datalist)

            logger.info("Train over memory")
            task_acc, eval_dict = method.train(
                cur_iter=cur_iter,
                n_epoch=args.n_epoch,
                batch_size=args.batchsize,
                n_worker=args.n_worker,
            )


            logger.info("[2-4] Update the information for the current task")
            method.after_task()
            task_records["task_acc"].append(task_acc)
            # task_records['cls_acc'][k][j] = break down j-class accuracy from 'task_acc'
            task_records["cls_acc"].append(eval_dict["cls_acc"])

            npy_save_path = f"{args.log_path}/{log_save_path}.npy"
            os.makedirs(os.path.dirname(npy_save_path), exist_ok=True)
            np.save(npy_save_path, task_records["task_acc"])

            logger.info("[2-5] Report task result")
            df = pd.DataFrame(method.memory_list)
            print(df.label.value_counts())
            if "true_label" in df.columns:
                n_clean = len(df[df["true_label"] == df["label"]])
            else:
                n_clean = len(df)
            logger.info("n_clean: {}\t memory_size: {}".format(n_clean, method.memory_size))

            writer.add_scalar("Metrics/TaskAcc", task_acc, cur_iter)

            report_dict = dict()
            report_dict["Metrics__TaskAcc"] = task_acc
            report_dict["Metrics__MemoryCleanRatio"] = n_clean / method.memory_size

            for key in report_dict.keys():
                writer.add_scalar(key, report_dict[key], cur_iter)
            logger.info(report_dict)

        # Accuracy (A)
        A_avg = np.mean(task_records["task_acc"])
        A_last = task_records["task_acc"][args.n_tasks - 1]
        cil_metrics = {
            'Metrics__A_last': A_last,
            'Metrics__A_avg': A_avg
        }

        return cil_metrics

    results = []
    for s in args.rnd_seed:
        results.append(run_seed(s))

    for s, result in enumerate(results):
        for key in result.keys():
            writer.add_scalar(key, result[key], s)


if __name__ == "__main__":
    main()
