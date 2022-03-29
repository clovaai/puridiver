"""
PuriDivER
Copyright 2022-present NAVER Corp.
GPLv3
"""
import os

from configuration.config import Args, build_command_line_args

# Default args
args = Args()

args.robust_type = "PuriDivER" # none, SELFIE, CoTeaching, DivideMix, PuriDivER
args.mem_manage = "PuriDivER" # RSV, GBS, RM, PuriDivER
args.dataset = "cifar10" # cifar10, cifar100, WebVision-V1-2, Food-101N
args.exp_name = "blurry10" # WebVision-V1-2, Food-101N: blurry10, others: blurry10_symN20, blurry10_symN40, blurry10_symN60, blurry10_asymN20, blurry10_asymN40
args.dataset_path = f"dataset/{args.dataset}"

if args.dataset == "cifar10":
    args.n_cls_a_task = 2
    args.n_init_cls = 10
    args.memory_size = 500
    args.n_tasks = 5
    args.model_name = "resnet18"
    args.warmup = 10 # change to 10 (1 for test)
    args.n_epoch = 255 # The epoch is 256 because of online learning (one-pass).

elif args.dataset == "cifar100":
    args.n_cls_a_task = 20
    args.n_init_cls = 100
    args.memory_size = 2000
    args.n_tasks = 5
    args.model_name = "resnet32"
    args.warmup = 30
    args.n_epoch = 255 # The epoch is 256 because of online learning (one-pass).

elif args.dataset == "WebVision-V1-2":
    args.n_cls_a_task = 5
    args.n_init_cls = 50
    args.memory_size = 1000
    args.n_tasks = 10
    args.model_name = "resnet34"
    args.warmup = 10
    args.n_epoch = 127 # The epoch is 128 because of online learning (one-pass).

elif args.dataset == "Food-101N":
    args.n_cls_a_task = 20
    args.n_init_cls = 101
    args.memory_size = 2000
    args.n_tasks = 5
    args.model_name = "resnet34"
    args.warmup = 10
    args.n_epoch = 127 # The epoch is 128 because of online learning (one-pass).

else:
    raise NotImplementedError()

args.batchsize = 16
args.lr = 0.05

args.rnd_seed = [1, 2, 3] # 1 2 3
args.log_path = "results"

args.transforms = ["autoaug", "cutmix"]
args.n_worker = 2

args.init_model = False
args.init_opt = True
args.topk = 1

args.debug = False

args.coeff = 0.5

args_text = build_command_line_args(args)
command = f'python main.py {args_text}'

print(f"[Execute] {command}")
os.system(command)

