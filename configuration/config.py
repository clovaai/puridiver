"""
PuriDivER
Copyright 2022-present NAVER Corp.
GPLv3
"""
from typing import List
from typing_extensions import Literal
import argparse


class Args:
    # memory management
    # mem_manage: Literal["default", "random", "balanced", "reservoir", "uncertainty", "prototype", "balanced_loss",
    #                     "avg_loss", "adaptive_balanced_loss"] = "default"
    mem_manage: Literal["none", "RSV", "GBS", "RM", "PuriDivER"] = "none"

    # Robust Learning Type when training
    robust_type: Literal["none", "const_reg", "SELFIE", "soft_relabel", "PuriDivER", "contrast"] = "none"

    # Dataset name
    dataset: Literal["cifar10", "cifar100", "WebVision-V1-2", "Food-101N"] = "cifar10"
    dataset_path: str = "../data/cifar10_jh/train"

    # The number of tasks
    n_tasks: int = 5

    # The number of class of each task
    n_cls_a_task: int = 2

    # The number of classes of initial task
    n_init_cls: int = 2

    # Warm-up Epochs
    warmup: int = 1

    # Random seed number
    rnd_seed: List[int] = [1]

    # Episodic memory size
    memory_size: int = 500

    # The path logs are saved. Only for local-machine
    log_path: str = "results"

    # Backbone model name
    model_name: Literal["resnet18", "resnet32", "resnet34"] = "resnet18"

    # Batch size
    batchsize: int = 512

    # Epochs
    n_epoch: int = 2

    # Number of workers
    n_worker: int = 0

    # Learning rate
    lr: float = 0.1

    # Initialize model parameters for every iterations
    init_model: bool = True

    # Initialize optimizer states for every iterations
    init_opt: bool = True

    # Set k when we want to set topk accuracy
    topk: int = 1

    # Additional train transforms
    transforms: List[Literal["cutmix", "cutout", "randaug", "autoaug"]] = []

    # Experiment name
    exp_name: Literal[
       "blurry10_symN20", "blurry10_symN40", "blurry10_symN60", "blurry10_asymN20", "blurry10_asymN40"] = "blurry10_symN20"

    # Turn on Debug mode
    debug: bool = False

    # Coefficiency
    coeff: float = 1.0

    # only for CoTeaching.
    noise_rate: float = 0.0


def get_properties(cls):
    return [prop for prop in dir(cls) if not prop.startswith("__")]


def build_command_line_args(args: Args):
    """Un-parse args to use in command line"""
    assert isinstance(args, Args), f"{args} must be an instance of Args"

    params = []
    for name in get_properties(args):
        value = getattr(args, name)
        if type(value) == list:
            if type(value[0]) == int:
                value = [str(i) for i in value]
            params.append(f'--{name} {" ".join(value)}')
        elif type(value) == bool:
            params.append(f'--{name} {value}')
        elif type(value) == str:
            params.append(f'--{name} {value}')
        else:
            params.append(f'--{name} {value}')
    cmd = " \\\n".join(params)
    return cmd


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def parse_args(arg_class=Args) -> argparse.Namespace:
    """Returns args from command line arguments"""
    parser = argparse.ArgumentParser()
    for name in get_properties(arg_class):
        value = getattr(arg_class, name)
        if type(value) == bool:
            # --option true | --option false
            parser.add_argument(f"--{name}", type=str_to_bool, default=value)
        elif type(value) == list:
            parser.add_argument(f"--{name}", nargs="*", default=value)
        else:
            parser.add_argument(f"--{name}", type=type(value), default=value)
    args = parser.parse_args()
    return args


def get_args_text(args) -> str:
    """for printing args"""
    params = []
    for name in get_properties(args):
        value = getattr(args, name)
        params.append(f"{name}: {value}")
    return "\n".join(params)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    print(build_command_line_args(Args()))
