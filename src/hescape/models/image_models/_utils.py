from __future__ import annotations

import math
from collections.abc import Mapping
from functools import partial
from types import MappingProxyType
from typing import Any

import torch
import torch.nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_scheduler(cfg: Any, optimizer: Any):
    if cfg.model.scheduler.name == "cosine":
        lr_lambda = partial(
            _get_cosine_schedule_with_warmup_lr_lambda,
            num_warmup_steps=cfg.model.scheduler.num_warmup_steps,
            num_training_steps=cfg.lightning.trainer.num_training_steps,
            num_cycles=cfg.model.scheduler.num_cycles,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif cfg.model.scheduler.name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    else:
        scheduler = None

    return scheduler


def freeze_batch_norm_2d(
    module: nn.Module, module_match: Mapping[str, Any] = MappingProxyType({}), name: str | None = None
) -> nn.Module:
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`.

    If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762

    Parameters
    ----------
        module
            Any PyTorch module.
        module_match
            Dictionary of full module names to freeze (all if empty)
        name
            Full module name (prefix)

    Returns
    -------
        Resulting module
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = ".".join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res
