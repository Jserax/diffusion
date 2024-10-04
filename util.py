import math
from typing import Optional

import torch


class LRCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 1e-3,
        min_lr: float = 1e-4,
        gamma: float = 1.0,
        last_epoch: int = -1,
        verbose: str = "deprecated",
    ) -> None:
        assert warmup_steps < cycle_steps
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.gamma = gamma
        self.verbose = verbose

        self.cur_cycle_steps = self.cycle_steps
        self.last_cycle_steps = -1
        super().__init__(optimizer, last_epoch, verbose)
        self._init_lr()

    def _init_lr(self) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr

    def get_lr(self) -> list[float]:
        if self.last_cycle_steps < self.warmup_steps:
            return [
                (self.max_lr - self.min_lr) / self.warmup_steps * self.last_cycle_steps
                + self.min_lr
                for _ in self.optimizer.param_groups
            ]
        else:
            decay = (self.last_cycle_steps - self.warmup_steps) / (
                self.cur_cycle_steps - self.warmup_steps
            )

            coeff = 0.5 * (1.0 + math.cos(math.pi * min(decay, 1.0)))
            return [
                self.min_lr + coeff * (self.max_lr - self.min_lr)
                for _ in self.optimizer.param_groups
            ]

    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.last_cycle_steps >= self.cur_cycle_steps:
            self.max_lr = (self.max_lr - self.min_lr) * self.gamma + self.min_lr
            self.cur_cycle_steps = int(self.cur_cycle_steps * self.cycle_mult)
            self.last_cycle_steps = 1
        else:
            self.last_cycle_steps += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


def normalize(
    images: torch.Tensor,
) -> torch.Tensor:
    return 2.0 * images - 1.0


def denormalize(
    images: torch.Tensor,
) -> torch.Tensor:
    return (images / 2 + 0.5).clamp(0, 1)
