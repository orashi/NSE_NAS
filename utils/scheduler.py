import torch
from bisect import bisect_right
import math


class _LRScheduler(object):
    def __init__(self, optimizer, last_iter=-1):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    group.setdefault('initial_lr', self.init_lr)
                    print(f"param 'initial_lr' is not specified in param_groups[{i}]",
                          " when resuming an optimizer, so we assume all starts from init lr (warm or base)")

        self.init_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_iter = last_iter

    def _get_new_lr(self):
        raise NotImplementedError

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = self.last_iter + 1
        self.last_iter = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_new_lr()):
            param_group['lr'] = lr


class _WarmUpLRScheduler(_LRScheduler):

    def __init__(self, optimizer, warmup_lr, base_lr, warmup_steps, last_iter=-1):
        self.warmup_lr = warmup_lr
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        if self.warmup_steps > 0:
            self.init_lr = warmup_lr
        else:
            self.init_lr = base_lr
        super(_WarmUpLRScheduler, self).__init__(optimizer, last_iter)

    def _get_base_lr(self):
        if self.warmup_steps > 0 and self.last_iter < self.warmup_steps:
            # first compute relative scale for self.warmup_lr, then multiply to warmup_lr
            scale = ((self.last_iter / self.warmup_steps) * (
                        self.base_lr - self.warmup_lr) + self.warmup_lr) / self.warmup_lr
            return [scale * warmup_lr for warmup_lr in self.init_lrs]
        else:
            return None


class StepLRScheduler(_WarmUpLRScheduler):
    def __init__(self, optimizer, milestones, lr_mults, warmup_lr, base_lr, warmup_steps, last_iter=-1):
        super(StepLRScheduler, self).__init__(optimizer, warmup_lr, base_lr, warmup_steps, last_iter)

        assert len(milestones) == len(lr_mults), "{} vs {}".format(milestones, lr_mults)
        for x in milestones:
            assert isinstance(x, int)
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.lr_mults = [1.0]
        for x in lr_mults:
            self.lr_mults.append(self.lr_mults[-1] * x)

    def _get_new_lr(self):
        base_lr = self._get_base_lr()
        if base_lr is not None:
            return base_lr

        pos = bisect_right(self.milestones, self.last_iter)
        scale = self.base_lr * self.lr_mults[pos] / self.init_lr
        return [warmup_lr * scale for warmup_lr in self.init_lrs]


class CosineLRScheduler(_WarmUpLRScheduler):
    def __init__(self, optimizer, T_max, base_lr, eta_min, warmup_steps, warmup_lr, last_iter=-1):
        super(CosineLRScheduler, self).__init__(optimizer, warmup_lr, base_lr, warmup_steps, last_iter)
        self.T_max = T_max
        self.eta_min = eta_min

    def _get_new_lr(self):
        base_lr = self._get_base_lr()
        if base_lr is not None:
            return base_lr

        step_ratio = (self.last_iter - self.warmup_steps) / (self.T_max - self.warmup_steps)
        target_lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * step_ratio)) / 2
        scale = target_lr / self.init_lr
        return [scale * warmup_lr for warmup_lr in self.init_lrs]
