import torch

from torch.optim import Optimizer
from lib.core.rdfgrad import rdfgrad


class QuaternionSGD(Optimizer):

    def __init__(self, params, lr=0.1, data_thred=0.05, tolerance_change=1e-5):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)

        super(QuaternionSGD, self).__init__(params, defaults)

        self.params = None

        self.data_thred = data_thred
        self.tolerance_change = tolerance_change

    def __setstate__(self, state):
        super(QuaternionSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        prev_loss = 10000
        for k in range(10000):
            loss = closure()
            for group in self.param_groups:
                for i, param in enumerate(group['params']):
                    if param.grad is None:
                        continue
                    d_p = param.grad.data
                    if i == 0:
                        effective_lr = loss
                        param.data = rdfgrad(
                                egrad=d_p,
                                q=param.data,
                                dist=effective_lr,
                                step_size=1.0,
                                norm=False
                        )
                    else:
                        effective_lr = compute_lr(group['lr'], d_p, loss)
                        param.data -= effective_lr * d_p

            if abs(loss - prev_loss) < self.tolerance_change:
                break
            prev_loss = loss
        return loss


class HybridSGD(Optimizer):

    def __init__(self, params, lr=0.1, data_thred=0.05, tolerance_change=1e-5):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)

        super(HybridSGD, self).__init__(params, defaults)

        self.params = None
        self.data_thred = data_thred
        self.tolerance_change = tolerance_change

    def __setstate__(self, state):
        super(HybridSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None, loss=None, data_err=None, prior_err=None):
        if closure is not None:
            loss, data_err, prior_err = closure()
        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                if param.grad is None:
                    continue
                d_p = param.grad.data
                if i == 0:
                    param.data = rdfgrad(
                        egrad=d_p,
                        q=param.data,
                        dist=group['lr'],
                        step_size=1.0,
                        norm=False
                    )
                else:
                    param.data -= group['lr'] * d_p
        return loss, data_err, prior_err


def compute_lr(lr, v, loss):
    if loss is None:
        return lr
    else:
        tmp = loss / torch.norm(v) ** 2
        return min(lr, tmp)