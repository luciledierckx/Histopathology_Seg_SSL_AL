import numpy as np
import torch
from torch.utils.data import DataLoader

from prepare_data.transformations import TransformUDA

from .strategy import QueryStrategy, Strategy


class TransformConsistency(TransformUDA):
    def __init__(self, size, channels, amount: int):
        super().__init__(size, channels)
        self.amount = amount

    def __call__(self, x):
        base = self.base(x)

        images = [base] + [self.color_weak(base) for _ in range(self.amount - 1)]
        return [self.tensorize(img) for img in images]


class empty_col:
    def __getitem__(self, *a):
        return torch.empty(0)


def _avg_consistency(strategy: Strategy, X: torch.Tensor, _: torch.Tensor):

    # trade faster processing for more accurate consistency computation
    augment = 50

    transform = TransformConsistency(
        size=strategy.args.img_size,
        channels=strategy.args.channels,
        amount=augment,
    )

    loader_te = DataLoader(
        strategy.handler(
            X,
            empty_col(),  # Y can be discarded
            transform=transform,
            weak_strong=True,
        ),
        shuffle=False,
        pin_memory=True,
        batch_size=1,
        num_workers=1,
    )

    consistency_lst = torch.zeros(len(X))

    strategy.clf.eval()
    with torch.no_grad():
        for x_all, _, idxs in loader_te:
            # reshape possible without data manipulation ?
            input_stack = torch.stack(x_all).to(strategy.device)
            # input_stack: Tensor.float[k, b, ic, h, w]
            k, b, ic, h, w = input_stack.shape
            input_all = input_stack.view(k * b, ic, h, w)
            # input_all: Tensor.float[k*b, ic, h, w]

            pred_all = strategy.clf(input_all)
            # pred_all: Tensor.float[k*b, ic, h, w]

            pred_stack = pred_all.view(k, b, -1, h, w)
            # pred_stack: Tensor.float[k, b, oc, h, w]

            # consistency measure
            consistency = torch.var(pred_stack, dim=0)  # variance across augmentations
            # consistency: Tensor.float[b, oc, h, w]
            consistency = torch.sum(consistency, dim=1)  # sum across classes
            # consistency: Tensor.float[b, h, w]
            consistency = torch.mean(
                torch.mean(consistency, dim=-1), dim=-1  # avg across pixels
            )
            # consistency: Tensor.float[b]

            consistency_lst[idxs] = consistency.cpu().data

    return consistency_lst


class ConsistencySamplingSeg(QueryStrategy):
    def query(self, strategy: Strategy, n: int):
        idxs_unlabeled = np.arange(strategy.n_pool)[~strategy.idxs_lb]
        avg_consistency = _avg_consistency(strategy, strategy.X[idxs_unlabeled], None)
        idxs = avg_consistency.argsort()

        return idxs_unlabeled[idxs[-n:]]
