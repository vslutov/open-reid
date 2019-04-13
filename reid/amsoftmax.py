import numpy as np
import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _assert_no_grad(tensor):
   assert not tensor.requires_grad, \
       "nn criterions don't compute the gradient w.r.t. targets - please " \
       "mark these tensors as not requiring gradients"

class CosDistance(torch.nn.Module):
    def __init__(self, input_features, output_features, weight=None):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        if weight is None:
            weight = torch.Tensor(input_features, output_features)
            torch.nn.init.xavier_uniform_(weight)

        self.weight = nn.Parameter(weight)

    def forward(self, X):
        W_norm = self.weight.norm(2, 0) # (self.weight ** 2).sum(dim=0).sqrt()
        X_norm = X.norm(2, 1) # (X ** 2).sum(dim=1).sqrt()
        cos = torch.mm(X, self.weight) / W_norm[np.newaxis, :] / X_norm[:, np.newaxis]

        return cos


def amsoftmax(cos, y, s=10, m=0.35, weight=None):
    n, c = cos.size()
    arange = torch.arange(n, dtype=torch.int64)

    s = torch.tensor(s, dtype=torch.float32).to(device)
    m = torch.tensor(m, dtype=torch.float32).to(device)

    logits = s * cos
    logits[arange, y] = s * (cos[arange, y] - m)
    return torch.nn.functional.cross_entropy(logits, y, weight=weight, reduction='mean')

class AMSoftmax(torch.nn.Module):
    def __init__(self, s=30, m=0.35, weight=None):
        super().__init__()
        self.s = s
        self.m = m
        self.register_buffer('weight', weight)

    def forward(self, cos, y):
        _assert_no_grad(y)
        return amsoftmax(cos=cos, y=y, s=self.s, m=self.m, weight=self.weight)
