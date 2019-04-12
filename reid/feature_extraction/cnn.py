from __future__ import absolute_import
from collections import OrderedDict

import torch
from torch.autograd import Variable

from ..utils import to_torch

def normalize(output):
    return output / output.norm(dim=1).reshape(-1, 1)

def extract_cnn_feature(model, inputs, modules=None, normalize_features=False):
    model.eval()
    inputs = to_torch(inputs).cuda()
    with torch.no_grad():
        if modules is None:
            outputs = model(inputs)
            if normalize_features:
                outputs = normalize(outputs)
            outputs = outputs.cpu()
            return outputs
        # Register forward hook for each module
        outputs = OrderedDict()
        handles = []
        for m in modules:
            outputs[id(m)] = None
            def func(m, i, o):
                if normalize_features:
                    o = normalize(o)
                outputs[id(m)] = o.cpu()
            handles.append(m.register_forward_hook(func))
        model(inputs)
        for h in handles:
            h.remove()
        return list(outputs.values())
