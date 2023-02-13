import torch
import torch.nn as nn
import torch.nn.functional as F
from module.mlp import MLP


class MLPDecoder(nn.Module):
    def __init__(self, hidden_dims, n_tag, mlp_drop):
        super(MLPDecoder, self).__init__()
        # MLP Layer
        self.mlp = MLP(hidden_dims, n_tag, mlp_drop)

    def forward(self, x, truth):
        hid = self.mlp(x)
        if self.training:
            return F.cross_entropy(hid, truth)
        else:
            pred = torch.argmax(hid, 1)
            return (pred == truth).sum().item(), len(pred)
