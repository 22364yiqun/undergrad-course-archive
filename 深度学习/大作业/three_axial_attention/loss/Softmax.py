import torch.nn as nn
import torch.nn.functional as f


class Softmax(nn.Module):
    def __init__(self, **options):
        super(Softmax, self).__init__()

    def forward(self, x,y,labels=None):
        logits = y
        if labels is None:
            return logits, 0, 0
        loss = f.cross_entropy(y, labels)
        return logits, loss, loss