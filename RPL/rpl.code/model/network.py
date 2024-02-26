import torch.nn as nn

from model.wide_network import DeepWV3Plus


class Network(nn.Module):
    def __init__(self, num_classes, wide=False):
        super(Network, self).__init__()
        # if wide:
        self.branch1 = DeepWV3Plus(num_classes)

    def forward(self, data):
        return self.branch1(data)
