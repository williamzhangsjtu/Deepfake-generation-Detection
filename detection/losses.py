import torch.nn as nn


class DoubleLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super(DoubleLoss, self).__init__()
        self.classfy_loss = nn.CrossEntropyLoss()
        self.reconstruct_loss = nn.MSELoss()
        self.lambd = lambd

    def forward(self, input, _input, target):
        B = input[0].shape[0]
        return self.lambd * self.classfy_loss(input[0], target) + \
            (1 - self.lambd) *self.reconstruct_loss(input[1].view(B, -1), _input.view(B, -1))

CrossEntropyLoss = nn.CrossEntropyLoss