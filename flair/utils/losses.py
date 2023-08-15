"""
Losses for segmentation tasks.
"""

import torch


class BinaryDice(torch.nn.Module):
    def __init__(self, reduction='mean', activation=None):
        super(BinaryDice, self).__init__()
        self.reduction = reduction
        self.smooth = 1
        if activation is None:
            self.activation = torch.nn.Identity()
        else:
            self.activation = activation

    def forward(self, predict, target):
        predict = self.activation(predict)

        num = torch.sum(predict*target, (2, 3))
        den = torch.sum(predict, (2, 3)) + torch.sum(target, (2, 3))

        dice_score = ((2 * num) + self.smooth) / (den + self.smooth)
        dice_loss = 1 - dice_score

        # Average per batch
        dice_loss = dice_loss.mean()

        return dice_loss


class BinaryDiceCE(torch.nn.Module):
    def __init__(self, reduction='mean', activation=None, alpha_dsc=0, alpha_ce=1.):
        super(BinaryDiceCE, self).__init__()
        self.reduction = reduction
        self.smooth = 1
        self.alpha_dsc = alpha_dsc
        self.alpha_ce = alpha_ce
        if activation is None:
            self.activation = torch.nn.Identity()
        else:
            self.activation = activation

    def forward(self, logits, target):
        predict = self.activation(logits)

        ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target.to(torch.float))

        num = torch.sum(predict*target, (2, 3))
        den = torch.sum(predict, (2, 3)) + torch.sum(target, (2, 3))

        dice_score = ((2 * num) + self.smooth) / (den + self.smooth)
        dice_loss = 1 - dice_score

        # Average per batch
        dice_loss = dice_loss.mean()

        loss = ce + dice_loss

        return loss