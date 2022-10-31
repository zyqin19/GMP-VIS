import torch.nn as nn
import torch

class Loss(nn.Module):
    def forward(self, x, gt):

        x = x.reshape(-1)
        gt = gt.reshape(-1)

        pa = len(gt) * len(gt) / (len(gt) - gt.sum()) / 2 / gt.sum()
        weight_0 = pa * gt.sum() / len(gt)
        weight_1 = pa * (len(gt) - gt.sum()) / len(gt)
        weight = torch.zeros(len(gt))
        for i in range(len(gt)):
            if gt[i] == 0:
                weight[i] = weight_0
            elif gt[i] == 1:
                weight[i] = weight_1
            else:
                raise RuntimeError('loss weight error')
        # print(weight)
        # weight = torch.abs(gt-0.1).detach()
        loss = nn.BCELoss(weight=weight.cuda())

        # loss = nn.BCELoss(reduction='sum')
        # print(gt.sum())
        out = loss(x, gt)
        return out
