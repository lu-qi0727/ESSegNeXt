import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS


class ObjectLoss(nn.Module):
    def __init__(self, max_object=50):
        super().__init__()
        self.max_object = max_object

    def forward(self, pred, gt):
        num_object = int(torch.max(gt)) + 1
        num_object = min(num_object, self.max_object)
        total_object_loss = 0

        for object_index in range(1, num_object):
            mask = torch.where(gt == object_index, 1, 0).unsqueeze(1).to('cuda')
            num_point = mask.sum(2).sum(2).unsqueeze(2).unsqueeze(2).to('cuda')
            avg_pool = mask / (num_point + 1)

            object_feature = pred.mul(avg_pool)

            avg_feature = object_feature.sum(2).sum(2).unsqueeze(2).unsqueeze(2).repeat(1, 1, gt.shape[1], gt.shape[2])
            avg_feature = avg_feature.mul(mask)

            object_loss = torch.nn.functional.mse_loss(num_point * object_feature, avg_feature, reduction='mean')
            total_object_loss = total_object_loss + object_loss

        return total_object_loss
