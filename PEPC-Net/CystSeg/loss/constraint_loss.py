import torch
from torch import nn


class ConstraintLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, seg_pred, edge_pred, seg, edge):
        b, c, h, w, d = seg_pred.shape
        seg_pred = torch.softmax(seg_pred, dim=1)[:, 1].view(b, 1, h, w, d)

        edge[edge < 0] = 0
        edge[edge > 1] = 1
        edge_pred = torch.sigmoid(edge_pred)

        union_pred = torch.max(torch.concat((seg_pred, edge_pred), dim=1), dim=1, keepdim=True)[0]
        union = torch.max(torch.concat((seg, edge), dim=1), dim=1, keepdim=True)[0]

        intersection_pred = torch.min(torch.concat((seg_pred, edge_pred), dim=1), dim=1, keepdim=True)[0]
        intersection = torch.min(torch.concat((seg, edge), dim=1), dim=1, keepdim=True)[0]

        union_pred = torch.where(union_pred > 1 - 1e-3, 1 - 1e-3, union_pred)
        union_pred = torch.where(union_pred < 1e-3, 1e-3, union_pred)
        intersection_pred = torch.where(intersection_pred > 1 - 1e-3, 1 - 1e-3, intersection_pred)
        intersection_pred = torch.where(intersection_pred < 1e-3, 1e-3, intersection_pred)

        return self.ce_loss(union_pred, union) + self.ce_loss(intersection_pred, intersection)

    def ce_loss(self, pre, y):

        loss = -(y * torch.log(pre) + (1 - y) * torch.log(1 - pre))
        loss = torch.mean(loss)
        return loss

