import torch
from torch import nn


class AdaptiveWingLoss(nn.Module):
    def __init__(self, alpha=2.1, omega=14.0, epsilon=1.0, theta=0.5):
        super().__init__()
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta

    def forward(self, y_pred, y):
        lossMat = torch.zeros_like(y_pred)
        A = self.omega * (1 / (1 + (self.theta / self.epsilon) ** (self.alpha - y))) * (self.alpha - y) * (
                (self.theta / self.epsilon) ** (self.alpha - y - 1)) / self.epsilon
        C = self.theta * A - self.omega * torch.log(1 + (self.theta / self.epsilon) ** (self.alpha - y))
        case1_ind = torch.abs(y - y_pred) < self.theta
        case2_ind = torch.abs(y - y_pred) >= self.theta
        lossMat[case1_ind] = (self.omega * torch.log(
            1 + torch.abs((y[case1_ind] - y_pred[case1_ind]) / self.epsilon) ** (self.alpha - y[case1_ind]))).to(
            lossMat.dtype)
        lossMat[case2_ind] = (A[case2_ind] * torch.abs(y[case2_ind] - y_pred[case2_ind]) - C[case2_ind]).to(
            lossMat.dtype)

        return lossMat


class WeightedAdaptiveWingLoss(nn.Module):
    def __init__(self, W=10, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.W = float(W)
        self.Awing = AdaptiveWingLoss(alpha, omega, epsilon, theta)

    def forward(self, y_pred, y):
        M = self.generate_weight_map(y)
        Loss = self.Awing(y_pred, y)
        weighted = Loss * (self.W * M + 1.)
        return weighted.mean()

    def generate_weight_map(self, y, threshold=0.2):
        M = torch.where(y > threshold, 1, 0)
        return M

