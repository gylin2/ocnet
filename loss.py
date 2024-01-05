import torch
import torch.nn as nn
    
class OCSoftmax(nn.Module):
    def __init__(self, r_real=0.5, r_fake=0.2, alpha=20.0):
        super(OCSoftmax, self).__init__()
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        x = x.squeeze()        
        x[labels == 0] = self.r_real - x[labels == 0]
        x[labels == 1] = x[labels == 1] - self.r_fake   
        loss = self.softplus(self.alpha * x).mean()
        return loss

class TOCSoftmax(nn.Module):
    def __init__(self, r_real=0.5, r_fake=0.2, alpha=20.0):
        super(TOCSoftmax, self).__init__()
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        x = x.squeeze()
        weight = torch.logical_not((labels == 1) & (x < self.r_fake)).float()
        x[labels == 0] = self.r_real - x[labels == 0]
        x[labels == 1] = x[labels == 1] - self.r_fake
        loss = (weight*self.softplus(self.alpha * x)).mean()
        return loss
    