from torch import nn
import torch
import torch.nn.functional as F


def structure_loss(logits, mask):
   """
    loss function (ref: F3Net-AAAI-2020)

    pred: logits without activation
    mask: binary mask {0, 1}
    """
   weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
   wbce = F.binary_cross_entropy_with_logits(logits, mask, reduction='mean')
   wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

   pred = torch.sigmoid(logits)
   inter = ((pred * mask) * weit).sum(dim=(2, 3))
   union = ((pred + mask) * weit).sum(dim=(2, 3))
   wiou = 1 - (inter + 1) / (union - inter + 1)
   return (wbce + wiou).mean()


class FocalLossWithLogits(nn.Module):
   def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
      super(FocalLossWithLogits, self).__init__()
      self.alpha = alpha
      self.gamma = gamma
      self.reduction = reduction
      self.crit = nn.BCEWithLogitsLoss(reduction='none')

   def forward(self, logits, label):
      probs = torch.sigmoid(logits)
      coeff = torch.abs(label - probs).pow(self.gamma).neg()
      log_probs = torch.where(logits >= 0,
                              F.softplus(logits, -1, 50),
                              logits - F.softplus(logits, 1, 50))
      log_1_probs = torch.where(logits >= 0,
                                -logits + F.softplus(logits, -1, 50),
                                -F.softplus(logits, 1, 50))
      loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
      loss = loss * coeff

      if self.reduction == 'mean':
         loss = loss.mean()
      if self.reduction == 'sum':
         loss = loss.sum()
      return loss