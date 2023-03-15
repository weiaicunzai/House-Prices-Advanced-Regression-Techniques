import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn

from losses import GlandContrastLoss



class ContrastiveHead(nn.Module):
    """Head for contrastive learning.

    The contrastive loss is implemented in this head and is used in SimCLR,
    MoCo, DenseCL, etc.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Defaults to 0.1.
    """

    def __init__(self, temperature=0.1):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, pos, neg):
        """Forward function to compute contrastive loss.

        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).to(pos.device)
        losses = dict()
        losses['loss'] = self.criterion(logits, labels)
        return losses['loss']


def dummy_inputs():
    logits_pos = torch.randn(2, 3)
    logits_neg = torch.randn(2, 3)

    return logits_pos, logits_neg

def dummy_infonce(logits_pos, logits_neg):

    contrasive_head = ContrastiveHead(temperature=0.07)
    #for i in range(logits_pos.shape[])
    n, k = logits_pos.shape

    loss = 0
    for i in range(k):
        #print(i)
        #print(logits_pos[:, i])
        lp = logits_pos[:, i].view(n, -1)
        #print(lp.shape)
        sub_loss = contrasive_head(lp, logits_neg)
        loss += sub_loss

    return loss / k


def test_infonce():
    loss = GlandContrastLoss(2, ignore_idx=255)
    logits_pos, logits_neg = dummy_inputs()
    loss = loss.infonce(logits_pos, logits_neg)

    #for pos in logits_pos:

    dummy_loss = dummy_infonce(logits_pos, logits_neg)
    print(dummy_loss, loss, dummy_loss - loss)

    assert loss - dummy_loss < 1e-6










test_infonce()