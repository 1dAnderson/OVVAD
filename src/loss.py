import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def CLAS(logits, label, seq_len, criterion,device='cpu'):  
    logits = logits.squeeze()
    ins_logits = torch.zeros(0).to(device)  # tensor([])
    for i in range(logits.shape[0]):
        if label[i] == 0:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=1, largest=True)

            #tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i]), largest=True)
            
        else:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i]//16+1), largest=True)
        tmp = torch.mean(tmp).view(1)
        ins_logits = torch.cat((ins_logits, tmp))

    clsloss = criterion(ins_logits, label)
    return clsloss

def CLAS_not1(logits, label, seq_len, criterion,device='cpu'): #for xd
    logits = logits.squeeze()
    ins_logits = torch.zeros(0).to(device)  # tensor([])
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i]//16+1), largest=True)
        tmp = torch.mean(tmp).view(1)
        ins_logits = torch.cat((ins_logits, tmp))

    clsloss = criterion(ins_logits, label)
    return clsloss

# focal loss without sigmiod
def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = inputs
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

class PseLoss(nn.Module):
    """Focal Loss without mask and scale operations"""

    def __init__(self, config):
        super(PseLoss, self).__init__()
        self.alpha = config.pse_alpha
        self.gamma = config.pse_gamma
        self.threshold = config.pse_threshold

    def forward(self, logits, pseudo_label):
        # 将伪标签二值化（大于阈值为正类，小于等于为负类）
        pseudo_label = (pseudo_label > self.threshold)
        # print("label:",len(logits), logits)
        # print("pse_label:", len(pseudo_label), pseudo_label)
        # 计算 Focal Loss
        loss = focal_loss(logits.float(), pseudo_label.float(), alpha=self.alpha, gamma=self.gamma, reduction="mean")

        return loss
