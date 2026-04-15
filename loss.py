import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce

        mask = targets != self.ignore_index
        return focal[mask].mean()
    

class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=0, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        mask = target != self.ignore_index
        losses = []
        for c in range(self.num_classes):
            p = pred[:, c][mask]
            t = (target[mask] == c).float()
            intersection = (p * t).sum()
            losses.append(1 - (2 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth))
        return torch.stack(losses).mean()


class TverskyLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.7, beta=0.3, ignore_index=0, smooth=1e-6):
        # alpha controls FN penalty, beta controls FP penalty
        # alpha > beta → penalise missed detections more
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        mask = target != self.ignore_index
        losses = []
        for c in range(self.num_classes):
            p  = pred[:, c][mask]
            t  = (target[mask] == c).float()
            tp = (p * t).sum()
            fn = ((1 - p) * t).sum()
            fp = (p * (1 - t)).sum()
            losses.append(1 - (tp + self.smooth) / (tp + self.alpha*fn + self.beta*fp + self.smooth))
        return torch.stack(losses).mean()


class FocalDiceCombinationLoss(nn.Module):
    def __init__(self, num_classes, focal_loss_weight=0.5, weight=None, ignore_index=0, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.focal_loss_weight = focal_loss_weight
        self.ignore_index = ignore_index
        self.weight = weight

        self.focal = FocalLoss(gamma, weight, ignore_index)
        self.dice = MultiClassDiceLoss(num_classes, ignore_index, smooth)

    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.focal_loss_weight * focal_loss + (1.0 - self.focal_loss_weight) * dice_loss


class OpenSetLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=None, loss_weights={"source_cset": 1.0, "source_oset": 0.5}):
        super().__init__()
        self.loss_weights = loss_weights
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def forward(self, logits, openset_logits, y):
        # cross entropy over closed set classes
        # print('loss inputs', logits.shape, openset_logits.shape, y.shape)
        
        cset_loss = F.cross_entropy(logits, y, ignore_index=self.ignore_index)

        # dealing with unknown class with open set head
        oset_prob = F.softmax(openset_logits, dim=1)

        # print(torch.bincount(y.ravel()), self.ignore_index)
        valid_mask = (y != self.ignore_index)  # shape same as labels
        oset_pos_target = F.one_hot(y * valid_mask, num_classes=self.num_classes)
        oset_pos_target = oset_pos_target.permute(0, 3, 1, 2).float()
        oset_neg_target = 1 - oset_pos_target
        # print('oset targets', oset_pos_target.shape, oset_neg_target.shape)
        
        # creating mask to ignore background pixels in the loss calculation
        valid_mask_exp = valid_mask.unsqueeze(1)
        oset_pos_target_m = oset_pos_target * valid_mask_exp
        oset_neg_target_m = oset_neg_target * valid_mask_exp
        oset_prob_m = oset_prob * valid_mask_exp.unsqueeze(1)
        # print('masked oset targets and prob', oset_pos_target_m.shape, oset_neg_target_m.shape, oset_prob_m.shape)
        
        oset_pos_loss = torch.sum(-oset_pos_target_m * torch.log(oset_prob_m[:, 0, :, :, :] + 1e-8), dim=[1, 2, 3])
        # print('oset pos loss', oset_pos_loss.shape)
        oset_neg_loss = torch.max((-oset_neg_target_m * torch.log(oset_prob_m[:, 1, :, :, :] + 1e-8)).reshape(logits.size(0), -1), dim=1)[0]
        # print('oset neg loss', oset_neg_loss.shape)

        n_valid = valid_mask.sum(dim=[1, 2]).clamp(min=1)  # (batch,) - clamp = avoid div by zero

        oset_pos_loss = (oset_pos_loss / n_valid).mean()
        oset_neg_loss = oset_neg_loss.mean()

        oset_loss = oset_pos_loss + oset_neg_loss

        loss = (
            cset_loss * self.loss_weights["source_cset"]
            + oset_loss * self.loss_weights["source_oset"]
        )
        
        return loss
