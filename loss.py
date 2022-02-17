import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CrossEntropy', 'FocalCrossEntropy', 'InverseCrossEntropy','FocalDiceLoss', 'DiceLoss', 'GeneralizedDiceLoss']

# Base
class _Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(_Loss, self).__init__()
        self.reduction = reduction
        
        
class _WeightedLoss(_Loss):
    """
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
    """
    def __init__(self, weight = None, ignore_index=None, reduction='mean'):
        super(_WeightedLoss, self).__init__(reduction)
        assert reduction in ['mean', 'sum', 'none']
        self.ignore_index = ignore_index
        self.weight = weight
        if self.weight is not None and self.ignore_index is not None:
            assert isinstance(ignore_index, int)
            self.weight[ignore_index] = 0
        
        
# Cross Entropy
class CrossEntropy(_WeightedLoss):
    """
    Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
    Return:
        a float value when reduction is 'mean' or 'sum'.
        a list of float values when reduction is 'none'.
    """
    def __init__(self, weight=None, ignore_index=None, reduction='mean'):
        super(CrossEntropy, self).__init__(weight, ignore_index, reduction)
        
    def forward(self, predict, target, softmax=True):  # (N, C, H, W) N, C
        assert predict.shape == target.shape, f'predict & target shape do not match. {predict.shape} != {target.shape}'
        if softmax:
            predict = F.softmax(predict, dim=1)
            
        predict = predict.contiguous().view(predict.shape[0], predict.shape[1], -1) # N, C, H*W
        target = target.contiguous().view(target.shape[0], target.shape[1], -1)
            
        loss = self.get_loss(predict, target) # N, C, H*W
        loss = loss.mean(2) # N, C
        if self.weight is not None:
            loss = torch.mul(self.weight, loss)
        loss = loss.sum(dim=1) # N
        loss *= -1
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
    
    def get_loss(self, predict, target):
        loss = torch.mul(target, torch.log(predict + 1e-9))
        return loss

    
class FocalCrossEntropy(CrossEntropy):
    def __init__(self, weight=None, ignore_index=None, alpha=1, gamma=2, reduction='mean'):
        super(FocalCrossEntropy, self).__init__(weight, ignore_index, reduction)
        self.alpha = alpha
        self.gamma = gamma
    
    def get_loss(self, predict, target):
        loss = (1-predict).pow(self.gamma).mul(self.alpha).mul(target).mul(torch.log(predict + 1e-9))
        return loss


class InverseCrossEntropy(CrossEntropy):
    def __init__(self, weight=None, ignore_index=None, alpha=1, smooth=1, reduction='mean'):
        super(InverseCrossEntropy, self).__init__(weight, ignore_index, reduction)
        self.alpha = alpha
        self.smooth = smooth
    
    def get_loss(self, predict, target):
        assert len(target.shape) == 3
        class_pixels = self.inverse_frequency_weight(target)
        loss = torch.mul(class_pixels, torch.mul(target, torch.log(predict + 1e-9)))
        return loss
    
    def inverse_frequency_weight(self, target):
        class_pixels = target.sum(2, keepdim=True) # N, C, H*W (0 ~ 1) -> N, C, 1 (0 ~ # of pixels)
        weight = 1 / (class_pixels.pow(self.alpha) + self.smooth)
        return weight
    

# Dice Loss
class DiceLoss(_WeightedLoss):
    """Dice loss, need one hot encode input (predict, target)
    Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
    Return:
        a float value when reduction is 'mean' or 'sum'.
        a list of float values when reduction is 'none'.
    """
    def __init__(self, weight=None, ignore_index=None, smooth=1, reduction='mean'): 
        super(DiceLoss, self).__init__(weight, ignore_index, reduction)
        self.smooth = smooth
        
    def get_num_den(self, predict, target):
        num = torch.sum(torch.mul(predict, target), dim=2)
        den = torch.sum(predict+target, dim=2)
        return num, den
    
    def forward(self, predict, target, softmax=True):
        assert predict.shape == target.shape, f'predict & target shape do not match. {predict.shape} != {target.shape}'
        if softmax:
            predict = F.softmax(predict, dim=1)
            
        predict = predict.contiguous().view(predict.shape[0], predict.shape[1], -1) 
        target = target.contiguous().view(target.shape[0], target.shape[1], -1)

        num, den = self.get_num_den(predict, target)
        if self.weight is not None:
            num = torch.mul(self.weight, num)
            den = torch.mul(self.weight, den)
        num = num.sum(1)
        den = den.sum(1)
        
        total_loss = 1 - (2.0 * num) / (den+self.smooth)
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        elif self.reduction == 'none':
            return total_loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
            
            
class FocalDiceLoss(DiceLoss):
    def __init__(self, weight=None, ignore_index=None, smooth=1, r=1, reduction='mean'): 
        super(FocalDiceLoss, self).__init__(weight, ignore_index, smooth, reduction)
        self.r = r
        
    def get_num_den(self, predict, target):
        num = torch.sum(torch.mul(predict, target).mul((1-predict).pow(self.r)), dim=2)
        den = torch.sum(predict+target, dim=2)
        return num, den
        
    
class GeneralizedDiceLoss(DiceLoss):
    def __init__(self, weight=None, ignore_index=None, alpha=2, smooth=1, reduction='mean'):
        super(GeneralizedDiceLoss, self).__init__(weight, ignore_index, smooth, reduction)
        self.alpha = alpha
        
    def get_num_den(self, predict, target):
        pixel_weight = self.inverse_frequency_weight(target)
        num = torch.sum(torch.mul(pixel_weight, torch.mul(predict, target)), dim=2)
        den = torch.sum(torch.mul(pixel_weight, predict + target), dim=2)
        return num, den
    
    def inverse_frequency_weight(self, target):
        class_pixels = target.sum(2, keepdim=True) # N, C, H*W (0 ~ 1) -> N, C, 1 (0 ~ # of pixels)
        weight = 1 / (class_pixels.pow(self.alpha) + self.smooth)
        return weight
    
    
if __name__=='__main__':
    from utils import make_one_hot
    print("random tensor size : (4,3,5,5)")
    print("weight = None")
    predict = torch.rand((4,3,5,5))
    target = torch.randint(low=0, high=3, size=(4,1,5,5))
    target = make_one_hot(target, num_classes=3)
    for loss in ['CrossEntropy', 'FocalCrossEntropy', 'InverseCrossEntropy', 'DiceLoss', 'FocalDiceLoss', 'GeneralizedDiceLoss']:
        print(f"{loss}: {eval(loss)()(predict, target)}")
        
