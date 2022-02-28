import torch
import statistics
from collections import defaultdict

def binary_precision(TP, FP): 
    if int(TP) == 0 and int(FP) == 0:
        return TP / (TP + FP + 1) # return 0
    return TP / (TP + FP)

def binary_recall(TP, FN):
    if int(TP) == 0 and int(FN) == 0:
        return TP / (TP + FN + 1) # return 0
    return TP / (TP + FN)

def binary_f1(PC, RC):
    if float(PC) == 0. and float(RC) == 0:
        return 2*RC*PC/(RC+PC+1)
    return 2*RC*PC/(RC+PC)


class _Evaluation(object):
    def __init__(self, num_classes):
        assert num_classes >=2, "num_class should be larger than 1"
        self.num_classes = num_classes
        self.classes = list(range(num_classes))

        
class ClassificationEvaluation(_Evaluation):
    def __init__(self, num_classes, threshold: list=[0.], return_threshold=0.5):
        assert isinstance(threshold, list), f"threshold is exptected 'list', but {type(threshold)}"
        assert len(threshold)>0, "threshold is exptected to contain more than 1 value, but no values in the list"
        assert return_threshold in threshold, f"return_threshold {return_threshold} is not in threshold {threshold}"
        super().__init__(num_classes)
        self.threshold = threshold
        self.return_threshold = return_threshold
        self.TP = {t: {c: 0 for c in self.classes} for t in self.threshold}
        self.TN = {t: {c: 0 for c in self.classes} for t in self.threshold}
        self.FP = {t: {c: 0 for c in self.classes} for t in self.threshold}
        self.FN = {t: {c: 0 for c in self.classes} for t in self.threshold}
        self.len = 0
            
        
        
    def __call__(self, pred, GT, SR, mask, rt=False):
        """
            pred : classification results. Tensor[shape N, C, d1, d2, ...]
            GT : classification labels. Tensor [shape N, C, d1, d2, ...]
            SR : segmentation results. Tensor[shape N, C, H, W]
            mask : segmentation masks. Tensor[shape N, C, H, W]
        """
        assert len(pred.shape) >= 2
        assert pred.shape == GT.shape, f"predict tensor and GT tensor are expected to be same shape, but {pred.shape} != {GT.shape}"
        assert pred.shape[1] == self.num_classes, f"The number of classes are expected {self.num_classes}, but {pred.shape[1]}"
        assert len(SR.shape) >= 2
        assert SR.shape == mask.shape, f"predict tensor and GT tensor are expected to be same shape, but {SR.shape} != {mask.shape}"
        assert SR.shape[1] == self.num_classes+1, f"The number of classes are expected {self.num_classes+1}, but {SR.shape[1]}"
        SR = SR.contiguous().view(SR.shape[0], SR.shape[1], -1)
        pred = pred.contiguous().view(pred.shape[0], -1)
        GT = GT.contiguous().view(GT.shape[0], -1)
        mask = mask.contiguous().view(mask.shape[0], mask.shape[1], -1)
        iou = SegmentationEvaluation.get_IoU(SR, mask)[:, 1:] # N, 2
        for t in self.threshold:
            for i in range(GT.shape[1]):
                CM = ClassificationEvaluation.binary_confusion_matrix(pred[:, i], GT[:, i], iou[:, i], t)
                try:
                    self.TP[t][i] += int(CM[:, 0].sum())
                    self.FP[t][i] += int(CM[:, 1].sum())
                    self.FN[t][i] += int(CM[:, 2].sum())
                    self.TN[t][i] += int(CM[:, 3].sum())
                except:
                    import pdb
                    pdb.set_trace()
                    self.TP[t][i] += int(CM[:, 0].sum())
                    self.FP[t][i] += int(CM[:, 1].sum())
                    self.FN[t][i] += int(CM[:, 2].sum())
                    self.TN[t][i] += int(CM[:, 3].sum())
                if t == self.return_threshold:
                    return_CM = CM
        self.len += GT.shape[0]
        if rt:
            return return_CM
        
    @staticmethod
    def binary_confusion_matrix(pred, GT, iou, threshold):
        assert len(pred.shape) == 1
        assert len(GT.shape) == 1
        assert len(iou.shape) == 1
        assert pred.shape[0] == GT.shape[0]
        assert pred.shape[0] == iou.shape[0]
        assert threshold >= 0 and threshold < 1
        
        TP = (pred == 1) * (GT == 1) * (iou >= threshold)
        TN = (pred == 0) * (GT == 0)
        FP = (pred == 1) * (GT == 0)
        FN = ((pred == 0) * (GT == 1) + (pred == 1) * (GT == 1) * (iou < threshold)) >= 1
        return torch.concat([TP.unsqueeze(1), FP.unsqueeze(1), FN.unsqueeze(1), TN.unsqueeze(1)], dim=1).type(torch.int)

        
class SegmentationEvaluation(_Evaluation): 
    def __init__(self, num_classes, reduction='mean'):
        super().__init__(num_classes)
        self.reduction = reduction
        
        self.IoU = {c: [] for c in range(self.num_classes)}
        self.DSC = {c: [] for c in range(self.num_classes)}
        self.IoU['samplewise'] = defaultdict(float)
        self.DSC['samplewise'] = defaultdict(float)
        self.IoU['classwise'] = []
        self.DSC['classwise'] = []
        
        self.len = defaultdict(int)
        
        self.reduction_fn = {"mean": statistics.mean,
                             "sum": sum,
                             "none": lambda x: x}.get(reduction,
                                                      lambda x: None)
    
    def __call__(self, pred, GT, smooth=1e-3):
        """
            pred : Tensor[shape N, C, d1, d2, ...]
            GT : Tensor [shape N, C, d1, d2, ...]
        """
        assert len(pred.shape) >= 2
        assert pred.shape == GT.shape, f"predict tensor and GT tensor are expected to be same shape, but {pred.shape} != {GT.shape}"
        assert pred.shape[1] == self.num_classes, f"The number of classes are expected {self.num_classes}, but {pred.shape[1]}"
        pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        GT = GT.contiguous().view(GT.shape[0], GT.shape[1], -1)
        
        IoU = SegmentationEvaluation.get_IoU(pred, GT, smooth) # N, C
        DSC = SegmentationEvaluation.get_DSC(pred, GT, smooth)
        for c in range(GT.shape[1]):
            self.IoU[c] += IoU[:, c].tolist()
            self.DSC[c] += DSC[:, c].tolist()
            self.len[c] += (GT.sum(2)[:, c] > 0).sum()

        self.mean_over_classes(GT.shape[0]) # mean over classes
        self.mean_over_samples()
        
    def mean_over_classes(self, num):
        for metric in [self.IoU, self.DSC]:
            for n in range(num, 0, -1):
                metric["classwise"].append(statistics.mean([class_score[self.len[0]-n] for c, class_score in metric.items() if isinstance(c, int)]))
    
    def mean_over_samples(self):
        for metric in [self.IoU, self.DSC]:
            for c, class_score in metric.items():
                if isinstance(c, int):
                    target_scores = [s for s in class_score if s!=0]
                    if len(target_scores) > 0:
                        metric["samplewise"][c] = self.reduction_fn(target_scores)
                    else:
                        metric["samplewise"][c] = 0

    @staticmethod
    def get_IoU(pred, GT, smooth=1):
        """
             pred : Tensor[shape N, C, H*W]
             GT : Tensor [shape N, C, H*W]
             output : Tensor[shape N, C]
         """
        inter = (pred * GT).sum(dim=2)
        union = ((pred + GT) >= 1).sum(dim=2)
        
        return inter / (union + smooth)
    
    @staticmethod
    def get_DSC(pred, GT, smooth=1):
        """
             pred : Tensor[shape N, C, H*W]
             GT : Tensor [shape N, C, H*W]
             output : Tensor[shape N, C]
        """
        
        inter = (pred * GT).sum(dim=2)
        union = pred.sum(dim=2) + GT.sum(dim=2)
        
        return (2 * inter) / (union + smooth)