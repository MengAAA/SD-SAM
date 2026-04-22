import warnings
import numpy as np
import torch


# 主要作用是计算单个样本的二值交并比（Intersection over Union, IoU）。在计算机视觉领域，尤其是图像分割任务中，IoU 是一种常用的评估指标，用于衡量预测结果与真实标签之间的重叠程度。
def calculate_binary_iou_single_sample(pre_label, seg_label, ignore_index=None):
    """
    :param pre_label: np.ndarray or torch.Tensor, shape (height, width)
    :param seg_label: np.ndarray or torch.Tensor, shape (height, width)
    :param ignore_index:
    :return: float
    """
    if isinstance(pre_label, torch.Tensor):
        pre_label = pre_label.detach().cpu().numpy()
    if isinstance(seg_label, torch.Tensor):
        seg_label = seg_label.detach().cpu().numpy()
    if not isinstance(pre_label, np.ndarray):
        raise TypeError(f'Cannot handle type of pre_label: {type(pre_label)}')
    if not isinstance(seg_label, np.ndarray):
        raise TypeError(f'Cannot handle type of seg_label: {type(seg_label)}')
    if pre_label.shape != seg_label.shape:
        raise ValueError(f'pre_label.shape != seg_label.shape: '
                         f'{pre_label.shape} != {seg_label.shape}')
    if ignore_index is not None:
        mask = seg_label != ignore_index
    else:
        mask = np.ones_like(seg_label, dtype=bool)
    pre_label = pre_label[mask]
    seg_label = seg_label[mask]

    intersect = pre_label[pre_label == seg_label]
    area_intersect = np.bincount(intersect)
    area_pre_label = np.bincount(pre_label)
    area_seg_label = np.bincount(seg_label)
    area_union = area_pre_label + area_seg_label - area_intersect
    if len(area_union) > 2:
        warnings.warn(f'Found {len(area_union)} 2 classes in the label')
    elif len(area_union) == 1:
        warnings.warn(f'Found only background label')
        return 0.0
    elif area_union[1] == 0:
        warnings.warn(f'Found invalid foreground label')
        return 0.0
    if len(area_intersect) == 1:
        return 0.0
    return float(area_intersect[1]) / float(area_union[1])
