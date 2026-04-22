from typing import Union
import cv2
import numpy as np
import torch

# 该函数在 focsam/engine/utils/click.py 文件中被调用，用于生成交互点击点。在生成点击点的过程中，需要计算预测错误的区域（假负和假正）的距离变换，以确定点击的位置
def mask_to_distance(mask: Union[torch.Tensor, np.ndarray],
                     boundary_padding: bool) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert a binary mask to a distance map.

    :param mask: A binary mask of shape (*, height, width).
    :param boundary_padding: A boolean flag indicating whether to pad the
    boundary of the mask before computing the distance map.
    :return: A distance map of the same shape as the input mask.
    """
    if not isinstance(mask, (torch.Tensor, np.ndarray)):
        raise TypeError(f'Cannot handle type of mask: {type(mask)}')

    to_tensor = isinstance(mask, torch.Tensor)
    device = mask.device if to_tensor else None
    mask = mask.detach().cpu().numpy() if to_tensor else mask

    if boundary_padding:
        mask = np.pad(
            mask, [(0, 0)] * (len(mask.shape) - 2) + [(1, 1)] * 2)

    dist = np.stack(
        list(map(
            lambda x: cv2.distanceTransform(x, cv2.DIST_L2, 0),
            mask.reshape((-1, ) + mask.shape[-2:]).astype(np.uint8))),
        axis=0).reshape(mask.shape)
    # cv2.distanceTransform：使用 OpenCV 的 distanceTransform 函数计算每个二维掩码的距离变换，cv2.DIST_L2 表示使用欧几里得距离。
    # map 函数：对 mask 中的每个二维掩码应用 cv2.distanceTransform 函数。
    # np.stack 函数：将计算得到的距离图堆叠起来，恢复到与输入 mask 相同的形状。
    if boundary_padding:
        dist = dist[..., 1:-1, 1:-1]

    dist = torch.from_numpy(dist).to(device) if to_tensor else dist
    return dist
