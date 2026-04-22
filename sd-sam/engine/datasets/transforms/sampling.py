import warnings
import numpy as np
import numpy.random as rng
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS

from engine.datasets.base import InvalidInterSegSampleError


@TRANSFORMS.register_module()
class ObjectSampler(BaseTransform):

    def __init__(self,
                 max_num_merged_objects=1,
                 min_area_ratio=0.0,
                 ignore_index=None,
                 merge_prob=0.0,
                 include_other=False,
                 max_retry=100):
        self.max_num_merged_objects = max_num_merged_objects
        self.min_area_ratio = min_area_ratio
        self.ignore_index = ignore_index
        self.merge_prob = merge_prob
        self.include_other = include_other
        self.max_retry = max_retry
# @TRANSFORMS.register_module()：这是一个装饰器，用于将 ObjectSampler 类注册到 TRANSFORMS 注册表中，这样在数据处理流程中可以方便地使用这个变换类。
# __init__ 方法用于初始化类的参数：
# max_num_merged_objects：最大合并对象数，默认值为 1。
# min_area_ratio：最小面积比例，用于判断采样后的分割标签中目标区域的面积是否满足要求，默认值为 0.0。
# ignore_index：需要忽略的索引值，例如在分割标签中表示背景或无效区域的索引，默认值为 None。
# merge_prob：合并对象的概率，默认值为 0.0。
# include_other：是否包含其他未合并的对象，默认值为 False。
# max_retry：最大重试次数，当采样不满足条件时进行重试，默认值为 100。

    def transform(self, results):
        gt_seg_map = results.pop('gt_seg_map')
        segments_info = []
        for info in results.pop('segments_info'):
            if np.any(gt_seg_map == info['id']):
                segments_info.append(info)
        if len(segments_info) == 0:
            print(results)
            print(f"Debug info: gt_seg_map unique values: {np.unique(gt_seg_map)}")
            # print(f"Debug info: segments_info ids: {[info['id'] for info in results['segments_info']]}")
            raise InvalidInterSegSampleError('Not found valid annotation')

        merge_prob = self.merge_prob
        ignore_index = self.ignore_index
        min_area_ratio = self.min_area_ratio
        num_objects = len(segments_info)
        max_num_merged_objects = max(self.max_num_merged_objects, num_objects)
        for _ in range(self.max_retry):
            seg_label = np.zeros_like(gt_seg_map)
            object_idxs = rng.permutation(range(num_objects))
            if max_num_merged_objects > 1 and rng.rand() < merge_prob:
                num_merged_objects = rng.randint(2, max_num_merged_objects + 1)
            else:
                num_merged_objects = 1
            for idx in object_idxs[:num_merged_objects]:
                seg_label[gt_seg_map == segments_info[idx]['id']] = 1
            if np.mean(seg_label) > min_area_ratio:
                if ignore_index is not None:
                    seg_label[gt_seg_map == ignore_index] = ignore_index
                if self.include_other:
                    for idx in object_idxs[num_merged_objects:]:
                        seg_label[gt_seg_map == segments_info[idx]['id']] = 2
                results['gt_seg_map'] = seg_label
                return results
        else:
            raise InvalidInterSegSampleError('Failed to sample valid objects')
# 从 results 中取出 gt_seg_map（真实分割标签）和 segments_info（目标信息列表）。
# 遍历 segments_info，筛选出在 gt_seg_map 中存在的目标信息，存储在 segments_info 中。如果筛选后 segments_info 为空，则抛出 InvalidInterSegSampleError 异常。
# 根据初始化参数设置 merge_prob、ignore_index 和 min_area_ratio。
# 计算目标对象的数量 num_objects，并确定最大合并对象数 max_num_merged_objects。
# 进行 max_retry 次重试：
# 初始化 seg_label 为与 gt_seg_map 形状相同的全零数组。
# 对目标对象索引进行随机排列。
# 根据 merge_prob 和 max_num_merged_objects 决定是否进行对象合并以及合并的对象数量 num_merged_objects。
# 将选中的对象在 seg_label 中标记为 1。
# 如果 seg_label 中目标区域的平均面积大于 min_area_ratio，则进行以下操作：
# 如果 ignore_index 不为 None，将 gt_seg_map 中等于 ignore_index 的像素在 seg_label 中也设置为 ignore_index。
# 如果 include_other 为 True，将未合并的对象在 seg_label 中标记为 2。
# 将处理后的 seg_label 放回 results 中并返回。
# 如果重试次数用尽仍未满足条件，则抛出 InvalidInterSegSampleError 异常。

    def __repr__(self):
        return f'{self.__class__.__name__}(' \
               f'max_num_merged_objects={self.max_num_merged_objects}, ' \
               f'min_area_ratio={self.min_area_ratio}, ' \
               f'ignore_index={self.ignore_index}, ' \
               f'merge_prob={self.merge_prob}, ' \
               f'include_other={self.include_other}, ' \
               f'max_retry={self.max_retry})'


@TRANSFORMS.register_module()
class MultiObjectSampler(BaseTransform):

    """
    Sample multiple objects from the gt_seg_map,
    each represented by a unique label within the same gt_seg_map.
    """

    def __init__(self, num_objects, min_area_ratio=0.0):
        self.num_objects = num_objects
        self.min_area_ratio = min_area_ratio

    def transform(self, results):
        gt_seg_map = results.pop('gt_seg_map')
        segments_info = []
        for info in results.pop('segments_info'):
            if np.any(gt_seg_map == info['id']):
                segments_info.append(info)
        if len(segments_info) == 0:
            raise InvalidInterSegSampleError('Not found valid annotation')

        num_objects = 0
        seg_label = np.zeros_like(gt_seg_map)
        for idx in rng.permutation(len(segments_info)):
            mask = (gt_seg_map == segments_info[idx]['id'])
            if mask.astype(np.float32).mean() > self.min_area_ratio:
                num_objects += 1
                seg_label[mask] = num_objects
                if num_objects >= self.num_objects:
                    break
        else:
            warnings.warn(f'Failed to sample {self.num_objects} objects, '
                          f'only sampled {num_objects} objects.')
        if num_objects > 0:
            results['gt_seg_map'] = seg_label
            return results
        else:
            raise InvalidInterSegSampleError('Failed to sample valid objects')

    def __repr__(self):
        return f'{self.__class__.__name__}(' \
               f'num_objects={self.num_objects}, ' \
               f'min_area_ratio={self.min_area_ratio})'
