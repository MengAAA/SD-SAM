from typing import Any, Dict

import torch

from mmseg.models import SegDataPreProcessor
from mmseg.registry import MODELS
from mmseg.utils import stack_batch


@MODELS.register_module()
class MMCSSegDataPreProcessor(SegDataPreProcessor):

    """
    Major changes:   #  交换了归一化和填充的顺序
    exchange the order of normalization and padding,
        - original order: normalization -> padding
        - new order:      padding -> normalization
    """

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        # TODO: whether normalize should be after stack_batch
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        inputs = [_input.float() for _input in inputs]
        '''  首先调用 self.cast_data(data) 对数据进行类型转换。
             从 data 字典中提取 inputs 和 data_samples。
             如果 self.channel_conversion 为 True 且输入图像的通道数为 3，则进行颜色通道转换（从 BGR 转换为 RGB） '''
        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
                ''' 如果处于训练阶段，确保 data_samples 不为 None。
                    调用 stack_batch 函数对输入数据和数据样本进行填充操作。
                    如果 self.batch_augments 不为 None，则对输入数据和数据样本进行批量增强操作。'''
        else:
            img_size = inputs[0].shape[1:]
            assert all(input_.shape[1:] == img_size for input_ in inputs),  \
                'The image size in a batch should be the same.'
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)
                ''' 如果处于测试阶段，确保一个批次中的所有图像大小相同。
                    如果 self.test_cfg 不为 None，则调用 stack_batch 函数对输入数据进行填充操作，并将填充信息存储到数据样本的元信息中。
                    如果 self.test_cfg 为 None，则直接将输入数据在第 0 维上进行堆叠'''

        if self._enable_normalize:
            inputs = (inputs - self.mean) / self.std
        return dict(inputs=inputs, data_samples=data_samples)
    '''     如果 self._enable_normalize 为 True，则对输入数据进行归一化处理。
            最后返回一个字典，包含处理后的输入数据和数据样本                    '''
