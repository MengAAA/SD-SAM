import copy
import logging

import numpy as np
from mmengine.dataset import BaseDataset, Compose
from mmengine.logging import print_log


class InvalidInterSegSampleError(Exception):
    """Invalid InterSeg sample."""
    pass


class NotFoundValidInterSegSampleError(Exception):
    """Not found valid InterSeg sample."""
    pass


class BaseInterSegDataset(BaseDataset):

    default_meta_root = ''

    def __init__(self,
                 pipeline,
                 data_root,
                 meta_root=None,
                 img_suffix='.jpg',
                 ann_suffix='.png',
                 filter_cfg=None,
                 serialize_data=True,
                 lazy_init=False,
                 max_refetch=1000,
                 ignore_index=255,
                 backend_args=None):
        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix
        self.ignore_index = ignore_index
        self.backend_args = backend_args.copy() if backend_args else None

        self.data_root = data_root
        self.meta_root = meta_root or self.default_meta_root
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = None
        self.serialize_data = serialize_data
        self.max_refetch = max_refetch
        self.data_list = []
        self.data_bytes: np.ndarray

        self.pipeline = Compose(pipeline)
        if not lazy_init:
            self.full_init()
# BaseInterSegDataset 继承自 BaseDataset，是一个基础的交互式分割数据集类。
# __init__ 方法：初始化数据集的基本参数，包括数据根目录、数据处理管道、图像和标注文件的后缀等。如果 lazy_init 为 False，则调用 full_init 方法进行全量初始化。
# load_data_list 方法：是一个抽象方法，需要在子类中实现具体的加载逻辑。
# __getitem__ 方法：根据索引获取数据样本。如果数据集未完全初始化，则发出警告并进行初始化。尝试多次获取有效的数据样本，如果多次尝试后仍未获取到，则抛出 NotFoundValidInterSegSampleError 异常。
# get_data_info 方法：获取数据信息，并添加 dataset 字段，其值为数据集类名去掉 Dataset 后的小写形式。
# metainfo 属性：返回一个空字典，用于存储数据集的元信息。

    def load_data_list(self):
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            print_log(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        for _ in range(self.max_refetch):
            try:
                data = self.prepare_data(idx)
            except InvalidInterSegSampleError  as e :
                # 打印无效样本的图像名称
                img_path = self.data_list[idx]['img_path']
                print(f"Invalid sample found: {img_path}, error: {e}")
                continue
            return data

        # print(self.prepare_data(idx))
        raise NotFoundValidInterSegSampleError(
            f'Cannot find valid InterSeg sample after '
            f'{self.max_refetch} retries.')

    def get_data_info(self, idx):
        data_info = super(BaseInterSegDataset, self).get_data_info(idx)
        data_info['dataset'] = \
            self.__class__.__name__.lower().replace('dataset', '')

        return data_info

    @property
    def metainfo(self):
        return dict()
