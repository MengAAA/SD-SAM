import warnings

import cv2
import numpy as np
from mmengine import fileio
from mmseg.datasets.basesegdataset import DATASETS,BaseSegDataset
# from mmseg.datasets.custom import CustomDataset
import os.path as osp
from mmengine.dist import get_dist_info
from mmseg.datasets.transforms import LoadAnnotations
from mmseg.registry import TRANSFORMS, DATASETS
# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from ..base import BaseInterSegDataset
from pathlib import Path
import mmengine


@DATASETS.register_module()
class QH_VALDataset(BaseInterSegDataset):

    METAINFO = dict(
        classes = ('good', 'defect'),
        palette = [[0, 0, 0], [255, 0, 0]]
    )
    default_meta_root = 'data/meta-info/QH_val_mvtec.json'
    # default_ignore_file = 'data/meta-info/QH_invalid_image_names.json'


    def __init__(self,
                 pipeline,
                 data_root,
                 meta_root=None,
                 img_suffix=('.jpg', '.png'),
                 ann_suffix='.png',
                 filter_cfg=None,
                 # ignore_file=None,
                 serialize_data=True,
                 lazy_init=False,
                 max_refetch=1000,
):
        # if ignore_file is None:
        #     ignore_file = self.default_ignore_file
        # if Path(ignore_file).is_file():
        #     self.ignore_sample_indices = \
        #         set(mmengine.load(ignore_file)['train'])
        # else:
        #     warnings.warn(f'Not found ignore_file {ignore_file}')
        #     self.ignore_names = set()
        super().__init__(
            pipeline=pipeline,
            data_root=data_root,
            meta_root=meta_root,
            img_suffix=img_suffix,
            ann_suffix=ann_suffix,
            filter_cfg=filter_cfg,
            serialize_data=serialize_data,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            ignore_index=255,
            backend_args=None
        )

    def load_data_list(self):
        # data_root = Path(self.data_root)
        img_root = Path(self.data_root+'/test/images')
        ann_root = Path(self.data_root + '/test/masks')

        meta_root = Path(self.meta_root)
        if meta_root.is_file():
            data_list = mmengine.load(meta_root)['data_list']
        else:
            data_list = []

            img_files = {}
            # 遍历所有支持的图像后缀
            for suffix in self.img_suffix:
                img_files.update({p.stem: p for p in img_root.rglob(f'*{suffix}')})
            # img_files = {int(p.stem): p for p in
            #              data_root.rglob(f'*{self.img_suffix}')}
            ann_files = {p.stem: p for p in
                         ann_root.rglob(f'*{self.ann_suffix}')}
            prefixes = set(img_files.keys()) & set(ann_files.keys())
            # print(self.ignore_sample_indices)
            # prefixes = prefixes - self.ignore_sample_indices

            for prefix in prefixes:
                data_info = dict(
                    img_path=str(img_files[prefix]),
                    seg_map_path=str(ann_files[prefix]),
                    label_map=None,
                    reduce_zero_label=False,
                    segments_info = [dict(id=1)],
                    seg_fields=[]
                )
                data_list.append(data_info)
            if mmengine.dist.get_dist_info()[0] == 0:
                mmengine.dump(dict(data_list=data_list), meta_root)
        return data_list


    def get_data_info(self, idx):
        data_info = super().get_data_info(idx)
        data_info['dataset'] = 'qhdataset_val'
        # data_info['metainfo']['sample_idx'] = idx
        return data_info

