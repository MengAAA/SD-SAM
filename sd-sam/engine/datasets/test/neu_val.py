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
class NEU_VALDataset(BaseInterSegDataset):

    METAINFO = dict(
        classes = ('good', 'defect'),
        palette = [[0, 0, 0], [255, 0, 0]]
    )
    default_meta_root = 'data/meta-info/neu_mvtec_val.json'

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
):
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
        data_root = Path(self.data_root)
        meta_root = Path(self.meta_root)
        if meta_root.is_file():
            data_list = mmengine.load(meta_root)['data_list']
        else:
            data_list = []
            img_files = {int(p.stem): p for p in
                         data_root.rglob(f'*{self.img_suffix}')}
            ann_files = {int(p.stem): p for p in
                         data_root.rglob(f'*{self.ann_suffix}')}
            prefixes = set(img_files.keys()) & set(ann_files.keys())
            # prefixes = prefixes - self.ignore_sample_indices

            # ann_infos = mmengine.load(
            #     next(data_root.rglob(f'*neu2coco_train.json')))['annotations']
            #
            # for info in ann_infos:
            #     prefix = int(Path(info.pop('file_name')).stem)
            #     if prefix in prefixes:
            #         data_list.append(
            #             dict(img_path=str(img_files[prefix]),
            #                  seg_map_path=str(ann_files[prefix]),
            #                  seg_fields=[], reduce_zero_label=False, **info)
            #         )
            # if get_dist_info()[0] == 0:
            #     mmengine.dump(dict(data_list=data_list), meta_root)
            #
            # print(data_list)
            #
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

    # def load_data_list(self):
    #     data_list = []
    #     img_dir = osp.join(self.data_root, 'images')
    #     ann_dir = osp.join(self.data_root, 'annotations')
    #
    #     meta_root = Path(self.meta_root)
    #     if meta_root.is_file():
    #         data_list = mmengine.load(meta_root)['data_list']
    #     else:
    #         _suffix_len = len(self.img_suffix)
    #         for img in fileio.list_dir_or_file(
    #                 dir_path=img_dir,
    #                 list_dir=False,
    #                 suffix=self.img_suffix,
    #                 recursive=True,
    #                 backend_args=self.backend_args):
    #             data_info = dict(img_path=osp.join(img_dir, img))
    #             seg_map = img[:-_suffix_len] + self.ann_suffix
    #             data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
    #             data_info['label_map'] = None
    #             data_info['reduce_zero_label'] = False
    #             data_info['seg_fields'] = []
    #             data_list.append(data_info)
    #         data_list = sorted(data_list, key=lambda x: x['img_path'])
    #
    #         if mmengine.dist.get_dist_info()[0] == 0:
    #             mmengine.dump(dict(data_list=data_list), meta_root)
    #     return data_list

    def get_data_info(self, idx):
        data_info = super().get_data_info(idx)
        data_info['dataset'] = 'neu-val'
        # data_info['metainfo']['sample_idx'] = idx
        return data_info