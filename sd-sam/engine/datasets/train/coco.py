import warnings
from pathlib import Path

import mmengine
import numpy as np
from mmengine.dist import get_dist_info
from mmseg.datasets.transforms import LoadAnnotations
from mmseg.registry import TRANSFORMS, DATASETS

from ..base import BaseInterSegDataset


@TRANSFORMS.register_module()
class LoadCOCOPanopticAnnotations(LoadAnnotations):

    def _load_seg_map(self, results):
        # print("results keys in LoadNEUAnnotations:", results.keys())
        super(LoadCOCOPanopticAnnotations, self)._load_seg_map(results)
        # print("results keys in LoadNEUAnnotations222222222222:", results.keys())
        gt_seg_map = results.pop('gt_seg_map')
        segments_info = results.pop('segments_info')
        gt_seg_map = (gt_seg_map[..., 0] * (1 << 16) +
                      gt_seg_map[..., 1] * (1 << 8) +
                      gt_seg_map[..., 2] * (1 << 0))
        results['gt_seg_map'] = np.zeros_like(gt_seg_map).astype(np.uint8)
        results['segments_info'] = []
        for i, info in enumerate(segments_info, 1):
            results['gt_seg_map'][gt_seg_map == info['id']] = i
            results['segments_info'].append(dict(id=i))

#@TRANSFORMS.register_module()：将这个类注册到 mmseg 的变换注册表中，这样在配置文件里就可以使用这个变换了。
# LoadCOCOPanopticAnnotations 继承自 LoadAnnotations，并重写了 _load_seg_map 方法。
# super(LoadCOCOPanopticAnnotations, self)._load_seg_map(results)：调用父类的 _load_seg_map 方法来加载标注信息。
# gt_seg_map 和 segments_info 从 results 中弹出。
# gt_seg_map 是一个三维数组（通常是 RGB 图像），通过位运算将其转换为一维的 ID 数组。这是因为 COCO 全景分割标注中，每个像素的 RGB 值组合起来表示一个唯一的实例 ID。
# 创建一个与 gt_seg_map 形状相同的全零数组，并将其作为新的 gt_seg_map。
# 遍历 segments_info，将 gt_seg_map 中对应 ID 的像素值设置为当前索引 i，并更新 segments_info 中的 ID。这样做的目的是重新编号实例 ID，方便后续处理。

@DATASETS.register_module()
class COCOPanopticDataset(BaseInterSegDataset):

    default_meta_root = 'data/meta-info/coco.json'
    default_ignore_file = 'data/meta-info/coco_invalid_image_names.json'

    def __init__(self,
                 data_root,
                 pipeline,
                 ignore_file=None,
                 serialize_data=True,
                 lazy_init=False,
                 max_refetch=1000):
        if ignore_file is None:
            ignore_file = self.default_ignore_file
        if Path(ignore_file).is_file():
            self.ignore_sample_indices = \
                set(map(int, mmengine.load(ignore_file)['train']))
        else:
            warnings.warn(f'Not found ignore_file {ignore_file}')
            self.ignore_names = set()
        super(COCOPanopticDataset, self).__init__(
            data_root=data_root,
            pipeline=pipeline,
            img_suffix='.jpg',
            ann_suffix='.png',
            filter_cfg=None,
            serialize_data=serialize_data,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            ignore_index=255,
            backend_args=None)
# 初始化数据集类的参数，包括数据根目录、数据处理管道、忽略文件等。
# 如果 ignore_file 未提供，则使用默认的忽略文件。
# 如果忽略文件存在，则加载其中的训练样本索引，并存储在 self.ignore_sample_indices 中。
# 如果忽略文件不存在，则发出警告，并将 self.ignore_names 初始化为空集合。
# 调用父类的 __init__ 方法进行初始化，传入相关参数。
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
            # ann_files = {int(p.stem): p for p in data_root.rglob(f'*{self.ann_suffix}') if p.stem.isdigit()}
            prefixes = set(img_files.keys()) & set(ann_files.keys())
            prefixes = prefixes - self.ignore_sample_indices

            ann_infos = mmengine.load(
                next(data_root.rglob(f'*panoptic_train2017.json'))
            )['annotations']
            for info in ann_infos:
                prefix = int(Path(info.pop('file_name')).stem)
                if prefix in prefixes:
                    data_list.append(
                        dict(img_path=str(img_files[prefix]),
                             seg_map_path=str(ann_files[prefix]),
                             seg_fields=[], reduce_zero_label=False, **info)
                    )
            if get_dist_info()[0] == 0:
                mmengine.dump(dict(data_list=data_list), meta_root)

        # print('\n data list   COCO dataset  !!!!!!!! \n ', data_list)
        return data_list
# 该方法用于加载数据集的所有数据信息。
# 如果元信息文件存在，则直接从文件中加载数据列表。
# 如果元信息文件不存在，则进行以下操作：
# 遍历数据根目录，查找所有的图像文件和标注文件，并将其文件名的前缀（转换为整数）作为键存储在字典中。
# 取图像文件和标注文件前缀的交集，并排除需要忽略的样本。
# 加载 COCO 全景分割的标注信息（从 panoptic_train2017.json 文件中），并遍历每个标注信息。
# 如果标注信息的前缀在可用前缀集合中，则将图像路径、标注路径等信息添加到数据列表中。
# 如果是主进程（get_dist_info()[0] == 0），则将数据列表保存到元信息文件中，以便下次加载时可以直接读取。

    def get_data_info(self, idx):
        data_info = super(BaseInterSegDataset, self).get_data_info(idx)
        data_info['dataset'] = 'coco'
        return data_info

# 调用父类的 get_data_info 方法获取数据信息。
# 添加 dataset 字段，并将其值设置为 'coco'，用于标识数据集的名称。
# 返回更新后的数据信息。