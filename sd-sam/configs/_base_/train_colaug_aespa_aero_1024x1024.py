dataset_type = 'aespa_AeroDataset'
data_root = 'data/Aero/split_train_9'
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAeroAnnotations'),
    # dict(
    #    type='Resize',
    #    scale=(1024, 1024),
    #    keep_ratio=True),
    dict(type='Resize', scale=(1024, 1024), scale_factor=(0.75, 1.4), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='RandomRotate', prob=0.5 ,degree =(-20.0,20.0)),
    dict(type='ObjectSampler',
         max_num_merged_objects=1,
         min_area_ratio=0.0),
    dict(type='PhotoMetricDistortion'),
    dict(type='InterSegPackSegInputs')
]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     #dict(type='Resize', scale=(2048, 512), keep_ratio=True),
#     # add loading annotation after ``Resize`` because ground truth
#     # does not need to do resize data transform
#     dict(type='LoadAnnotations', reduce_zero_label=False),
#     dict(type='PackSegInputs')
# ]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
# tta_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=None),
#     dict(
#         type='TestTimeAug',
#         transforms=[
#             #[
#             #    dict(type='Resize', scale_factor=r, keep_ratio=True)
#             #    for r in img_ratios
#             #],
#             [
#                 dict(type='RandomFlip', prob=0., direction='horizontal'),
#                 dict(type='RandomFlip', prob=1., direction='horizontal')
#             ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
#         ])
# ]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    # reduce_zero_label=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # data_prefix=dict(
        #     img_path='images/training', seg_map_path='annotations/training'),
        pipeline=train_pipeline))

