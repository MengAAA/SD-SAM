_base_ = ['../../_base_/optimizer_adamw_160k_1e5.py',
          '../../_base_/train_colaug_qhdataset_1024x1024.py',
          '../hqnextsam_vwv0_v3.py']
batch_size = 1
train_dataloader = dict(batch_size=batch_size, num_workers=batch_size)
model = dict(
    type='ClickMixSegmentorDecode_HQ',
    remove_backbone=False,
    init_cfg=dict(type='Pretrained',
                      checkpoint='your path to SAM pretrained weights'),  # SAM pretrained
    train_cfg=dict(
        interact_params = { 'qhdataset' : dict(gamma=0.8)} ,max_num_clicks = 24, )
)
