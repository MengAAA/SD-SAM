_base_ = ['../../_base_/optimizer_adamw_160k_1e5_decoder0.1.py',
          '../../_base_/train_colaug_aespa_aero_1024x1024.py',
          '../qhsam_vwv0_noreduce_10_point_2_PSA.py']
batch_size = 1
train_dataloader = dict(batch_size=batch_size, num_workers=batch_size)
model = dict(
    type='ClickMixSegmentorRefine_HQ_v2_point',
    remove_backbone=False,
    init_cfg=dict(type='Pretrained',
                      checkpoint='your path to pretrained model'),  
    train_cfg=dict(
        interact_params={'aespa_aero': dict(gamma=0.8, refine_gamma=0.5)},
        expand_ratio_range=(1.0, 1.4),
        max_num_clicks=24)
)
