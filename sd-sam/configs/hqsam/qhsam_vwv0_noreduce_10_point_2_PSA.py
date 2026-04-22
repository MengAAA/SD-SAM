_base_ = './hqnextsam_vwv0_v3.py'
model = dict(
    type='ClickMixSegmentorRefine_HQ_v2',
    decode_head=dict(type='SAMDecoderForRefiner_QHvwv0_1_v6_tov7'),
    refine_head=dict(type='FocusRefiner_noreduce_noconv_andpoint_andca_andPSA',
                     embed_dim=256,
                     depth=12,  # 12
                     num_heads=4,  # 8
                     mlp_ratio=4.0,  # 4.0
                     window_size=16),
    refine_extra_params=dict(mode='single mask with token')
)