_base_ = 'runtime_160k.py'
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer= dict(type='AdamW',
                   lr=1e-5,
                   betas=(0.9, 0.999),
                   weight_decay=0.05,
                   ),
    paramwise_cfg = dict(custom_keys={'decode_head': dict(lr_mult=0.1, decay_mult=1.0),
                                      'refine_head': dict(lr_mult=10, decay_mult=1.0)
                                      })
)
param_scheduler = [dict(type='LinearLR',
                        start_factor=1e-6,
                        by_epoch=False, begin=0, end=3000),
                   dict(type='PolyLR',
                        eta_min=0.0,
                        power=1.0,
                        begin=3000,
                        end=160000,
                        by_epoch=False)]
