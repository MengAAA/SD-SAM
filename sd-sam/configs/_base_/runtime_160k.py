_base_ = 'runtime_40k.py'
train_cfg = dict(max_iters=160000, val_interval=10000)
default_hooks = dict(checkpoint=dict(interval=10000))
