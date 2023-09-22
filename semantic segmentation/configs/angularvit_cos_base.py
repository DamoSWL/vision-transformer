_base_ = [
    '../_base_/models/upernet_angularvit.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]


checkpoint_file = None

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dim=[96,192,384,768],
        depth=[4,8,24,4], 
        num_heads=[1,2,4,8],
        n_window=[(9,8),(5,4),(3,2),(1,1)],
        kv_pooling=[0,0,0,0],
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        attn_type='cos',
        ),
        
    decode_head=dict(in_channels=[96,192,384,768], num_classes=150),

    auxiliary_head=dict(in_channels=384, num_classes=150))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
runner = dict(max_iters=320000)