_base_ = [
    '../_base_/models/upernet_convnext.py', '../_base_/datasets/building.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_test.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)


checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth'  # noqa

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmpretrain.ConvNeXt',
        arch='small',

        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.3,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=2,

out_channels=1,
        threshold=0.4,

    ),
    auxiliary_head=dict(in_channels=384, num_classes=2,out_channels=1,
        threshold=0.4,),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    },
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=100),
    dict(
        type='PolyLR',
        power=1.0,
        begin=100,
        end=50000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader


