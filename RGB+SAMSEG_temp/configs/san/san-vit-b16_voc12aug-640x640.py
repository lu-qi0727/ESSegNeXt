_base_ = [
    '../_base_/models/san_vit-b16.py',
    '../_base_/datasets/building.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_test.py'
]
crop_size = (512, 512)

metainfo = dict(
    classes=('background', 'building'),
    palette=[[0, 0, 0], [128, 0, 0]])

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeShortestEdge', scale=crop_size, max_size=2560),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(
    batch_size=1, dataset=dict(metainfo=metainfo, pipeline=test_pipeline))
test_dataloader = val_dataloader

data_preprocessor = dict(
    mean=[122.7709, 116.7460, 104.0937],
    std=[68.5005, 66.6322, 70.3232],
    size_divisor=640,
    test_cfg=dict(size_divisor=32))
model = dict(
    data_preprocessor=data_preprocessor,
    # pretrained='pretrain/vit_base_patch16_224.pth',
    text_encoder=dict(dataset_name='BuildingDataset'),
    decode_head=dict(num_classes=2))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone

# optimizer
iters = 14000
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=14000,
        by_epoch=False)
]
# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
#     paramwise_cfg=dict(
#         custom_keys={
#             'pos_embed': dict(decay_mult=0.),
#             'cls_token': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))
#
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,
#         begin=1500,
#         end=160000,
#         by_epoch=False,
#     )
# ]
