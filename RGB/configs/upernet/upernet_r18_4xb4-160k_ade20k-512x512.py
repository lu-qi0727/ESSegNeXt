_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/building.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_test.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet18_v1c',
    # backbone=dict(depth=50),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))
