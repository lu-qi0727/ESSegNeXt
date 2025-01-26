_base_ = [
    '../_base_/models/pspnet_unet_s5-d16.py', '../_base_/datasets/building.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_test.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(crop_size=(512, 512), stride=(170, 170)))
