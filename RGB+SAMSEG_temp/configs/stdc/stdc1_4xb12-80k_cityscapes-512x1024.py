_base_ = [
    '../_base_/models/stdc.py', '../_base_/datasets/building.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_test.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
# param_scheduler = [
#     dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=1400),
#     dict(
#         type='PolyLR',
#         eta_min=1e-4,
#         power=0.9,
#         begin=1400,
#         end=80000,
#         by_epoch=False,
#     )
# ]
train_dataloader = dict(batch_size=2, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
