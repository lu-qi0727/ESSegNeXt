# dataset settings
dataset_type = 'BuildingDataset'
# data_root = 'data/building_datasets'
data_root =r'E:\lq\mmsegmentation-3channel-order\data\building_datasets-UVA'
# data_root =r'E:\lq\mmsegmentation-main\data\building_datasets.UVA'
crop_size = (512, 512)
# data_root =r'C:\Users\A\Desktop\ourdatasets'
# crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(
    #     type='RandomResize',
    #     scale=(2048, 512),
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
        #     [
        #         dict(type='Resize', scale_factor=r, keep_ratio=True)
        #         for r in img_ratios
        #     ],
        #     [
        #         dict(type='RandomFlip', prob=0., direction='horizontal'),
        #         dict(type='RandomFlip', prob=1., direction='horizontal')
        #     ],
        [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(

img_path=r'images/training', seg_map_path=r'annotations/training'
            # img_path='train', seg_map_path='mask/train'
        ),
        # ann_file='ImageSets/Segmentation/train.txt',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(

img_path=r'images/validation', seg_map_path=r'annotations/validation'
            # img_path='val', seg_map_path='mask/val'
        ),
        # ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator















# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', reduce_zero_label=False),
#     dict(
#         type='RandomResize',
#         scale=(2048, 512),
#         ratio_range=(0.5, 2.0),
#         keep_ratio=True),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='PackSegInputs')
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', scale=(2048, 512), keep_ratio=True),
#     # add loading annotation after ``Resize`` because ground truth
#     # does not need to do resize data transform
#     dict(type='LoadAnnotations', reduce_zero_label=True),
#     dict(type='PackSegInputs')
# ]
# img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
# tta_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=None),
#     dict(
#         type='TestTimeAug',
#         transforms=[
#             [
#                 dict(type='Resize', scale_factor=r, keep_ratio=True)
#                 for r in img_ratios
#             ],
#             [
#                 dict(type='RandomFlip', prob=0., direction='horizontal'),
#                 dict(type='RandomFlip', prob=1., direction='horizontal')
#             ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
#         ])
# ]
# train_dataloader = dict(
#     batch_size=4,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='InfiniteSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(
#             img_path='images/training', seg_map_path='annotations/training'),
#         pipeline=train_pipeline))
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(
#             img_path='images/validation',
#             seg_map_path='annotations/validation'),
#         pipeline=test_pipeline))
# test_dataloader = val_dataloader
#
# val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
# test_evaluator = val_evaluator
# # #
# # #
# #
#
#
#
#
# # # dataset settings
# # dataset_type = 'BuildingDataset'
# # data_root = 'data/building_datasets'
# # crop_size = (512, 512)
# # train_pipeline = [
# #     dict(type='LoadImageFromFile'),
# #     dict(type='LoadAnnotations', reduce_zero_label=True),
# #     dict(
# #         type='RandomResize',
# #         scale=(512, 512),
# #         ratio_range=(0.5, 2.0),
# #         keep_ratio=True),
# #     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
# #     dict(type='RandomFlip', prob=0.5),
# #     dict(type='PhotoMetricDistortion'),
# #     dict(type='PackSegInputs')
# # ]
# # test_pipeline = [
# #     dict(type='LoadImageFromFile'),
# #     dict(type='Resize', scale=(512, 512), keep_ratio=True),
# #     # add loading annotation after ``Resize`` because ground truth
# #     # does not need to do resize data transform
# #     dict(type='LoadAnnotations', reduce_zero_label=True),
# #     dict(type='PackSegInputs')
# # ]
# # img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
# # tta_pipeline = [
# #     dict(type='LoadImageFromFile', backend_args=None),
# #     dict(
# #         type='TestTimeAug',
# #         transforms=[
# #             [
# #                 dict(type='Resize', scale_factor=r, keep_ratio=True)
# #                 for r in img_ratios
# #             ],
# #             [
# #                 dict(type='RandomFlip', prob=0., direction='horizontal'),
# #                 dict(type='RandomFlip', prob=1., direction='horizontal')
# #             ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
# #         ])
# # ]
# # train_dataloader = dict(
# #     batch_size=4,
# #     num_workers=4,
# #     persistent_workers=True,
# #     sampler=dict(type='InfiniteSampler', shuffle=True),
# #     dataset=dict(
# #         type=dataset_type,
# #         data_root=data_root,
# #         data_prefix=dict(
# #             img_path='img_dir/train', seg_map_path='ann_dir/train'),
# #         pipeline=train_pipeline))
# # val_dataloader = dict(
# #     batch_size=1,
# #     num_workers=4,
# #     persistent_workers=True,
# #     sampler=dict(type='DefaultSampler', shuffle=False),
# #     dataset=dict(
# #         type=dataset_type,
# #         data_root=data_root,
# #         data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
# #         pipeline=test_pipeline))
# # test_dataloader = val_dataloader
# #
# # val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
# # test_evaluator = val_evaluator
