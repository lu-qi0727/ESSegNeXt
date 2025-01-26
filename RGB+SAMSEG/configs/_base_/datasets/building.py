# dataset settings
dataset_type = 'BuildingDataset'
# data_root = 'data/building_datasets'
data_root =r'E:\lq\mmsegmentation-5channel\data\building_datasets-UVA'
# data_root =r'E:\lq\mmsegmentation-main\data\building_datasets.UVA'
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    dict(type='LoadMultipleAnnotations'),
    # dict(
    #     type='RandomResize',
    #     scale=(2048, 512),
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    #dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    #dict(type='RandomFlip', prob=0.5),
    # # dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadTiffImageFromFile'),
    # dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type='LoadMultipleAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(

img_path=r'images/training', img_path2=r'boundary/training', seg_map_path=r'annotations/training'
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

img_path=r'images/validation', img_path2=r'boundary/validation', seg_map_path=r'annotations/validation'
            # img_path='val', seg_map_path='mask/val'
        ),
        # ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator

