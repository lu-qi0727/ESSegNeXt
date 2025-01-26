# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class BuildingDataset(BaseSegDataset):
    """ISPRS Potsdam dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('background', 'building'),
        palette=[[0, 0, 0], [255, 255, 255]])

    # METAINFO = dict(
    #     classes=('building', 'road', "pavement", "vegetation", "bare soil", "water"),
    #     palette=[[255, 0, 0], [255, 255, 0], [192, 192, 0], [0, 255, 0], [128, 128, 128], [0, 0, 255]])

    def __init__(self,
                 img_suffix='.tif',
                 # img_suffix='.png',
                 seg_map_suffix='.tif',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
