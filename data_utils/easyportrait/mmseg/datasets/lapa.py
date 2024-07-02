import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class LaPaDataset(CustomDataset):
    """EasyPortrait dataset.
    
    In segmentation map annotation for LaPa, 0 stands for background,
    which is included in 11 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    
    CLASSES = ('background', 'skin', 'left eyebrow',
               'right eyebrow', 'left eye', 'right eye',
               'nose', 'upper lip', 'inner mouth', 'lower lip', 'hair')
    
    PALETTE = [[0, 0, 0], [0, 153, 255], [102, 255, 153],
               [0, 204, 153], [255, 255, 102], [255, 255, 204],
               [255, 153, 0], [255, 102, 255], [102, 0, 51],
               [255, 204, 255], [255, 0, 102]]
    
    def __init__(self, **kwargs):
        super(EasyPortraitDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)