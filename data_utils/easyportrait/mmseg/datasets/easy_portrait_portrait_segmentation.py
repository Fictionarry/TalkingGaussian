import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class EasyPortraitPSDataset(CustomDataset):
    """EasyPortrait dataset.
    
    In segmentation map annotation for EasyPortrait, 0 stands for background,
    which is included in 9 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    
    CLASSES = ('background', 'person')
    
    PALETTE = [[0, 0, 0], [160, 221, 255]]
    
    def __init__(self, **kwargs):
        super(EasyPortraitPSDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)