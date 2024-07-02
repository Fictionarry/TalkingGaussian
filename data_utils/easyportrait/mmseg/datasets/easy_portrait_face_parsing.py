import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class EasyPortraitFPDataset(CustomDataset):
    """EasyPortraitFPDataset dataset.
    
    In segmentation map annotation for EasyPortrait, 0 stands for background,
    which is included in 9 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    
    CLASSES = ('background', 'skin',
               'left brow', 'right brow', 'left eye',
               'right eye', 'lips', 'teeth')
    
    PALETTE = [[0, 0, 0], [160, 221, 255],
               [130, 106, 237], [200, 121, 255], [255, 183, 255],
               [0, 144, 193], [113, 137, 255], [230, 232, 230]]
    
    def __init__(self, **kwargs):
        super(EasyPortraitFPDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)
        
@DATASETS.register_module()
class EasyPortraitFPDatasetCross(CustomDataset):
    """EasyPortraitFPDatasetCross dataset.
    
    In segmentation map annotation for EasyPortrait, 0 stands for background,
    which is included in 9 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
        
    CLASSES = ('background', 'left brow', 'right brow', 'left eye', 'right eye', 'lips')
    PALETTE = [[0, 0, 0], [160, 221, 255],
               [130, 106, 237], [200, 121, 255], [255, 183, 255],
               [0, 144, 193]]
    
    def __init__(self, **kwargs):
        super(EasyPortraitFPDatasetCross, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)