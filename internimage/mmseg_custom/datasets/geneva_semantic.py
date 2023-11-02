from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class GenevaSemanticDataset(CustomDataset):

    CLASSES = ('Background', 'Borderline', 'Building', 'Unbuilt', 'Wall', 'Road', 'River')

    PALETTE = [[0, 0, 0], [255, 255, 255], [255, 115, 223], [211, 255, 190], [78, 78, 78], 
    [255, 255, 0], [190, 232, 255]]

    def __init__(self, **kwargs):
        super(GenevaSemanticDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
