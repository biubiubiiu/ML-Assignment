import glob
import json
import os

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class FabricData(BaseDataset):

    CLASSES = [
        'unknown',
        'escape_printing',
        'clogging_screen',
        'broken_hole',
        'toe_closing_defects',
        'water_stain',
        'smudginess',
        'white_stripe',
        'hazy_printing',
        'billet_defects',
        'trachoma',
        'color_smear',
        'crease',
        'false_positive',
        'no_alignment'
    ]

    TEMPLATE_PATH = 'temp'
    DEFECT_PATH = 'trgt'

    def __init__(
        self,
        path,
        partitions,
        pipeline,
        data_prefix='',
        classes=None,
        ann_file=None,
        test_mode=False
    ):
        self.path = path
        self.partitions = partitions
        super(FabricData, self).__init__(data_prefix,
                                         pipeline, classes, ann_file, test_mode)

    def load_annotations(self):
        data_infos = []

        for partition in self.partitions:
            data_root = os.path.join(self.path, str(partition))
            json_files = glob.glob(os.path.join(data_root, 'label_json/**/*.json'), recursive=True)
            for json_path in json_files:
                fh = open(json_path, mode='r')
                img_anns = json.load(fh)

                info = dict()
                info['gt_label'] = img_anns['flaw_type']

                image_id = '/'.join((os.path.splitext(json_path)
                                    [0]).split(os.sep)[-2:])
                info['img_id'] = image_id
                info['lq_path'] = os.path.join(
                    data_root, self.DEFECT_PATH, image_id + '.jpg')
                info['gt_path'] = os.path.join(
                    data_root, self.TEMPLATE_PATH, image_id + '.jpg')

                # NOTES seems there's only one bbox in each data
                info['bbox'] = img_anns['bbox']

                data_infos.append(info)
                fh.close()

        return data_infos
