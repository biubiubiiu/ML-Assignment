import glob
import json
import os
import random

from tqdm import tqdm

src_dir = 'fabric_data'
dst_dir = 'fabric_data_partitions'


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


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
PARTITION = 10

count = dict((c, []) for c in range(len(CLASSES)))

json_files = glob.glob(os.path.join(
    src_dir, 'label_json/**/*.json'), recursive=True)
for json_path in tqdm(json_files, desc='Collect data info'):
    fh = open(json_path, mode='r')

    img_anns = json.load(fh)
    flaw_type = int(img_anns['flaw_type'])
    item_id = '/'.join((os.path.splitext(json_path)[0]).split(os.sep)[-2:])
    count[flaw_type].append(item_id)

    fh.close()

data = []

for c, items in count.items():
    print(CLASSES[c], ':', len(items))

for c, items in count.items():
    if len(items) < PARTITION:
        print('====> Samples too few, skip class %s: %s' % (c, CLASSES[c]))
        continue

    random.shuffle(items)
    partition = len(items) // PARTITION
    assert partition > 0
    data.append(items[:partition*PARTITION])

data_size = sum(len(dt) for dt in data)
print('Size of data: ', data_size)

pbar = tqdm(total=data_size, desc='Split data')
for ci, items in enumerate(data):
    size = len(items)
    assert size % PARTITION == 0
    chunk_size = len(items) // PARTITION
    for i, chunk in enumerate(chunks(items, chunk_size)):
        dst_root = os.path.join(dst_dir, str(i))
        for id in chunk:
            json_path = os.path.join(src_dir, 'label_json', id + '.json')
            fabric_image_path = os.path.join(src_dir, 'trgt', id + '.jpg')
            template_image_path = os.path.join(src_dir, 'temp', id + '.jpg')

            dst_json_path = os.path.join(dst_root, 'label_json', id + '.json')
            dst_fabric_image_path = os.path.join(dst_root, 'trgt', id + '.jpg')
            dst_template_image_path = os.path.join(
                dst_root, 'temp', id + '.jpg')

            os.makedirs(os.path.split(dst_json_path)[0], exist_ok=True)
            os.makedirs(os.path.split(dst_fabric_image_path)[0], exist_ok=True)
            os.makedirs(os.path.split(dst_template_image_path)[0], exist_ok=True)

            os.system(f'cp -r {json_path} {dst_json_path}')
            os.system(f'cp -r {fabric_image_path} {dst_fabric_image_path}')
            os.system(f'cp -r {template_image_path} {dst_template_image_path}')

            pbar.update(1)
