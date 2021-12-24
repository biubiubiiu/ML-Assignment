import glob
import json
import os

import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from tqdm import tqdm

src_dir = '../fabric_data'
dst_dir = '../fabric_data_split'


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
PARTITION = 11

count = dict((c, []) for c in range(len(CLASSES)))
ssims = []
psnrs = []

json_files = glob.glob(os.path.join(
    src_dir, 'label_json/**/*.json'), recursive=True)
for json_path in tqdm(json_files, desc='Collect data info'):
    fh = open(json_path, mode='r')

    img_anns = json.load(fh)
    flaw_type = int(img_anns['flaw_type'])
    item_id = '/'.join((os.path.splitext(json_path)[0]).split(os.sep)[-2:])

    fabric_image_path = os.path.join(src_dir, 'trgt', item_id + '.jpg')
    template_image_path = os.path.join(src_dir, 'temp', item_id + '.jpg')

    fabric_image = cv2.imread(fabric_image_path)
    template_image = cv2.imread(template_image_path)

    fabric_image = cv2.resize(fabric_image, (400, 400))
    template_image = cv2.resize(template_image, (400, 400))

    ssim = compare_ssim(fabric_image, template_image, channel_axis=2)
    ssim = 1 - (1 + ssim) / 2
    ssims.append(ssim)

    psnr = compare_psnr(fabric_image, template_image)
    psnrs.append(psnr)

    fh.close()

ssims = np.array(ssims)
print('SSIM Mean: ', np.mean(ssims))
print('SSIM Median: ', np.median(ssims))
print('SSIM 25%: ', np.percentile(ssims, 25))
print('SSIM 75%: ', np.percentile(ssims, 75))

psnrs = np.array(psnrs)
print('PSNR Mean: ', np.mean(psnrs))
print('PSNR Median: ', np.median(psnrs))
print('PSNR 25%: ', np.percentile(psnrs, 25))
print('PSNR 75%: ', np.percentile(psnrs, 75))
print(len(psnrs[psnrs > 27.5]))

plt.hist(ssims, bins=np.arange(0, 1.05, 0.05))
plt.show()

plt.hist(psnrs)
plt.show()
