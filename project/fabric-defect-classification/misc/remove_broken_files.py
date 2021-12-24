import os
import cv2
import glob


broken_files = 0
shape_unmatched_files = 0
json_files = glob.glob('fabric_data/label_json/**/*.json', recursive=True)
for json_path in json_files:
    image_id = '/'.join((os.path.splitext(json_path)[0]).split(os.sep)[-2:])
    lq_path = os.path.join('fabric_data/trgt', image_id + '.jpg')
    gt_path = os.path.join('fabric_data/temp', image_id + '.jpg')

    if cv2.imread(lq_path) is None or cv2.imread(gt_path) is None:
        print(f'Broken: {image_id}')
        os.system(f'rm {json_path} {lq_path} {gt_path}')
        broken_files += 1
        continue
    if cv2.imread(lq_path).shape != cv2.imread(gt_path).shape:
        h1, w1, _ = cv2.imread(lq_path).shape
        h2, w2, _ = cv2.imread(gt_path).shape
        if not 0.5 <= h1 / h2 <= 2 or not 0.5 <= w1 / w2 <= 2:
            print(f'Unmatched shape: {image_id}: {cv2.imread(lq_path).shape}, {cv2.imread(gt_path).shape}')
            os.system(f'rm {json_path} {lq_path} {gt_path}')
            shape_unmatched_files += 1
            continue

print(f'remove {broken_files} broken files, {shape_unmatched_files} shape unmatched files')
