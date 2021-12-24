# fabric-defect-classification

ResNet-34 for fabric defect classification.

## Summary

- Classification of fabric defects
- Each fabric image has a corresponding template image and a rough bounding box of fabric area (which may not be accurate) availble
- 15 defect categories in total
- Uses a 34 layers ResNet here as it's good balance between performance and speed
- Fabric images and their corresponding template images are rescaled to the same size and concatenated together as the input of network
- Train from scratch

---

## Requirements

- Python 3.6+
- PyTorch 1.5+
- [mmcv](https://github.com/open-mmlab/mmcv)
- [mmclassification](https://github.com/open-mmlab/mmclassification)

(using mmclassification==0.18.0, mmcv==1.3.17 in this project)

## Datasets

- Download dataset from [google drive](https://drive.google.com/file/d/1glWVBxS1xZNj6yQgoe3Fbzvdk0PdZOlO/view?usp=sharing)
- Unzip data to `./fabric_data`
- Post processing

```python
# remove broken files
python misc/remove_broken_files.py

# split into 10-folds
python misc/train-test_split.py
```

## Training

```python
python train.py configs/resnet34_mixup_aug_balanced.py
```

## Testing

```python
python test.py configs/resnet34_mixup_aug_balanced.py work_dirs/resnet34_mixup_aug_balanced/epoch_270.pth --metrics accuracy precision recall
```
