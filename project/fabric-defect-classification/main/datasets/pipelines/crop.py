from ..builder import PIPELINES


@PIPELINES.register_module()
class CropBoundingBoxArea:
    """Crop bounding box area from paired data.

    Args:
        keys (Sequence[str]): The images to be cropped.
    """

    def __init__(self, keys):
        self.keys = keys

    def _crop(self, data, x_offset, y_offset, crop_w, crop_h):
        crop_bbox = [x_offset, y_offset, crop_w, crop_h]
        data_ = data[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w,
                     ...]
        return data_, crop_bbox

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        data_h, data_w = results[self.keys[0]].shape[:2]
        crop_h = results['bbox']['y1'] - results['bbox']['y0']
        crop_w = results['bbox']['x1'] - results['bbox']['x0']
        x_offset = results['bbox']['x0']
        y_offset = results['bbox']['y0']

        assert crop_h <= data_h and crop_w <= data_w

        for k in self.keys:
            # In fixed crop for paired images, sizes should be the same
            if (results[k].shape[0] != data_h
                    or results[k].shape[1] != data_w):
                raise ValueError(
                    'The sizes of paired images should be the same. Expected '
                    f'({data_h}, {data_w}), but got ({results[k].shape[0]}, '
                    f'{results[k].shape[1]}).')
            data_, crop_bbox = self._crop(results[k], x_offset, y_offset,
                                          crop_w, crop_h)
            results[k] = data_
            results[k + '_crop_bbox'] = crop_bbox
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'keys={self.keys}')
        return repr_str
