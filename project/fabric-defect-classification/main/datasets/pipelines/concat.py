from ..builder import PIPELINES
import torch


@PIPELINES.register_module()
class Concat:
    """Concat two images with given keys.

    Args:
        keys (Sequence[str]): The images to be cropped.
        output_key (str): Key of the output
    """

    def __init__(self, keys, output_key='img'):
        self.keys = keys
        self.output_key = output_key

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        results[self.output_key] = torch.cat(
            [results[key] for key in self.keys], dim=0)
        return results
