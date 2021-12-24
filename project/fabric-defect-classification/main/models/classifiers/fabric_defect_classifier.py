import copy
import warnings
from collections import OrderedDict

import mmcv
import torch
import torch.distributed as dist
from mmcls.core.visualization import imshow_infos
from mmcls.models.utils.augment import Augments
from mmcv.runner import BaseModule

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck


@CLASSIFIERS.register_module()
class FabricDefectClassifier(BaseModule):

    def __init__(
        self,
        backbone,
        neck=None,
        head=None,
        pretrained=None,
        train_cfg=None,
        init_cfg=None
    ):
        super(FabricDefectClassifier, self).__init__(init_cfg)

        if pretrained is not None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        return_tuple = backbone.pop('return_tuple', True)
        self.backbone = build_backbone(backbone)
        if return_tuple is False:
            warnings.warn(
                'The `return_tuple` is a temporary arg, we will force to '
                'return tuple in the future. Please handle tuple in your '
                'custom neck or head.', DeprecationWarning)
        self.return_tuple = return_tuple

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)
            else:
                # Considering BC-breaking
                mixup_cfg = train_cfg.get('mixup', None)
                cutmix_cfg = train_cfg.get('cutmix', None)
                assert mixup_cfg is None or cutmix_cfg is None, \
                    'If mixup and cutmix are set simultaneously,' \
                    'use augments instead.'
                if mixup_cfg is not None:
                    warnings.warn('The mixup attribute will be deprecated. '
                                  'Please use augments instead.')
                    cfg = copy.deepcopy(mixup_cfg)
                    cfg['type'] = 'BatchMixup'
                    # In the previous version, mixup_prob is always 1.0.
                    cfg['prob'] = 1.0
                    self.augments = Augments(cfg)
                if cutmix_cfg is not None:
                    warnings.warn('The cutmix attribute will be deprecated. '
                                  'Please use augments instead.')
                    cfg = copy.deepcopy(cutmix_cfg)
                    cutmix_prob = cfg.pop('cutmix_prob')
                    cfg['type'] = 'BatchCutMix'
                    cfg['prob'] = cutmix_prob
                    self.augments = Augments(cfg)

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self):
        return hasattr(self, 'head') and self.head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck."""
        x = self.backbone(img)
        if self.return_tuple:
            if not isinstance(x, tuple):
                x = (x, )
        else:
            if isinstance(x, tuple):
                x = x[-1]
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        x = self.extract_feat(img)

        losses = dict()
        try:
            loss = self.head.forward_train(x, gt_label)
        except TypeError as e:
            if 'not tuple' in str(e) and self.return_tuple:
                return TypeError(
                    'Seems the head cannot handle tuple input. We have '
                    'changed all backbones\' output to a tuple. Please '
                    'update your custom head\'s forward function. '
                    'Temporarily, you can set "return_tuple=False" in '
                    'your backbone config to disable this feature.')
            raise e

        losses.update(loss)

        return losses

    def forward_test(self, imgs, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
        """
        if isinstance(imgs, torch.Tensor):
            imgs = [imgs]
        for var, name in [(imgs, 'imgs')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        if len(imgs) == 1:
            return self.simple_test(imgs[0], **kwargs)
        else:
            raise NotImplementedError('aug_test has not been implemented')

    def forward(self, img, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_meta are single-nested (i.e. Tensor and
        List[dict]), and when `resturn_loss=False`, img and img_meta should be
        double nested (i.e.  List[Tensor], List[List[dict]]), with the outer
        list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, **kwargs)
        else:
            return self.forward_test(img, **kwargs)

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer=None, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict, optional): The
                optimizer of runner is passed to ``train_step()``. This
                argument is unused and reserved.

        Returns:
            dict: Dict of outputs. The following fields are contained.
                - loss (torch.Tensor): A tensor for back propagation, which \
                    can be a weighted sum of multiple losses.
                - log_vars (dict): Dict contains all the variables to be sent \
                    to the logger.
                - num_samples (int): Indicates the batch size (when the model \
                    is DDP, it means the batch size on each GPU), which is \
                    used for averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs

    def val_step(self, data, optimizer=None, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict, optional): The
                optimizer of runner is passed to ``train_step()``. This
                argument is unused and reserved.

        Returns:
            dict: Dict of outputs. The following fields are contained.
                - loss (torch.Tensor): A tensor for back propagation, which \
                    can be a weighted sum of multiple losses.
                - log_vars (dict): Dict contains all the variables to be sent \
                    to the logger.
                - num_samples (int): Indicates the batch size (when the model \
                    is DDP, it means the batch size on each GPU), which is \
                    used for averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs

    def show_result(self,
                    img,
                    result,
                    text_color='white',
                    font_scale=0.5,
                    row_width=20,
                    show=False,
                    fig_size=(15, 10),
                    win_name='',
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or ndarray): The image to be displayed.
            result (dict): The classification results to draw over `img`.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            font_scale (float): Font scales of texts.
            row_width (int): width between each row of results on the image.
            show (bool): Whether to show the image.
                Default: False.
            fig_size (tuple): Image show figure size. Defaults to (15, 10).
            win_name (str): The window name.
            wait_time (int): How many seconds to display the image.
                Defaults to 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (ndarray): Image with overlaid results.
        """
        img = mmcv.imread(img)
        img = img.copy()

        img = imshow_infos(
            img,
            result,
            text_color=text_color,
            font_size=int(font_scale * 50),
            row_width=row_width,
            win_name=win_name,
            show=show,
            fig_size=fig_size,
            wait_time=wait_time,
            out_file=out_file)

        return img

    def simple_test(self, img, img_metas):
        """Test without augmentation."""
        x = self.extract_feat(img)

        try:
            res = self.head.simple_test(x)
        except TypeError as e:
            if 'not tuple' in str(e) and self.return_tuple:
                return TypeError(
                    'Seems the head cannot handle tuple input. We have '
                    'changed all backbones\' output to a tuple. Please '
                    'update your custom head\'s forward function. '
                    'Temporarily, you can set "return_tuple=False" in '
                    'your backbone config to disable this feature.')
            raise e

        return res
