# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Override palette, classes, and state dict keys

from copy import deepcopy

import matplotlib.pyplot as plt
import mmcv
import torch
from mmengine.config import Config

from mmseg.apis.checkpoint import (load_student_checkpoint,
                                   set_checkpoint_metadata)
from mmseg.datasets import build_dataset
from mmseg.datasets.collate import uda_collate
from mmseg.datasets.pipelines import Compose
from mmseg.engine.config import evaluation_dataset_config, student_config
from mmseg.models import build_segmentor
from mmseg.models.data_preprocessor import UDADataPreprocessor


def init_segmentor(config,
                   checkpoint=None,
                   device='cuda:0',
                   classes=None,
                   palette=None,
                   revise_checkpoint=[(r'^module\.', '')]):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config = deepcopy(config)
    model_cfg = student_config(config)
    model_cfg['pretrained'] = None
    model_cfg['train_cfg'] = None
    model = build_segmentor(model_cfg)
    loaded = load_student_checkpoint(model, checkpoint, revise_checkpoint) \
        if checkpoint is not None else None
    dataset = build_dataset(evaluation_dataset_config(config))
    set_checkpoint_metadata(model, loaded, dataset, classes, palette)
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_segmentor(model, img):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    pipeline = evaluation_dataset_config(cfg)['pipeline']
    if isinstance(img, str) and img.endswith('.npy'):
        data = dict(
            img_info=dict(filename=img), img_prefix=None, seg_fields=[])
        data = Compose(pipeline)(data)
    else:
        # Ordinary images/arrays retain the raw BGR image API.
        data = Compose([LoadImage()] + pipeline[1:])(dict(img=img))
    data = UDADataPreprocessor().to(device)(uda_collate([data]))

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def show_result_pyplot(model,
                       img,
                       result,
                       palette=None,
                       fig_size=(15, 10),
                       opacity=0.5,
                       title='',
                       block=True):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        block (bool): Whether to block the pyplot figure.
            Default is True.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(
        img, result, palette=palette, show=False, opacity=opacity)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.title(title)
    plt.tight_layout()
    plt.show(block=block)
