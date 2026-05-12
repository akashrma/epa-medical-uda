# Obtained from: https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License
# A copy of the license is available at resources/license_dacs

import kornia
import numpy as np
import torch
import torch.nn as nn


def strong_transform(param, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = one_mix(mask=param['mix'], data=data, target=target)
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target


def get_mean_std(img_metas, dev):
    mean = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=dev)
        for i in range(len(img_metas))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=dev)
        for i in range(len(img_metas))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)


def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)


def color_jitter(color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target


def get_class_masks(labels, class_mix_ignore=None):
    class_masks = []
    if class_mix_ignore is None:
        class_mix_ignore = []
    ignore = None
    if len(class_mix_ignore) > 0:
        ignore = torch.as_tensor(class_mix_ignore, device=labels.device)
    for label in labels:
        classes = torch.unique(labels)
        if ignore is not None:
            keep = torch.ones_like(classes, dtype=torch.bool)
            for ig in ignore:
                keep &= classes != ig
            classes = classes[keep]
        nclasses = classes.shape[0]
        if nclasses == 0:
            class_masks.append(torch.zeros_like(label).unsqueeze(0))
            continue
        class_choice = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        classes = classes[torch.as_tensor(class_choice,
                                          device=classes.device).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks


def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target


def _fda_low_freq_mutate(amp_src, amp_trg, L=0.1):
    if L <= 0:
        return amp_src
    _, _, h, w = amp_src.size()
    b = int(np.floor(min(h, w) * L))
    if b < 1:
        return amp_src
    amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]
    amp_src[:, :, 0:b, w - b:w] = amp_trg[:, :, 0:b, w - b:w]
    amp_src[:, :, h - b:h, 0:b] = amp_trg[:, :, h - b:h, 0:b]
    amp_src[:, :, h - b:h, w - b:w] = amp_trg[:, :, h - b:h, w - b:w]
    return amp_src


def _fda_is_complex(tensor):
    return hasattr(torch, 'is_complex') and torch.is_complex(tensor)


def _fda_fft2(img):
    if hasattr(torch, 'fft') and hasattr(torch.fft, 'fft2'):
        return torch.fft.fft2(img, dim=(-2, -1))
    return torch.rfft(img, signal_ndim=2, onesided=False)


def _fda_ifft2(fft_img, spatial_shape):
    if hasattr(torch, 'fft') and hasattr(torch.fft, 'ifft2'):
        return torch.fft.ifft2(fft_img, dim=(-2, -1)).real
    return torch.irfft(
        fft_img,
        signal_ndim=2,
        onesided=False,
        signal_sizes=spatial_shape)


def _fda_extract_amp_phase(fft_img):
    if _fda_is_complex(fft_img):
        amp = torch.abs(fft_img)
        pha = torch.angle(fft_img)
    else:
        amp = torch.sqrt(fft_img[..., 0]**2 + fft_img[..., 1]**2)
        pha = torch.atan2(fft_img[..., 1], fft_img[..., 0])
    return amp, pha


def _fda_polar(amp, pha, like_fft):
    if _fda_is_complex(like_fft):
        return torch.polar(amp, pha)
    real = amp * torch.cos(pha)
    imag = amp * torch.sin(pha)
    return torch.stack((real, imag), dim=-1)


def fda_source_to_target(src_img, trg_img, L=0.1):
    fft_src = _fda_fft2(src_img)
    fft_trg = _fda_fft2(trg_img)
    amp_src, pha_src = _fda_extract_amp_phase(fft_src)
    amp_trg, _ = _fda_extract_amp_phase(fft_trg)
    amp_src = _fda_low_freq_mutate(amp_src, amp_trg, L=L)
    fft_src = _fda_polar(amp_src, pha_src, fft_src)
    src_in_trg = _fda_ifft2(fft_src, spatial_shape=src_img.shape[-2:])
    return src_in_trg
