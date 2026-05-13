# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Delete tensors after usage to free GPU memory
# - Add HRDA debug visualizations
# - Support ImageNet feature distance for LR and HR predictions of HRDA
# - Add masked image consistency
# - Update debug image system
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn import functional as F
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, HRDAEncoderDecoder, build_segmentor
from mmseg.models.segmentors.hrda_encoder_decoder import crop
from mmseg.models.uda.masking_consistency_module import \
    MaskingConsistencyModule
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std,
                                                strong_transform,
                                                fda_source_to_target)
from mmseg.models.utils.etf_prototypes import (generate_etf_class_prototypes,
                                               get_default_etf_dir,
                                               load_etf_class_prototypes)
from mmseg.models.utils.visualization import prepare_debug_out, subplotimg
from mmseg.utils.utils import downscale_label_ratio


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.source_only = cfg['source_only']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        if self.mix in (None, 'none'):
            self.mix = None
        self.class_mix_ignore = cfg.get('class_mix_ignore', None)
        self.mix_start_iter = cfg.get('mix_start_iter', 0)
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.mask_mode = cfg['mask_mode']
        self.enable_masking = self.mask_mode is not None
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix in (None, 'class')
        self.fda_source_to_target = cfg.get('fda_source_to_target', False)
        self.fda_source_prob = cfg.get('fda_source_prob', 0.0)
        self.fda_L = cfg.get('fda_L', 0.1)
        self.fda_in_denorm = cfg.get('fda_in_denorm', True)

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        self.etf_contrastive_lambda = cfg.get('etf_contrastive_lambda', 0.0)
        self.etf_contrastive_temperature = cfg.get(
            'etf_contrastive_temperature', 0.07)
        self.enable_etf_contrastive = self.etf_contrastive_lambda > 0
        self.etf_contrastive_target_lambda = cfg.get(
            'etf_contrastive_target_lambda', 0.0)
        self.etf_contrastive_target_temperature = cfg.get(
            'etf_contrastive_target_temperature',
            self.etf_contrastive_temperature)
        self.enable_etf_contrastive_target = \
            self.etf_contrastive_target_lambda > 0
        self.prototype_mode = cfg.get('prototype_mode', 'none')
        self.dynamic_proto_momentum = cfg.get('dynamic_proto_momentum', 0.99)
        self.dynamic_proto_min_pixels = cfg.get('dynamic_proto_min_pixels',
                                                16)
        self.dynamic_proto_source_lambda = cfg.get(
            'dynamic_proto_source_lambda', 0.0)
        self.dynamic_proto_target_lambda = cfg.get(
            'dynamic_proto_target_lambda', 0.0)
        self.dynamic_proto_target_start_iter = cfg.get(
            'dynamic_proto_target_start_iter', self.mix_start_iter)
        self.dynamic_proto_temperature = cfg.get(
            'dynamic_proto_temperature', 0.07)
        self.dynamic_proto_target_temperature = cfg.get(
            'dynamic_proto_target_temperature',
            self.dynamic_proto_temperature)
        self.dynamic_proto_ignore_background = cfg.get(
            'dynamic_proto_ignore_background', False)
        self.enable_dynamic_proto = self.prototype_mode == 'source_ema' and (
            self.dynamic_proto_source_lambda > 0
            or self.dynamic_proto_target_lambda > 0)
        self.enable_dynamic_proto_source = self.enable_dynamic_proto and \
            self.dynamic_proto_source_lambda > 0
        self.enable_dynamic_proto_target = self.enable_dynamic_proto and \
            self.dynamic_proto_target_lambda > 0
        self.dynamic_proto_feature_dim = cfg.get('dynamic_proto_feature_dim',
                                                 None)
        decode_head_cfg = cfg['model'].get('decode_head', {})
        if self.dynamic_proto_feature_dim is None:
            decoder_params = decode_head_cfg.get('decoder_params', {})
            self.dynamic_proto_feature_dim = decoder_params.get('embed_dim')
        if self.dynamic_proto_feature_dim is None:
            self.dynamic_proto_feature_dim = decode_head_cfg.get('channels')
        if self.enable_dynamic_proto and self.dynamic_proto_feature_dim:
            self.register_buffer(
                'source_prototypes',
                torch.zeros((self.num_classes,
                             int(self.dynamic_proto_feature_dim))))
            self.register_buffer('source_proto_valid',
                                 torch.zeros((self.num_classes, ),
                                             dtype=torch.bool))
            self.register_buffer('source_proto_seen',
                                 torch.zeros((self.num_classes, ),
                                             dtype=torch.long))
        else:
            self.register_buffer('source_prototypes', torch.empty(0))
            self.register_buffer('source_proto_valid',
                                 torch.empty(0, dtype=torch.bool))
            self.register_buffer('source_proto_seen',
                                 torch.empty(0, dtype=torch.long))
        ema_cfg = deepcopy(cfg['model'])
        if not self.source_only:
            self.ema_model = build_segmentor(ema_cfg)
        self.mic = None
        if self.enable_masking:
            self.mic = MaskingConsistencyModule(require_teacher=False, cfg=cfg)
        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        self.etf_prototype_dir = get_default_etf_dir()
        self.etf_prototypes = None
        self.source_class_means = None

    def _maybe_apply_fda(self, img, target_img, mean, std):
        if not self.fda_source_to_target:
            return img
        if target_img is None or self.fda_source_prob <= 0:
            return img
        if random.random() >= self.fda_source_prob:
            return img
        with torch.no_grad():
            if self.fda_in_denorm:
                src = denorm(img, mean, std)
                trg = denorm(target_img, mean, std)
                src = fda_source_to_target(src, trg, L=self.fda_L)
                src = torch.clamp(src, 0.0, 1.0)
                src = src.mul(255.0).sub(mean).div(std)
            else:
                src = fda_source_to_target(img, target_img, L=self.fda_L)
        return src

    def _compute_class_feature_means(self, feat_map, gt_semantic_seg):
        num_classes = self.num_classes
        if gt_semantic_seg.dim() == 3:
            gt_semantic_seg = gt_semantic_seg.unsqueeze(1)
        if gt_semantic_seg.shape[2:] != feat_map.shape[2:]:
            gt_resized = F.interpolate(
                gt_semantic_seg.float(),
                size=feat_map.shape[2:],
                mode='nearest').long()
        else:
            gt_resized = gt_semantic_seg
        gt_flat = gt_resized.squeeze(1).reshape(-1)
        feat_flat = feat_map.permute(0, 2, 3, 1).reshape(-1,
                                                         feat_map.shape[1])

        means = torch.zeros((num_classes, feat_map.shape[1]),
                            device=feat_map.device)
        for cls in range(num_classes):
            mask = gt_flat == cls
            if mask.any():
                means[cls] = feat_flat[mask].mean(dim=0)
        return means

    def _compute_class_feature_means_and_mask(self, feat_map,
                                              gt_semantic_seg):
        num_classes = self.num_classes
        ignore_index = self.get_model().decode_head.ignore_index
        if gt_semantic_seg.dim() == 3:
            gt_semantic_seg = gt_semantic_seg.unsqueeze(1)
        if gt_semantic_seg.shape[2:] != feat_map.shape[2:]:
            gt_resized = F.interpolate(
                gt_semantic_seg.float(),
                size=feat_map.shape[2:],
                mode='nearest').long()
        else:
            gt_resized = gt_semantic_seg
        gt_flat = gt_resized.squeeze(1).reshape(-1)
        feat_flat = feat_map.permute(0, 2, 3, 1).reshape(-1,
                                                         feat_map.shape[1])

        means = torch.zeros((num_classes, feat_map.shape[1]),
                            device=feat_map.device)
        present = torch.zeros((num_classes, ), device=feat_map.device,
                              dtype=torch.bool)
        for cls in range(num_classes):
            if cls == ignore_index:
                continue
            mask = gt_flat == cls
            if mask.any():
                means[cls] = feat_flat[mask].mean(dim=0)
                present[cls] = True
        return means, present

    def _compute_class_feature_means_mask_counts(self,
                                                 feat_map,
                                                 gt_semantic_seg,
                                                 pixel_weight=None,
                                                 min_pixels=1):
        num_classes = self.num_classes
        ignore_index = self.get_model().decode_head.ignore_index
        if gt_semantic_seg.dim() == 3:
            gt_semantic_seg = gt_semantic_seg.unsqueeze(1)
        if gt_semantic_seg.shape[2:] != feat_map.shape[2:]:
            gt_resized = F.interpolate(
                gt_semantic_seg.float(),
                size=feat_map.shape[2:],
                mode='nearest').long()
        else:
            gt_resized = gt_semantic_seg

        weight_flat = None
        if pixel_weight is not None:
            if pixel_weight.dim() == 3:
                pixel_weight = pixel_weight.unsqueeze(1)
            if pixel_weight.shape[2:] != feat_map.shape[2:]:
                pixel_weight = F.interpolate(
                    pixel_weight.float(),
                    size=feat_map.shape[2:],
                    mode='nearest')
            weight_flat = pixel_weight.squeeze(1).reshape(-1)

        gt_flat = gt_resized.squeeze(1).reshape(-1)
        feat_flat = feat_map.permute(0, 2, 3, 1).reshape(-1,
                                                         feat_map.shape[1])

        means = torch.zeros((num_classes, feat_map.shape[1]),
                            device=feat_map.device)
        present = torch.zeros((num_classes, ), device=feat_map.device,
                              dtype=torch.bool)
        counts = torch.zeros((num_classes, ), device=feat_map.device,
                             dtype=torch.long)
        for cls in range(num_classes):
            if cls == ignore_index:
                continue
            if self.dynamic_proto_ignore_background and cls == 0:
                continue
            mask = gt_flat == cls
            if weight_flat is not None:
                mask = mask & (weight_flat > 0)
            count = int(mask.sum().item())
            counts[cls] = count
            if count >= min_pixels:
                means[cls] = feat_flat[mask].mean(dim=0)
                present[cls] = True
        return means, present, counts

    def _maybe_init_source_prototypes(self, feat_map):
        if self.source_prototypes.numel() > 0 and \
                self.source_prototypes.shape[1] == feat_map.shape[1]:
            return
        feature_dim = feat_map.shape[1]
        device = feat_map.device
        self.source_prototypes = torch.zeros(
            (self.num_classes, feature_dim), device=device)
        self.source_proto_valid = torch.zeros(
            (self.num_classes, ), device=device, dtype=torch.bool)
        self.source_proto_seen = torch.zeros(
            (self.num_classes, ), device=device, dtype=torch.long)

    def _update_source_ema_prototypes(self, feat_map, gt_semantic_seg):
        if not self.enable_dynamic_proto or feat_map is None:
            return
        self._maybe_init_source_prototypes(feat_map)
        with torch.no_grad():
            means, present, _ = self._compute_class_feature_means_mask_counts(
                feat_map.detach(),
                gt_semantic_seg,
                min_pixels=self.dynamic_proto_min_pixels)
            present_classes = torch.nonzero(
                present, as_tuple=False).squeeze(1)
            momentum = float(self.dynamic_proto_momentum)
            for cls in present_classes.tolist():
                mean = F.normalize(means[cls], p=2, dim=0)
                if not self.source_proto_valid[cls]:
                    self.source_prototypes[cls] = mean
                    self.source_proto_valid[cls] = True
                else:
                    updated = momentum * self.source_prototypes[cls] + \
                        (1 - momentum) * mean
                    self.source_prototypes[cls] = F.normalize(
                        updated, p=2, dim=0)
                self.source_proto_seen[cls] += 1

    def _dynamic_proto_stats(self):
        if self.source_prototypes.numel() == 0:
            return {
                'valid_classes': 0.0,
                'mean_norm': 0.0,
                'min_pairwise_cos': 0.0,
                'mean_pairwise_cos': 0.0,
                'max_pairwise_cos': 0.0,
            }
        valid = self.source_proto_valid
        prototypes = self.source_prototypes[valid]
        if prototypes.numel() == 0:
            return {
                'valid_classes': 0.0,
                'mean_norm': 0.0,
                'min_pairwise_cos': 0.0,
                'mean_pairwise_cos': 0.0,
                'max_pairwise_cos': 0.0,
            }
        norms = torch.norm(prototypes, p=2, dim=1)
        stats = {
            'valid_classes': float(valid.sum().item()),
            'mean_norm': float(norms.mean().detach().item()),
            'min_pairwise_cos': 0.0,
            'mean_pairwise_cos': 0.0,
            'max_pairwise_cos': 0.0,
        }
        if prototypes.shape[0] > 1:
            prototypes = F.normalize(prototypes, p=2, dim=1)
            cos = torch.matmul(prototypes, prototypes.t())
            off_diag = cos[~torch.eye(
                cos.shape[0], dtype=torch.bool, device=cos.device)]
            stats.update({
                'min_pairwise_cos': float(off_diag.min().detach().item()),
                'mean_pairwise_cos': float(off_diag.mean().detach().item()),
                'max_pairwise_cos': float(off_diag.max().detach().item()),
            })
        return stats

    def _calc_dynamic_proto_loss(self,
                                 feat_map,
                                 semantic_seg,
                                 *,
                                 pixel_weight=None,
                                 use_target_temp=False):
        if feat_map is None or self.source_prototypes.numel() == 0:
            return None, {'loss_dynamic_proto': 0.0}
        means, present, _ = self._compute_class_feature_means_mask_counts(
            feat_map,
            semantic_seg,
            pixel_weight=pixel_weight,
            min_pixels=self.dynamic_proto_min_pixels)
        candidate_mask = self.source_proto_valid.clone()
        if self.dynamic_proto_ignore_background and candidate_mask.numel() > 0:
            candidate_mask[0] = False
        present_classes = torch.nonzero(
            present & candidate_mask, as_tuple=False).squeeze(1)
        candidate_classes = torch.nonzero(
            candidate_mask, as_tuple=False).squeeze(1)
        if present_classes.numel() == 0 or candidate_classes.numel() == 0:
            return None, {'loss_dynamic_proto': 0.0}

        prototypes = F.normalize(
            self.source_prototypes[candidate_classes].detach(), p=2, dim=1)
        means = F.normalize(means[present_classes], p=2, dim=1)
        target_positions = []
        for cls in present_classes.tolist():
            match = torch.nonzero(
                candidate_classes == cls, as_tuple=False).squeeze(1)
            if match.numel() > 0:
                target_positions.append(match[0])
        if len(target_positions) == 0:
            return None, {'loss_dynamic_proto': 0.0}
        targets = torch.stack(target_positions).to(feat_map.device)

        if use_target_temp:
            temperature = max(float(self.dynamic_proto_target_temperature),
                              1e-6)
            weight = self.dynamic_proto_target_lambda
        else:
            temperature = max(float(self.dynamic_proto_temperature), 1e-6)
            weight = self.dynamic_proto_source_lambda

        logits = torch.matmul(means, prototypes.t()) / temperature
        loss = weight * F.cross_entropy(logits, targets)
        loss, log_vars = self._parse_losses({'loss_dynamic_proto': loss})
        log_vars.pop('loss', None)
        return loss, log_vars

    def _calc_etf_contrastive_loss(self, feat_map, gt_semantic_seg, *,
                                   use_target_temp=False):
        if feat_map is None or self.etf_prototypes is None:
            return None, {'loss_etf_contrastive': 0.0}
        means, present = self._compute_class_feature_means_and_mask(
            feat_map, gt_semantic_seg)
        present_classes = torch.nonzero(present, as_tuple=False).squeeze(1)
        if present_classes.numel() == 0:
            return None, {'loss_etf_contrastive': 0.0}

        prototypes = F.normalize(self.etf_prototypes.detach(), p=2, dim=1)
        means = F.normalize(means[present_classes], p=2, dim=1)
        if use_target_temp:
            temperature = max(float(self.etf_contrastive_target_temperature),
                              1e-6)
            weight = self.etf_contrastive_target_lambda
        else:
            temperature = max(float(self.etf_contrastive_temperature), 1e-6)
            weight = self.etf_contrastive_lambda

        logits = torch.matmul(means, prototypes.t()) / temperature
        targets = present_classes.to(logits.device)
        loss = weight * F.cross_entropy(logits, targets)
        loss, log_vars = self._parse_losses(
            {'loss_etf_contrastive': loss})
        log_vars.pop('loss', None)
        return loss, log_vars

    def _maybe_init_etf_prototypes(self, feat_map, gt_semantic_seg):
        if self.etf_prototypes is not None:
            return
        if feat_map is None:
            mmcv.print_log(
                'ETF prototypes not initialized: missing pre-logit features.',
                'mmseg')
            return

        feature_dim = feat_map.shape[1]
        num_classes = self.num_classes
        prototypes = load_etf_class_prototypes(
            feature_dim,
            num_classes,
            device=feat_map.device,
            save_dir=self.etf_prototype_dir)
        if prototypes is None:
            prototypes = generate_etf_class_prototypes(
                feature_dim,
                num_classes,
                device=feat_map.device,
                save_dir=self.etf_prototype_dir)
            mmcv.print_log(
                f'Generated ETF prototypes: K={num_classes}, D={feature_dim} '
                f'at {self.etf_prototype_dir}.',
                'mmseg')
        else:
            mmcv.print_log(
                f'Loaded ETF prototypes: K={num_classes}, D={feature_dim} '
                f'from {self.etf_prototype_dir}.',
                'mmseg')

        self.etf_prototypes = prototypes
        self.source_class_means = self._compute_class_feature_means(
            feat_map.detach(), gt_semantic_seg)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        if self.source_only:
            return
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        if self.source_only:
            return
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        # If the mask is empty, the mean will be NaN. However, as there is
        # no connection in the compute graph to the network weights, the
        # network gradients are zero and no weight update will happen.
        # This can be verified with print_grad_magnitude.
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        # Features from multiple input scales (see HRDAEncoderDecoder)
        if isinstance(self.get_model(), HRDAEncoderDecoder) and \
                self.get_model().feature_scale in \
                self.get_model().feature_scale_all_strs:
            lay = -1
            feat = [f[lay] for f in feat]
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f[lay].detach() for f in feat_imnet]
            feat_dist = 0
            n_feat_nonzero = 0
            for s in range(len(feat_imnet)):
                if self.fdist_classes is not None:
                    fdclasses = torch.tensor(
                        self.fdist_classes, device=gt.device)
                    gt_rescaled = gt.clone()
                    if s in HRDAEncoderDecoder.last_train_crop_box:
                        gt_rescaled = crop(
                            gt_rescaled,
                            HRDAEncoderDecoder.last_train_crop_box[s])
                    scale_factor = gt_rescaled.shape[-1] // feat[s].shape[-1]
                    gt_rescaled = downscale_label_ratio(
                        gt_rescaled, scale_factor, self.fdist_scale_min_ratio,
                        self.num_classes, 255).long().detach()
                    fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses,
                                           -1)
                    fd_s = self.masked_feat_dist(feat[s], feat_imnet[s],
                                                 fdist_mask)
                    feat_dist += fd_s
                    if fd_s != 0:
                        n_feat_nonzero += 1
                    del fd_s
                    if s == 0:
                        self.debug_fdist_mask = fdist_mask
                        self.debug_gt_rescale = gt_rescaled
                else:
                    raise NotImplementedError
        else:
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f.detach() for f in feat_imnet]
            lay = -1
            if self.fdist_classes is not None:
                fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
                scale_factor = gt.shape[-1] // feat[lay].shape[-1]
                gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                    self.fdist_scale_min_ratio,
                                                    self.num_classes,
                                                    255).long().detach()
                fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                                  fdist_mask)
                self.debug_fdist_mask = fdist_mask
                self.debug_gt_rescale = gt_rescaled
            else:
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def update_debug_state(self):
        debug = self.local_iter % self.debug_img_interval == 0
        self.get_model().automatic_debug = False
        self.get_model().debug = debug
        if not self.source_only:
            self.get_ema_model().automatic_debug = False
            self.get_ema_model().debug = debug
        if self.mic is not None:
            self.mic.debug = debug

    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=logits.device)
        return pseudo_label, pseudo_weight

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      target_img,
                      target_img_metas,
                      rare_class=None,
                      valid_pseudo_mask=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training
        if self.mic is not None:
            self.mic.update_weights(self.get_model(), self.local_iter)

        self.update_debug_state()
        seg_debug = {}

        means, stds = get_mean_std(img_metas, dev)
        img = self._maybe_apply_fda(img, target_img, means, stds)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        src_prelogit = getattr(self.get_model().decode_head,
                               'latest_linear_pred_input', None)
        src_prelogit_detached = None
        if src_prelogit is not None:
            src_prelogit_detached = src_prelogit.detach()
        if self.enable_etf_contrastive or self.enable_etf_contrastive_target:
            self._maybe_init_etf_prototypes(src_prelogit_detached,
                                            gt_semantic_seg)
        if self.enable_dynamic_proto:
            self._update_source_ema_prototypes(src_prelogit_detached,
                                               gt_semantic_seg)
            log_vars.update(
                add_prefix(self._dynamic_proto_stats(), 'dynamic_proto'))
        seg_debug['Source'] = self.get_model().debug_output
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(
            retain_graph=self.enable_fdist or self.enable_etf_contrastive
            or self.enable_dynamic_proto_source)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # Source ETF contrastive loss
        if self.enable_etf_contrastive:
            etf_loss, etf_log = self._calc_etf_contrastive_loss(
                src_prelogit, gt_semantic_seg)
            log_vars.update(add_prefix(etf_log, 'src'))
            if etf_loss is not None:
                etf_loss.backward()
                del etf_loss

        # Source dynamic prototype loss. Source anchors are updated only from
        # source GT features, then detached as class anchors for this loss.
        if self.enable_dynamic_proto_source:
            proto_loss, proto_log = self._calc_dynamic_proto_loss(
                src_prelogit, gt_semantic_seg)
            log_vars.update(add_prefix(proto_log, 'src'))
            if proto_loss is not None:
                proto_loss.backward()
                del proto_loss

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            log_vars.update(add_prefix(feat_log, 'src'))
            feat_loss.backward()
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')
        del src_feat, clean_loss
        if self.enable_fdist:
            del feat_loss

        pseudo_label, pseudo_weight = None, None
        if not self.source_only:
            # Generate pseudo-label
            for m in self.get_ema_model().modules():
                if isinstance(m, _DropoutNd):
                    m.training = False
                if isinstance(m, DropPath):
                    m.training = False
            ema_logits = self.get_ema_model().generate_pseudo_label(
                target_img, target_img_metas)
            seg_debug['Target'] = self.get_ema_model().debug_output

            pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(
                ema_logits)
            del ema_logits

            pseudo_weight = self.filter_valid_pseudo_region(
                pseudo_weight, valid_pseudo_mask)
            gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

            do_mix = self.local_iter >= self.mix_start_iter

            do_dynamic_proto_target = self.local_iter >= \
                self.dynamic_proto_target_start_iter

            # Target prototype losses start after their warmup.
            if (self.enable_etf_contrastive_target and do_mix) or (
                    self.enable_dynamic_proto_target
                    and do_dynamic_proto_target):
                feats = self.get_model().extract_feat(target_img)
                _ = self.get_model().decode_head(feats)
                tgt_prelogit = getattr(self.get_model().decode_head,
                                       'latest_linear_pred_input', None)
            else:
                tgt_prelogit = None

            # Target ETF contrastive loss starts after the self-training warmup.
            if self.enable_etf_contrastive_target and do_mix:
                tgt_loss, tgt_log = self._calc_etf_contrastive_loss(
                    tgt_prelogit,
                    pseudo_label,
                    use_target_temp=True)
                log_vars.update(add_prefix(tgt_log, 'tgt'))
                if tgt_loss is not None:
                    tgt_loss.backward()
                    del tgt_loss

            # Target dynamic prototype loss explicitly aligns target class
            # means to the source EMA anchors. Target never updates anchors.
            if self.enable_dynamic_proto_target and do_dynamic_proto_target:
                tgt_proto_loss, tgt_proto_log = self._calc_dynamic_proto_loss(
                    tgt_prelogit,
                    pseudo_label,
                    pixel_weight=pseudo_weight,
                    use_target_temp=True)
                log_vars.update(add_prefix(tgt_proto_log, 'tgt'))
                if tgt_proto_loss is not None:
                    tgt_proto_loss.backward()
                    del tgt_proto_loss

            mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
            mixed_seg_weight = pseudo_weight.clone()
            if self.mix == 'class':
                # Apply class mixing between source labels and target pseudo-labels.
                mix_masks = get_class_masks(
                    gt_semantic_seg, class_mix_ignore=self.class_mix_ignore)
            else:
                mix_masks = [
                    torch.zeros_like(gt_semantic_seg[i])
                    for i in range(batch_size)
                ]

            for i in range(batch_size):
                strong_parameters['mix'] = mix_masks[i]
                mixed_img[i], mixed_lbl[i] = strong_transform(
                    strong_parameters,
                    data=torch.stack((img[i], target_img[i])),
                    target=torch.stack(
                        (gt_semantic_seg[i][0], pseudo_label[i])))
                _, mixed_seg_weight[i] = strong_transform(
                    strong_parameters,
                    target=torch.stack(
                        (gt_pixel_weight[i], pseudo_weight[i])))
            del gt_pixel_weight
            mixed_img = torch.cat(mixed_img)
            mixed_lbl = torch.cat(mixed_lbl)
            if mixed_lbl.dim() == 3:
                mixed_lbl = mixed_lbl.unsqueeze(1)

            if do_mix:
                # Train on mixed images, or target-only images when mixing is disabled.
                mix_losses = self.get_model().forward_train(
                    mixed_img,
                    img_metas,
                    mixed_lbl,
                    seg_weight=mixed_seg_weight,
                    return_feat=False,
                )
                seg_debug['Mix'] = self.get_model().debug_output
                mix_losses = add_prefix(mix_losses, 'mix')
                mix_loss, mix_log_vars = self._parse_losses(mix_losses)
                log_vars.update(mix_log_vars)
                mix_loss.backward()

        # Masked Training
        if self.enable_masking and self.mask_mode.startswith('separate'):
            masked_loss = self.mic(self.get_model(), img, img_metas,
                                   gt_semantic_seg, target_img,
                                   target_img_metas, valid_pseudo_mask,
                                   pseudo_label, pseudo_weight)
            seg_debug.update(self.mic.debug_output)
            masked_loss = add_prefix(masked_loss, 'masked')
            masked_loss, masked_log_vars = self._parse_losses(masked_loss)
            log_vars.update(masked_log_vars)
            masked_loss.backward()

        if self.local_iter % self.debug_img_interval == 0 and \
                not self.source_only:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                if mixed_lbl is not None:
                    subplotimg(
                        axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][3],
                    mixed_seg_weight[j],
                    'Pseudo W.',
                    vmin=0,
                    vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'debug')
            os.makedirs(out_dir, exist_ok=True)
            if seg_debug['Source'] is not None and seg_debug:
                if 'Target' in seg_debug:
                    seg_debug['Target']['Pseudo W.'] = mixed_seg_weight.cpu(
                    ).numpy()
                for j in range(batch_size):
                    cols = len(seg_debug)
                    rows = max(len(seg_debug[k]) for k in seg_debug.keys())
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(5 * cols, 5 * rows),
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0,
                            'top': 0.95,
                            'bottom': 0,
                            'right': 1,
                            'left': 0
                        },
                        squeeze=False,
                    )
                    for k1, (n1, outs) in enumerate(seg_debug.items()):
                        for k2, (n2, out) in enumerate(outs.items()):
                            subplotimg(
                                axs[k2][k1],
                                **prepare_debug_out(f'{n1} {n2}', out[j],
                                                    means, stds))
                    for ax in axs.flat:
                        ax.axis('off')
                    plt.savefig(
                        os.path.join(out_dir,
                                     f'{(self.local_iter + 1):06d}_{j}_s.png'))
                    plt.close()
                del seg_debug
        self.local_iter += 1

        return log_vars
