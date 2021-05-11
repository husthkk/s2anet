from __future__ import division
import numpy as np
import torch
import torch.nn as nn
from mmdet.core import (AnchorGeneratorRotated, anchor_target,
                        delta2bbox_rotated, force_fp32, multi_apply,
                        multiclass_nms_rotated, images_to_levels, build_bbox_coder)

from mmdet.core import AnchorGeneratorRotated, delta2bbox_rotated, multiclass_nms_rotated, build_bbox_coder,force_fp32,multi_apply,anchor_target,images_to_levels, anchor_target_atss, anchor_target_hrsc_multianchors, anchor_hrsc_target
from ..anchor_heads import AnchorHead
from ..registry import HEADS


@HEADS.register_module
class AnchorHeadRotated(AnchorHead):

    def __init__(self, *args, 
                     anchor_angles=[0., ],
                     bbox_coder=dict(
                        type='DeltaXYWHABBoxCoder',
                        target_means=(.0, .0, .0, .0, .0),
                        target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)), 
                     **kargs):
        super(AnchorHeadRotated, self).__init__(*args, **kargs)

        self.anchor_angles = anchor_angles
        self.reg_decoded_bbox = False
        self.use_vfl = True
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGeneratorRotated(
                    anchor_base, self.anchor_scales, self.anchor_ratios, angles=anchor_angles))

        self.num_anchors = len(self.anchor_ratios) * \
            len(self.anchor_scales) * len(self.anchor_angles)

        self._init_layers()

    def _init_layers(self):
        self.conv_cls = nn.Conv2d(self.in_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.in_channels, self.num_anchors * 5, 1)

    def get_refine_anchors(self,
                           featmap_sizes,
                           refine_anchors,
                           img_metas,
                           is_train=True,
                           device='cuda'):
        num_levels = len(featmap_sizes)

        refine_anchors_list = []   #list[list] 外面这层list是bs大小，里面这层list是feature level数 refine_anchors是一个list，len为feature level数
        for img_id, img_meta in enumerate(img_metas):
            mlvl_refine_anchors = []
            for i in range(num_levels):
                refine_anchor = refine_anchors[i][img_id].reshape(-1, 5)
                mlvl_refine_anchors.append(refine_anchor)
            refine_anchors_list.append(mlvl_refine_anchors)

        valid_flag_list = []
        if is_train:
            for img_id, img_meta in enumerate(img_metas):
                multi_level_flags = []
                for i in range(num_levels):
                    anchor_stride = self.anchor_strides[i]
                    feat_h, feat_w = featmap_sizes[i]
                    h, w, _ = img_meta['pad_shape']
                    valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                    valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                    flags = self.anchor_generators[i].valid_flags(
                        (feat_h, feat_w), (valid_feat_h, valid_feat_w),
                        device=device)
                    multi_level_flags.append(flags)
                valid_flag_list.append(multi_level_flags)
        return refine_anchors_list, valid_flag_list


    def forward_single(self, x, stride):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)

        num_level = self.anchor_strides.index(stride)
        featmap_size = bbox_pred.shape[-2:]
        device = bbox_pred.device
        init_anchors = self.anchor_generators[num_level].grid_anchors(
                featmap_size, self.anchor_strides[num_level], device=device)

        bs = bbox_pred.shape[0]
        init_anchors = init_anchors.repeat(bs,1,1)
        #refine_anchor shape: bs x h x w x 5
        refine_anchor = bbox_decode(
            bbox_pred.detach(),
            init_anchors,
            self.target_means,
            self.target_stds)

        return cls_score, bbox_pred, refine_anchor

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.anchor_strides)

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        # classification loss
        # labels = labels.reshape(-1)
        # import pdb;pdb.set_trace()
        if self.use_vfl:
            labels = labels.reshape(-1, 1)
        else:
            labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(
            0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 5) #erenzhou 4 to 5
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             bboxes,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        output_bboxes, _ = self.get_refine_anchors(
            featmap_sizes, bboxes, img_metas, device=device)
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_scores_list = []   #list[list] 外面这层list是bs大小，里面这层list是feature level数 refine_anchors是一个list，len为feature level数
        for img_id, img_meta in enumerate(img_metas):
            mlvl_cls_score = []
            for i in range(5):
                cls_score = cls_scores[i][img_id].permute(1,2,0).reshape(-1, 1)
                mlvl_cls_score.append(cls_score)
            cls_scores_list.append(mlvl_cls_score)
        cls_reg_targets = anchor_hrsc_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            output_bboxes,
            cls_scores_list,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling,
            reg_decoded_bbox=self.reg_decoded_bbox,
            use_vfl=self.use_vfl)
        # cls_reg_targets = anchor_target(
        #     anchor_list,
        #     valid_flag_list,
        #     gt_bboxes,
        #     img_metas,
        #     self.target_means,
        #     self.target_stds,
        #     cfg,
        #     output_bboxes,
        #     gt_bboxes_ignore_list=gt_bboxes_ignore,
        #     gt_labels_list=gt_labels,
        #     label_channels=label_channels,
        #     sampling=self.sampling,
        #     reg_decoded_bbox=self.reg_decoded_bbox,
        #     use_vfl=self.use_vfl)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        all_anchor_list = images_to_levels(anchor_list,
                                           num_level_anchors)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def get_bboxes_single(self,
                          cls_score_list,
                          bbox_pred_list,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        """
        Transform outputs for a single batch item into labeled boxes.
        """
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = delta2bbox_rotated(anchors, bbox_pred, self.target_means,
                                        self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes[..., :4] /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms_rotated(mlvl_bboxes, mlvl_scores,
                                                        cfg.score_thr, cfg.nms,
                                                        cfg.max_per_img)
        return det_bboxes, det_labels


def bbox_decode(bbox_preds, anchors, means=[0, 0, 0, 0, 0],stds=[1, 1, 1, 1, 1]):
    num_imgs, _, H, W = bbox_preds.shape
    bboxes_list = []
    for img_id in range(num_imgs):
        bbox_pred = bbox_preds[img_id]
        anchor = anchors[img_id]
        # bbox_pred.shape=[5,H,W]
        bbox_delta = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
        bboxes = delta2bbox_rotated(
            anchor, bbox_delta, means, stds, wh_ratio_clip=1e-6)
        bboxes = bboxes.reshape(H, W, 5)
        bboxes_list.append(bboxes)
    return torch.stack(bboxes_list, dim=0)