import torch

from ..bbox import PseudoSampler, assign_and_sample, build_assigner, build_bbox_coder
from ..utils import multi_apply
from mmdet.core.bbox.iou_calculators import build_iou_calculator

def anchor_hrsc_target(anchor_list,
                  valid_flag_list,
                  gt_bboxes_list,
                  img_metas,
                  target_means,
                  target_stds,
                  cfg,
                  output_bboxes,
                  cls_scores_list,
                  gt_bboxes_ignore_list=None,
                  gt_labels_list=None,
                  label_channels=1,
                  sampling=True,
                  reg_decoded_bbox=True,
                  use_vfl=False,
                  unmap_outputs=True):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs
    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]] #首先获取了每张图片中，每种尺度anchor的数目
    # concat all level anchors and flags to a single tensor 将每张图片中所有尺度的anchor放在一起 shape为[总共的anchor数,5]
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])
        output_bboxes[i] = torch.cat(output_bboxes[i])
        cls_scores_list[i] = torch.cat(cls_scores_list[i])
    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list, pos_total_num_list, num_total_neg_list, pos_anchor_num_list) = multi_apply(
        anchor_target_single,
        anchor_list,
        valid_flag_list,
        gt_bboxes_list,
        gt_bboxes_ignore_list,
        gt_labels_list,
        img_metas,
        output_bboxes,
        cls_scores_list,
        target_means=target_means,
        target_stds=target_stds,
        cfg=cfg,
        label_channels=label_channels,
        sampling=sampling,
        reg_decoded_bbox=reg_decoded_bbox,
        use_vfl=use_vfl,
        unmap_outputs=unmap_outputs)
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    pos_total_num = sum(pos_total_num_list)
    neg_total_num = sum(num_total_neg_list)
    pos_anchor_num = sum(pos_anchor_num_list)
    pos_ratio = pos_total_num / num_total_pos
    neg_ratio = neg_total_num / num_total_neg
    # with open('pos_iter.txt', 'a') as f:
    #     f.write(str(pos_total_num.item()) + '  ' + str(num_total_pos) + '  ' + str(pos_anchor_num.item()))
    #     f.write('\n')
    # with open('neg_iter.txt', 'a') as f:
    #     f.write(str(neg_total_num.item()) + '  ' + str(num_total_neg))
    #     f.write('\n')
    # split targets to a list w.r.t. multiple levels
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_total_pos, num_total_neg)


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def anchor_target_single(flat_anchors,
                         valid_flags,
                         gt_bboxes,
                         gt_bboxes_ignore,
                         gt_labels,
                         img_meta,
                         output_bboxes,
                         cls_scores,
                         target_means,
                         target_stds,
                         cfg,
                         label_channels=1,
                         sampling=True,
                         reg_decoded_bbox=False,
                         use_vfl=False,
                         unmap_outputs=True):
    bbox_coder_cfg = cfg.get('bbox_coder', '')
    if bbox_coder_cfg == '':
        bbox_coder_cfg = dict(type='DeltaXYWHBBoxCoder')
    bbox_coder = build_bbox_coder(bbox_coder_cfg)

    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)
    if not inside_flags.any():
        return (None,) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :]

    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)

    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    if use_vfl:
        labels = anchors.new_zeros((num_valid_anchors, label_channels),
                                           dtype=torch.float)
    else:
        labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        if not reg_decoded_bbox:
            pos_bbox_targets = bbox_coder.encode(sampling_result.pos_bboxes,
                                             sampling_result.pos_gt_bboxes)
        else:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
        bbox_targets[pos_inds, :] = pos_bbox_targets.to(bbox_targets)
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            if use_vfl:
                iou_calculator = build_iou_calculator(dict(type='BboxOverlaps2D_rotated'))
                iou = iou_calculator(output_bboxes, gt_bboxes)
                labels[pos_inds, gt_labels[sampling_result.pos_assigned_gt_inds]-1] = iou[pos_inds, sampling_result.pos_assigned_gt_inds]
                pos_iou, _ = iou[pos_inds, :].max(dim=1)
                all_zero = torch.zeros_like(pos_iou)
                all_one = torch.ones_like(pos_iou)
                output_iou = torch.where(pos_iou>0.5, all_one, all_zero)
                pos_total_num = output_iou.sum()
                neg_iou, _ = iou[neg_inds, :].max(dim=1)

                output_iou, _ = iou.max(dim=1)
                inds = output_iou > 0.05
                write_iou, write_score = output_iou[inds], cls_scores[inds].sigmoid()
                with open('vfl_output_iou_score.txt', 'a') as f:
                    for i in range(len(write_iou)):
                        f.write(str(write_iou[i].item()) + ' ' + str(write_score[i].item()))
                        f.write('\n')

                all_zero = torch.zeros_like(neg_iou)
                all_one = torch.ones_like(neg_iou)
                neg_iou = torch.where(neg_iou>0.5, all_one, all_zero)
                neg_total_num = neg_iou.sum()
                
                input_iou = iou_calculator(flat_anchors, gt_bboxes)
                pos_iou = input_iou[pos_inds, sampling_result.pos_assigned_gt_inds]
                all_zero = torch.zeros_like(pos_iou)
                all_one = torch.ones_like(pos_iou)
                output_iou = torch.where(pos_iou>0.5, all_one, all_zero)
                pos_anchor_num = output_iou.sum()

                input_iou, _ = input_iou.max(dim=1)
                inds = input_iou > 0.2
                write_iou, write_score = input_iou[inds], cls_scores[inds].sigmoid()
                with open('input_iou_score.txt', 'a') as f:
                    for i in range(len(write_iou)):
                        f.write(str(write_iou[i].item()) + ' ' + str(write_score[i].item()))
                        f.write('\n')
            else:
                iou_calculator = build_iou_calculator(dict(type='BboxOverlaps2D_rotated'))
                iou = iou_calculator(output_bboxes, gt_bboxes)
                pos_iou, _ = iou[pos_inds, :].max(dim=1)
                all_zero = torch.zeros_like(pos_iou)
                all_one = torch.ones_like(pos_iou)
                output_iou = torch.where(pos_iou>0.5, all_one, all_zero)
                pos_total_num = output_iou.sum()
                neg_iou, _ = iou[neg_inds, :].max(dim=1)

                output_iou, _ = iou.max(dim=1)
                inds = output_iou > 0.05
                write_iou, write_score = output_iou[inds], cls_scores[inds].sigmoid()
                with open('vfl_output_iou_score.txt', 'a') as f:
                    for i in range(len(write_iou)):
                        f.write(str(write_iou[i].item()) + ' ' + str(write_score[i].item()))
                        f.write('\n')

                all_zero = torch.zeros_like(neg_iou)
                all_one = torch.ones_like(neg_iou)
                neg_iou = torch.where(neg_iou>0.5, all_one, all_zero)
                neg_total_num = neg_iou.sum()
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]

                input_iou = iou_calculator(flat_anchors, gt_bboxes)
                pos_iou = input_iou[pos_inds, sampling_result.pos_assigned_gt_inds]
                all_zero = torch.zeros_like(pos_iou)
                all_one = torch.ones_like(pos_iou)
                output_iou = torch.where(pos_iou>0.5, all_one, all_zero)
                pos_anchor_num = output_iou.sum()

                input_iou, _ = input_iou.max(dim=1)
                inds = input_iou > 0.2
                write_iou, write_score = input_iou[inds], cls_scores[inds].sigmoid()
                with open('input_iou_score.txt', 'a') as f:
                    for i in range(len(write_iou)):
                        f.write(str(write_iou[i].item()) + ' ' + str(write_score[i].item()))
                        f.write('\n')
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds, pos_total_num, neg_total_num, pos_anchor_num)


# TODO for rotated box
def anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                        allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
                       (flat_anchors[:, 0] >= -allowed_border).type(torch.uint8) & \
                       (flat_anchors[:, 1] >= -allowed_border).type(torch.uint8) & \
                       (flat_anchors[:, 2] < img_w + allowed_border).type(torch.uint8) & \
                       (flat_anchors[:, 3] < img_h + allowed_border).type(torch.uint8)
    else:
        inside_flags = valid_flags
    return inside_flags


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret