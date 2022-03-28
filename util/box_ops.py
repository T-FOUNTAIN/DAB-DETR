# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
import torchsnooper
import torchvision
from torchvision.ops.boxes import box_area
import numpy as np
from torchvision.ops.boxes import box_area
import BboxToolkit as bt
import time
import matplotlib.pyplot as plt


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

def obox2polys(oboxes):
    cx, cy, w, h, theta = oboxes.unbind(-1)
    center = torch.cat([cx.unsqueeze(-1), cy.unsqueeze(-1)], dim=-1)
    w = w.unsqueeze(-1)
    h = h.unsqueeze(-1)
    theta = theta.unsqueeze(-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    vector1 = torch.cat([w / 2 * Cos, -w / 2 * Sin], dim=-1)
    vector2 = torch.cat([-h / 2 * Sin, -h / 2 * Cos], dim=-1)
    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    return torch.cat([point1, point2, point3, point4], dim=-1)

def box_vis(src, pred_polys, pred_labels, pred_scores, tgt_polys, tgt_labels, output_dir):
    """
    :param src:  Tensor, normalizes img, [3, width, height]
    :param pred_polys: Tensor, [num_queries, 8]
    :param pred_labels: Tensor, [num_queries, ]
    :return: tgt_polys [n, 8]
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=src.device).view(3, 1, 1).repeat(1, src.shape[1], src.shape[2])
    std = torch.tensor([0.229, 0.224, 0.225], device=src.device).view(3, 1, 1).repeat(1, src.shape[1], src.shape[2])
    src = (torch.mul(src, std) + mean)

    keep = torch.where(pred_scores>=0.2)[0]

    img = src.permute((1, 2, 0)).detach().to('cpu').numpy()
    pred_polys_np = pred_polys[keep, :].detach().to('cpu').numpy()
    pred_labels_np = pred_labels[keep].detach().to('cpu').numpy()
    pred_scores_np =  pred_scores[keep].detach().to('cpu').numpy()
    tgt_polys_np = tgt_polys.detach().to('cpu').numpy()
    tgt_labels_np =  tgt_labels.detach().to('cpu').numpy()

    height, width = img.shape[:2]
    ax, fig = bt.plt_init('', width, height)
    ax.imshow(img)
    text_vis_tgt = [tgt_labels_np[ind] for ind in range(tgt_labels_np.shape[0])]
    text_vis_pred_label = [pred_labels_np[ind] for ind in range(pred_labels_np.shape[0])]
    text_vis_pred_score = [pred_scores_np[ind] for ind in range(pred_scores_np.shape[0])]
    for ind, box in enumerate(pred_polys_np):
        bt.draw_poly(ax, box, texts=None, color='red')
        ax.text(box[0], box[1], text_vis_pred_label[ind], bbox=dict(facecolor='red', alpha=0.5))
        ax.text(box[2], box[3], text_vis_pred_score[ind], bbox=dict(facecolor='red', alpha=0.5))
    for ind, box in enumerate(tgt_polys_np):
        bt.draw_poly(ax, box, texts=None, color='green')
        ax.text(box[0], box[1], text_vis_tgt[ind], bbox=dict(facecolor='green', alpha=0.5))
    #plt.show()
    if tgt_polys_np.shape[0] > 0:
        img = bt.get_img_from_fig(fig, width, height)
        img = torchvision.transforms.ToPILImage()(img)
        img.save(output_dir+'/vis/' + str(time.time()) + '_with_bbox.png')
    plt.close()