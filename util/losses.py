import torch
from torch import nn
import torch.nn.functional as F
import shapely
from shapely.geometry import Polygon,MultiPoint
from mmdet.models.losses.utils import weighted_loss
from mmdet.models.builder import MODELS
import BboxToolkit as bt
ROTATED_LOSSES = MODELS
eps = 1e-8

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

def angle_cls_period_focal_loss(inputs_prob, targets, input_oboxes, tgt_oboxes, alpha=None, gamma=2.0, aspect_ratio_threshold=1.5):
    # compute the focal loss
    # inputs_probs has been after sigmoid()
    per_entry_cross_ent =  F.binary_cross_entropy_with_logits(inputs_prob, targets, reduction="none")
    p_t = inputs_prob * targets + (1 - inputs_prob) * (1 - targets)
    modulating_factor = 1.0
    if gamma:
        modulating_factor = torch.pow(1.0 - p_t, gamma)
    alpha_weight_factor = 1.0
    if alpha is not None:
        alpha_weight_factor = targets * alpha + (1 - targets) * (1 - alpha)

    _, _, w, h, theta = torch.unbind(tgt_oboxes, -1)
    tgt_theta = theta
    inputs_theta = input_oboxes[:, -1]
    aspect_ratio = w/h
    period_weight_90 = torch.le(aspect_ratio, aspect_ratio_threshold).int() * 2
    period_weight_180 = torch.ge(aspect_ratio, aspect_ratio_threshold).int() * 1
    period_weight = (period_weight_90 + period_weight_180).float()
    diff_weight = (torch.abs(torch.sin(period_weight * (tgt_theta - inputs_theta)))).view([-1, 1])
    focal_cross_entropy_loss = (diff_weight * modulating_factor * alpha_weight_factor * per_entry_cross_ent)

    return focal_cross_entropy_loss

def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma

def cal_kfiou(inputs, targets, fun='exp', beta=1.0 / 9.0, eps=1e-6):

    """Kalman filter IoU loss.

    Args:
        inputs (torch.Tensor):Predicted xywhr bboxes with shape[M, N, 5] or [N, 5]
        target (torch.Tensor): target xywhr bboxes with shape[M, N, 5] or [N, 5]
        fun(str): caculate kfiou loss mode
    Returns:
        loss (torch.Tensor): with shape [M, N]
    """
    size = inputs.shape
    if inputs.dim() == 3:
        if size[1] == 0: # no objects
            return torch.randn((size[0], size[1]), device=inputs.device)
        inputs = inputs.flatten(0, 1)
        targets = targets.flatten(0, 1)
    else:
        if size[0] == 0:
            return torch.randn((0), device=inputs.device)
    _, Sigma_p = xy_wh_r_2_xy_sigma(inputs)
    _, Sigma_t = xy_wh_r_2_xy_sigma(targets)

    Vb_p = 4 * Sigma_p.det().sqrt()
    Vb_t = 4 * Sigma_t.det().sqrt()
    K = Sigma_p.bmm((Sigma_p + Sigma_t).inverse())
    Sigma = Sigma_p - K.bmm(Sigma_p)
    Vb = 4 * Sigma.det().sqrt()
    Vb = torch.where(torch.isnan(Vb), torch.full_like(Vb, 0), Vb)
    KFIoU = Vb / (Vb_p + Vb_t - Vb + eps)

    if fun == 'ln':
        kf_loss = -torch.log(KFIoU + eps)
    elif fun == 'exp':
        kf_loss = torch.exp(1 - KFIoU) - 1
    else:
        kf_loss = 1 - KFIoU
    return kf_loss.view(size[0], size[1]) if len(size)==3 else kf_loss

def matched_l1_loss(pred, gt):
    '''
    Args:
        pred: Tensor[M, N, 8]
        gt: Tensor[M, N, 8]
    Return:
        matched_l1_loss: Tensor[M, N]
    '''
    size = pred.shape
    if pred.dim() == 3:
        if size[1] == 0: # no objects
            return (torch.randn((size[0], size[1]), device=pred.device), torch.randn((size[0], size[1]), device=pred.device))
    else:
        if size[0] == 0:
            return (torch.randn((0), device=pred.device), torch.randn((0), device=pred.device))
    gt = gt.unsqueeze(2).repeat(1, 1, 4, 1)
    pred1 = torch.cat([pred[:, :, 2:], pred[:, :, :2]], dim=-1) # (x2,y2,x3,y3,x4,y4,x1,y1)
    pred2 = torch.cat([pred[:, :, 4:], pred[:, :, :4]], dim=-1) # (x3,y3,x4,y4,x1,y1,x2,y2)
    pred3 = torch.cat([pred[:, :, 6:], pred[:, :, :6]], dim=-1) # (x4,y4,x1,y1,x2,y2,x3,y3)
    pred = torch.cat([pred.unsqueeze(2), pred1.unsqueeze(2), pred2.unsqueeze(2), pred3.unsqueeze(2)], dim = 2)
    return torch.min(torch.sum(torch.abs(pred - gt), dim=-1), dim=-1)

