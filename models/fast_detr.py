# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from util.oriented_iou_loss import cal_giou
from .backbone import build_backbone
from .matcher import build_matcher
from util.losses import sigmoid_focal_loss
from .transformer import build_transformer
from .misc import _get_clones, MLP


class FastDETR(nn.Module):
    """ This is the SAM-DETR module that performs object detection """
    def __init__(self, args, backbone, transformer, num_classes):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         that our model can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.args = args
        self.multiscale = args.multiscale
        self.num_feature_levels = 3 if self.multiscale else 1          # Hard-coded multiscale parameters
        self.num_queries = args.num_queries
        self.aux_loss = args.aux_loss
        self.hidden_dim = args.hidden_dim
        assert self.hidden_dim == transformer.d_model

        self.backbone = backbone
        self.transformer = transformer

        # Instead of modeling query_embed as learnable parameters in the shape of (num_queries, d_model),
        # we directly model reference boxes in the shape of (num_queries, 5), in the format of (xc yc w h theta).
        self.query_embed = nn.Embedding(self.num_queries, 5)           # Reference rotated boxes

        # ====================================================================================
        #                                   * Clarification *
        #  -----------------------------------------------------------------------------------
        #  Whether self.input_proj contains nn.GroupNorm should not affect performance much.
        #  nn.GroupNorm() is introduced in some of our experiments by accident.
        #  Our experience shows that it even slightly degrades the performance.
        #  We recommend you to simply delete nn.GroupNorm() in self.input_proj for your own
        #  experiments, but if you wish to reproduce our results, please follow our setups below.
        # ====================================================================================
        if self.multiscale:
            input_proj_list = []
            for _ in range(self.num_feature_levels):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    # nn.GroupNorm(32, self.hidden_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            if self.args.epochs >= 25:
                self.input_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], self.hidden_dim, kernel_size=1),
                    )])
            else:
                self.input_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], self.hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, self.hidden_dim),
                    )])

        self.class_embed = nn.Linear(self.hidden_dim, num_classes)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 5, 3) # (cx cy w h theta) offset

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init bbox_embed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        self.class_embed = _get_clones(self.class_embed, args.dec_layers)
        self.bbox_embed = _get_clones(self.bbox_embed, args.dec_layers)

        self.transformer.decoder.bbox_embed = self.bbox_embed

    def forward(self, samples: NestedTensor):
        """The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos_embeds = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        hs, reference = self.transformer(srcs, masks, self.query_embed.weight, pos_embeds)

        outputs_coords = []
        outputs_class = []
        for lvl in range(hs.shape[0]):
            reference_before_sigmoid = reference[lvl]
            bbox_offset = self.bbox_embed[lvl](hs[lvl])
            anchor_box = (reference_before_sigmoid + bbox_offset)
            outputs_coord= torch.cat([anchor_box[..., :4].sigmoid(), anchor_box[...,4:5]], dim=-1)
            outputs_coords.append(outputs_coord)
            outputs_class.append(self.class_embed[lvl](hs[lvl]))
        outputs_coords = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_class)
        outputs_polys = box_ops.obox2polys(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coords[-1], 'pred_polys': outputs_polys[-1]}
        if self.aux_loss:
           out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coords, outputs_polys)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coords, output_polys):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_polys': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coords[:-1], output_polys[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for SAM-DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) \
                  * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        assert 'pred_polys' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_oboxes = outputs['pred_boxes'][idx]
        src_polys = outputs['pred_polys'][idx]

        target_polys = torch.cat([t['polys'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_theta = torch.cat([t['theta'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_oboxes = torch.cat([target_boxes, target_theta], dim=-1)

        target_polys_matched = target_polys.unsqueeze(1).repeat(1, 4, 1) # [n, 4, 8]
        pred1 = torch.cat([src_polys[:, 2:], src_polys[:, :2]], dim=-1)  # (x2,y2,x3,y3,x4,y4,x1,y1)
        pred2 = torch.cat([src_polys[:, 4:], src_polys[:, :4]], dim=-1)  # (x3,y3,x4,y4,x1,y1,x2,y2)
        pred3 = torch.cat([src_polys[:, 6:], src_polys[:, :6]], dim=-1)  # (x4,y4,x1,y1,x2,y2,x3,y3)
        src_polys_matched = torch.cat([src_polys.unsqueeze(1), pred1.unsqueeze(1), pred2.unsqueeze(1), pred3.unsqueeze(1)], dim=1)# [n, 4, 8]
        loss_bbox = F.smooth_l1_loss(src_polys_matched.to(dtype=torch.float64), target_polys_matched.to(dtype=torch.float64), reduction='none').sum(-1) #[n, 4]
        if target_polys.shape[0] > 0: # no object
            loss_bbox = torch.min(loss_bbox, dim=-1)[0]

        loss_giou = cal_giou(src_oboxes, target_oboxes)

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_boxes = outputs['pred_logits'], outputs['pred_polys']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 500, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        polys = torch.gather(out_boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,8)).view(-1, 8)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale = torch.stack([img_w, img_h, img_w, img_h, img_w, img_h, img_w, img_h], dim=1).unsqueeze(1).repeat(1, out_logits.shape[1],1)
        polys = polys.view(out_logits.shape[0], -1, 8) * scale

        results = [{'scores': s, 'labels': l, 'polys': b} for s, l, b in zip(scores, labels, polys)]

        return results


def build(args):
    device = torch.device(args.device)
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    if args.dataset_file == "dota1.0":
        num_classes = 15+1
    elif args.dataset_file == "dota1.5":
        num_classes = 16+1

    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = FastDETR(args, backbone, transformer, num_classes=num_classes)

    matcher = build_matcher(args)
    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
    }

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    #losses = ['labels', 'boxes', 'cardinality']
    losses = ['labels', 'boxes']

    criterion = SetCriterion(num_classes,
                             matcher=matcher,
                             weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha,
                             losses=losses)
    criterion.to(device)
    post_processors = {'bbox': PostProcess()}

    return model, criterion, post_processors
