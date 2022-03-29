# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from util.box_ops import box_vis
from datasets.dota_eval import DotaEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1000

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # boxes_batch = torch.cat([t['boxes'] for t in targets])
        # if boxes_batch.shape[0]==0:
        #     continue
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name, param.is_leaf)
        for name, parms in model.named_parameters():
            grad = parms.grad
            if grad==None:
                continue
            else:
                if torch.all(grad==0):
                    print('-->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value_shape:',parms.grad.shape)

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        del samples
        del targets
        del outputs
        del loss_dict
        del loss_dict_reduced
        del loss_dict_reduced_unscaled
        del weight_dict
        del losses
        del losses_reduced_scaled

    torch.cuda.empty_cache()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir, gt_cls, vis = False):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    dota_evaluator =  DotaEvaluator(gt_cls = gt_cls)
    for samples, targets in metric_logger.log_every(data_loader, 1000, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["img_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        srcs, _ = samples.decompose()
        for target, output, src in zip(targets, results, srcs):
            img_h, img_w = target['orig_size']
            tgt_polys = target['polys']
            scale = torch.tensor([img_w, img_h, img_w, img_h, img_w, img_h, img_w, img_h], device=device)[None, ...].\
                repeat(tgt_polys.shape[0], 1)
            tgt_polys = tgt_polys * scale
            tgt_labels = target['labels']


            output_polys = output['polys']
            output_labels = output['labels']
            output_scores = output['scores']

            dota_evaluator.update(tgt_polys, tgt_labels, output_polys, output_labels, output_scores)

            if vis:
                box_vis(src, output_polys, output_labels, output_scores, tgt_polys, tgt_labels, output_dir)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    dota_evaluator.eval_map()
    
    del samples
    del targets
    del outputs
    del loss_dict
    del loss_dict_reduced
    del loss_dict_reduced_unscaled
    del weight_dict

    torch.cuda.empty_cache()

    return stats
