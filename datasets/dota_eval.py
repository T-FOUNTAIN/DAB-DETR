import torch
import BboxToolkit as bt
from util.misc import all_gather
class DotaEvaluator(object):
    def __init__(self, gt_cls, iou_thr=0.5, voc_metric = True, nproc = 10):
        self.res = []
        self.gts = []
        self.gt_cls = gt_cls
        self.iou_thr = iou_thr
        self.voc_metric = voc_metric
        self.nproc = nproc

    def update(self, tgt_polys, tgt_labels, pred_polys, pred_labels, pred_scores):
        res_dets = torch.cat([pred_polys, pred_scores[..., None]], dim=1)
        res_dets = [res_dets[pred_labels == i].detach().to('cpu').numpy() for i in range(len(self.gt_cls))]
        self.res.append(res_dets)

        gt_ann = {'bboxes': tgt_polys.detach().to('cpu').numpy(),
                  'labels': tgt_labels.detach().to('cpu').numpy()}
        self.gts.append(gt_ann)
    def eval_map(self):
        print('Starting calculating mAP')
        res_all_gather = all_gather(self.res)
        gts_all_gather = all_gather(self.gts)
        merged_res = []
        for r in res_all_gather:
            merged_res += r
        merged_gts = []
        for g in gts_all_gather:
            merged_gts += g
        res_all_gather = []
        gts_all_gather = []
        bt.eval_map(merged_res, merged_gts, iou_thr=self.iou_thr,
                    nproc=self.nproc, dataset=self.gt_cls, use_07_metric=self.voc_metric)
