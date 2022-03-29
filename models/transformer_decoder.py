# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision

from models.misc import _get_clones, _get_activation_fn, MLP
from models.position_encoding import gen_sineembed_for_position, gen_sineembed_for_single_position
from models.attention import MultiheadAttention
from util.box_ops import box_cxcywh_to_xyxy


class TransformerDecoder(nn.Module):
    def __init__(self, args, decoder_layer, num_layers):
        super().__init__()
        self.args = args
        self.multiscale = args.multiscale
        self.num_layers = num_layers
        self.layers = _get_clones(decoder_layer, num_layers)
        self.offset = None
        assert num_layers == self.args.dec_layers

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                memory_h=None,
                memory_w=None,
                grid=None):

        output = tgt

        intermediate = []
        intermediate_reference_boxes = []
        reference_boxes = None

        for layer_id, layer in enumerate(self.layers):
            if layer_id == 0 or layer_id == 1:
                scale_level = 2
            elif layer_id == 2 or layer_id == 3:
                scale_level = 1
            elif layer_id == 4 or layer_id == 5:
                scale_level = 0
            else:
                assert False

            if layer_id == 0:
                reference_boxes = query_pos  # [num_queries, batch_size, 5]
            else:
                reference_boxes = reference_boxes

            reference_boxes = reference_boxes.sigmoid()
            # get sine embedding for the query vector
            query_ref_boxes_sine_embed_cx = gen_sineembed_for_single_position(reference_boxes[..., 0])
            query_ref_boxes_sine_embed_cy = gen_sineembed_for_single_position(reference_boxes[..., 1])
            query_ref_boxes_sine_embed_w =  gen_sineembed_for_single_position(reference_boxes[..., 2])
            query_ref_boxes_sine_embed_h = gen_sineembed_for_single_position(reference_boxes[..., 3])
            query_ref_boxes_sine_embed_theta = gen_sineembed_for_single_position(reference_boxes[..., 4])
            # [num_queries, batch_size, 128 * 5]
            query_ref_boxes_sine_embed = torch.cat([query_ref_boxes_sine_embed_cx, query_ref_boxes_sine_embed_cy,
                                query_ref_boxes_sine_embed_w, query_ref_boxes_sine_embed_h, query_ref_boxes_sine_embed_theta], dim = 2)

            if self.multiscale:
                memory_ = memory[scale_level]
                memory_h_ = memory_h[scale_level]
                memory_w_ = memory_w[scale_level]
                memory_key_padding_mask_ = memory_key_padding_mask[scale_level]
                pos_ = pos[scale_level]
                grid_ = grid[scale_level]
            else:
                memory_ = memory
                memory_h_ = memory_h
                memory_w_ = memory_w
                memory_key_padding_mask_ = memory_key_padding_mask
                pos_ = pos
                grid_ = grid

            output, self.offset = layer(output,
                           memory_,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask_,
                           pos=pos_,
                           query_ref_boxes_sine_embed=query_ref_boxes_sine_embed,
                           reference_boxes=reference_boxes,
                           memory_h=memory_h_,
                           memory_w=memory_w_,
                           grid=grid_, )

            intermediate_reference_boxes.append((reference_boxes + self.offset).transpose(0, 1))
            if layer_id > 1:
                reference_boxes = reference_boxes.detach() + self.offset # updated_anchor_box ,without sigmoid()
            else:
                reference_boxes = reference_boxes + self.offset
            # reference_boxes = reference_boxes.detach() + self.offset
            # reference_boxes = reference_boxes + self.offset
            intermediate.append(output)

        return torch.stack(intermediate).transpose(1, 2), \
               torch.stack(intermediate_reference_boxes)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, args, activation="relu"):
        super().__init__()
        self.args = args
        self.d_model = args.hidden_dim
        self.nheads = args.nheads
        self.num_queries = args.num_queries
        self.dim_feedforward = args.dim_feedforward
        self.dropout = args.dropout
        self.activation = _get_activation_fn(activation)

        # Decoder Self-Attention
        self.sa_qpos_proj = MLP(self.d_model//2 * 5, self.d_model, self.d_model, 2)
        self.self_attn = MultiheadAttention(self.d_model, self.nheads, dropout=self.dropout, vdim=self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)
        self.norm1 = nn.LayerNorm(self.d_model)

        # Decoder Cross-Attention
        self.ca_qcontent_modulated_proj = MLP(self.d_model, self.d_model, self.d_model, 2)
        self.cross_attn = MultiheadAttention(self.d_model*2, self.nheads, dropout=self.dropout, vdim=self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)
        self.norm2 = nn.LayerNorm(self.d_model)

        # FFN
        self.linear1 = nn.Linear(self.d_model, self.dim_feedforward)
        self.dropout88 = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.dim_feedforward, self.d_model)
        self.dropout3 = nn.Dropout(self.dropout)
        self.norm3 = nn.LayerNorm(self.d_model)

        # Update anchor box
        self.tgt_offset_proj = MLP(self.d_model, self.d_model, 5, 2)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_ref_boxes_sine_embed = None,
                reference_boxes: Optional[Tensor] = None,
                memory_h=None,
                memory_w=None,
                grid=None):

        num_queries, bs, c = tgt.shape
        anchor_cx = reference_boxes[..., 0:1].flatten(0, 1) * memory_w
        anchor_cy = reference_boxes[..., 1:2].flatten(0, 1) * memory_h
        anchor_w = reference_boxes[..., 2:3].flatten(0, 1) * memory_w
        anchor_h = reference_boxes[..., 3:4].flatten(0, 1) * memory_h
        anchor_theta = (reference_boxes[..., 4:5].flatten(0, 1) -0.5) * 3.1415926 #(num_queries*bs, 1)

        cos_r = torch.cos(anchor_theta)
        sin_r = torch.sin(anchor_theta)
        R = torch.cat([cos_r, sin_r, -sin_r, cos_r], dim=-1).view(-1, 2, 2) #(num_queries*bs, 2, 2)
        gamma = torch.cat([anchor_w.mul(anchor_w)/4., torch.zeros_like(anchor_w),
                           torch.zeros_like(anchor_h), anchor_h.mul(anchor_h)/4.], dim=-1).view(-1, 2, 2) #(num_queries*bs, 2, 2)
        Sigma_inv = torch.inverse(R.bmm(gamma).bmm(R.transpose(1, 2))) #(num_queries*bs, 2, 2)
        Sigma_inv_ext = Sigma_inv.unsqueeze(1).repeat(1, grid.shape[1], 1, 1).flatten(0, 1) #(num_q * bs, hw, 2, 2)->(num_q*bs*hw, 2, 2)

        grid_cord = grid.unsqueeze(0).repeat(num_queries, 1, 1, 1).flatten(0, 2).unsqueeze(1) #(num_q, bs, hw, 2)->(num_q*bs*hw, 2)->(num_q*bs*hw, 1, 2)
        miu = torch.cat([anchor_cx, anchor_cy], dim=-1).unsqueeze(1).repeat(1, grid.shape[1], 1).flatten(0, 1).unsqueeze(1) #(num_q*bs, hw, 2)->(num_q*bs*hw, 2)->(num_q*bs*hw, 1, 2)
        gaussian = torch.exp(-0.5 * (grid_cord - miu).bmm(Sigma_inv_ext).bmm((grid_cord - miu).transpose(1, 2))).sigmoid()\
            .view(num_queries, bs, -1).permute(1, 0, 2).repeat(8, 1, 1) #(num_q*bs*hw, 1, 1)->(num_q, bs, hw)->(bs, num_q, hw)->(bs*head, num_q, hw)
        # gaussian_vis= gaussian.clone().detach().to('cpu').view(gaussian.shape[0], gaussian.shape[1], 50, 50)[0]
        # gaussians_vis = gaussian_vis[:20]
        # for i in range(20):
        #     g = gaussians_vis[i]
        #     img_g = torchvision.transforms.ToPILImage()(g.unsqueeze(0).repeat(3,1,1))
        #     img_g.save('/home/ttfang/code/DAB-DETR/output/gaussian_vis/'+str(i)+'.png')


        # ========== Begin of Self-Attention =============
        query_pos = self.sa_qpos_proj(query_ref_boxes_sine_embed)
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        # ========== End of Self-Attention =============
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt) # [num_queries, batch_size, 256]

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content_modulate = self.ca_qcontent_modulated_proj(tgt)

        q_pos_x = query_ref_boxes_sine_embed[..., :128]\
            .mul(q_content_modulate[..., :128]).view(num_queries, bs, self.nheads, c//(2*self.nheads))
        q_pos_y = query_ref_boxes_sine_embed[..., 128:256]\
            .mul(q_content_modulate[..., 128:]).view(num_queries, bs, self.nheads, c//(2*self.nheads))

        q_content = tgt.view(num_queries, bs, self.nheads, c//self.nheads)
        q = torch.cat([q_content, q_pos_x, q_pos_y], dim=-1).view(num_queries, bs, c * 2)

        k_content = memory.view(-1, bs, self.nheads, c//self.nheads)
        k_pos_x = pos[..., :128].view(-1, bs, self.nheads, c//(2*self.nheads))
        k_pos_y = pos[..., 128:].view(-1, bs, self.nheads, c//(2*self.nheads))
        k = torch.cat([k_content, k_pos_x, k_pos_y], dim=-1).view(-1, bs, c * 2)

        tgt2 = self.cross_attn(query=q, key=k, value=memory, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask, gaussian=gaussian)[0]

        # ========== End of Cross-Attention =============
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout88(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        anchor_box_offset = None
        if self.tgt_offset_proj:
            anchor_box_offset= self.tgt_offset_proj(tgt)

        return tgt, anchor_box_offset

