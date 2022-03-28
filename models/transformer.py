# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn

from models.transformer_encoder import TransformerEncoder, TransformerEncoderLayer
from models.transformer_decoder import TransformerDecoder, TransformerDecoderLayer


class Transformer(nn.Module):
    def __init__(self, args, activation="relu"):
        super().__init__()
        self.args = args
        self.multiscale = args.multiscale
        self.d_model = args.hidden_dim
        self.nheads = args.nheads
        self.num_queries = args.num_queries
        self.enc_layers = args.enc_layers
        self.dec_layers = args.dec_layers
        self.dim_feedforward = args.dim_feedforward
        self.dropout = args.dropout

        if self.multiscale:
            # Reminder: To use multiscale DAB-DETR, you need to compile CUDA operators for Deformable Attention.
            from models.transformer_encoder import DeformableTransformerEncoder, DeformableTransformerEncoderLayer
            self.num_feature_levels = 3  # Hard-coded multiscale parameters
            # Use Deformable Attention in Transformer Encoder for efficient computation of multiscale features
            encoder_layer = DeformableTransformerEncoderLayer(args, activation)
            self.encoder = DeformableTransformerEncoder(args, encoder_layer, self.enc_layers)
            self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, self.d_model))
        else:
            encoder_layer = TransformerEncoderLayer(args, activation)
            self.encoder = TransformerEncoder(args, encoder_layer, self.enc_layers)

        decoder_layer = TransformerDecoderLayer(args, activation)
        self.decoder = TransformerDecoder(args, decoder_layer, self.dec_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.multiscale:
            from models.deformable_attn_ops.modules import MSDeformAttn
            for m in self.modules():
                if isinstance(m, MSDeformAttn):
                    m._reset_parameters()
            nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, query_embed, pos_embeds):
        if self.multiscale:
            return self.forward_multi_scale(srcs, masks, query_embed, pos_embeds)
        else:
            return self.forward_single_scale(srcs[0], masks[0], query_embed, pos_embeds[0])

    def forward_single_scale(self, src, mask, query_embed, pos):
        bs, c, memory_h, memory_w = src.shape

        grid_y, grid_x = torch.meshgrid(torch.arange(0, memory_h), torch.arange(0, memory_w))
        grid = torch.stack((grid_x, grid_y), 2).float().to(src.device)
        grid = grid.reshape(-1, 2).unsqueeze(0).repeat(bs, 1, 1) # [bs, hw, 2]

        src = src.flatten(2).permute(2, 0, 1)  # flatten NxCxHxW to HWxNxC
        pos = pos.flatten(2).permute(2, 0, 1)  # flatten NxCxHxW to HWxNxC
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros(self.num_queries, bs, c, device=query_embed.device)

        # encoder
        memory = self.encoder(src,
                              src_key_padding_mask=mask,
                              pos=pos)

        # decoder
        hs, references = self.decoder(tgt,
                                      memory,
                                      memory_key_padding_mask=mask,
                                      pos=pos,
                                      query_pos=query_embed,
                                      memory_w=memory_w,
                                      memory_h=memory_h,
                                      grid=grid)
        return hs, references

    def forward_multi_scale(self, srcs, masks, query_embed, pos_embeds):

        orig_pos_embed_16 = pos_embeds[0] + self.level_embed[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        pos_embed_16 = orig_pos_embed_16.flatten(2).permute(2, 0, 1)
        mask_16 = masks[0].flatten(1)

        orig_pos_embed_32 = pos_embeds[1] + self.level_embed[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        pos_embed_32 = orig_pos_embed_32.flatten(2).permute(2, 0, 1)
        mask_32 = masks[1].flatten(1)

        orig_pos_embed_64 = pos_embeds[2] + self.level_embed[2].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        pos_embed_64 = orig_pos_embed_64.flatten(2).permute(2, 0, 1)
        mask_64 = masks[2].flatten(1)

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, _, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten,
                              spatial_shapes,
                              level_start_index,
                              valid_ratios,
                              lvl_pos_embed_flatten,
                              mask_flatten)

        # prepare input for decoder
        tgt = torch.zeros(self.num_queries, bs, self.d_model, device=query_embed.device)
        mem1 = memory[:, level_start_index[0]: level_start_index[1], :]
        mem2 = memory[:, level_start_index[1]: level_start_index[2], :]
        mem3 = memory[:, level_start_index[2]:, :]
        memory_flatten = []
        for m in [mem1, mem2, mem3]:
            memory_flatten.append(m.permute(1, 0, 2))
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        # decoder
        hs, references = self.decoder(tgt,
                                      [memory_flatten[0], memory_flatten[1], memory_flatten[2]],
                                      memory_key_padding_mask=[mask_16, mask_32, mask_64],
                                      pos=[pos_embed_16, pos_embed_32, pos_embed_64],
                                      query_pos=query_embed)

        return hs, references


def build_transformer(args):
    return Transformer(args, activation="relu")
