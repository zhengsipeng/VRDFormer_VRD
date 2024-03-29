# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import math
import torch
from torch import nn
from torch.nn.init import constant_, normal_, xavier_uniform_
from .ops.modules import MSDeformAttn
from util.compute import inverse_sigmoid
from torchvision.ops import roi_align
from models.transformer import _get_clones, _get_activation_fn
        
        
class DeformableTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, num_queries=100, dim_feedforward=2048, 
                 dropout=0.1, activation="relu", normalize_before=False,
                 return_intermediate_dec=False, num_feature_levels=4 , dec_n_points=4,  enc_n_points=4,
                 multi_frame_attention_separate_encoder=False, stage=1):
        super().__init__()
        self.stage = stage
        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels
        self.multi_frame_attention_separate_encoder = multi_frame_attention_separate_encoder
        
        enc_num_feature_levels = num_feature_levels
        if multi_frame_attention_separate_encoder:
            enc_num_feature_levels = enc_num_feature_levels // 2

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          enc_num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        
        num_feature_levels = enc_num_feature_levels if self.stage==2 else num_feature_levels
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        
        #num_lvl = num_feature_levels if stage==1 else num_feature_levels*2
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        
        self.reference_points = nn.Linear(d_model, 2)
        
        self._reset_parameters()

        if self.stage==2:
            # Relation Module
            self.num_queries = num_queries
            self.so_linear = nn.Linear(self.d_model*2, self.d_model)

            # ROI Pooling
            self.roi_output_scales = [[7,7]] #[[7, 7], [7, 7], [7, 7], [3, 3]]
            self.downsample_scales = [32] #[8,16,32,64]
            self.roi_pool_type = "avg"
            if self.roi_pool_type == 'avg':
                self.roi_pool_layer = nn.AvgPool2d(kernel_size=self.roi_output_scales[-1])
            elif self.roi_pool_type == "max":
                self.roi_pool_layer = nn.MaxPool2d(kernel_size=self.roi_output_scales[-1])
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)
    
    def extract_roi_feat(self, src, boxes):
        box_fts = roi_align(src, 
                            torch.cat([torch.full((len(boxes), 1), 0).cuda(), boxes], dim=1), 
                            self.roi_output_scales[-1], 
                            spatial_scale=1./self.downsample_scales[-1], 
                            sampling_ratio=-1
                        )  
        return self.roi_pool_layer(box_fts).squeeze(2).squeeze(2)
    
    def prepare_tag_query(self, so_embed, targets):
        # during relation tag, the query embed is initialized by roi feats of boxes
        num_svo = targets["num_inst"]
        query_sboxes = torch.zeros((self.num_queries, 4)).cuda()
        query_oboxes = torch.zeros((self.num_queries, 4)).cuda()
        query_embed = torch.zeros((self.num_queries, self.d_model)).cuda()
        query_masks = torch.ones(self.num_queries).bool().cuda()

        query_embed[:num_svo] = so_embed
        query_embed = query_embed.unsqueeze(0)

        query_sboxes[:num_svo] = targets["sub_boxes"]
        query_oboxes[:num_svo] = targets["obj_boxes"]
        
        query_masks[:num_svo] = 0
        query_masks = query_masks.unsqueeze(0)

        return query_embed
    
    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos
    
    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    
    def forward(self, srcs, masks, pos_embeds, query_embed=None, targets=None):
        
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape) 
            src = src.flatten(2).transpose(1, 2)  # 2，HW,288
            mask = mask.flatten(1)  # 2,HW
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # 2,288,HW
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # lvl_pos_embed = pos_embed + self.level_embed[lvl % self.num_feature_levels].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        
        src_flatten = torch.cat(src_flatten, 1)  # 2,HWx8,288
        mask_flatten = torch.cat(mask_flatten, 1)
        
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        
        # encoder
        if self.multi_frame_attention_separate_encoder and self.stage==1:   
            prev_memory = self.encoder(
                src_flatten[:, :src_flatten.shape[1] // 2],
                spatial_shapes[:self.num_feature_levels // 2],
                valid_ratios[:, :self.num_feature_levels // 2],
                lvl_pos_embed_flatten[:, :src_flatten.shape[1] // 2],
                mask_flatten[:, :src_flatten.shape[1] // 2])
            
            memory = self.encoder(
                src_flatten[:, src_flatten.shape[1] // 2:],
                spatial_shapes[self.num_feature_levels // 2:],
                valid_ratios[:, self.num_feature_levels // 2:],
                lvl_pos_embed_flatten[:, src_flatten.shape[1] // 2:],
                mask_flatten[:, src_flatten.shape[1] // 2:])
       
            memory = torch.cat([memory, prev_memory], 1)
        else:
            memory = self.encoder(src_flatten, spatial_shapes, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        # memory: 2,HxWx4x2, C
        
        if self.stage == 2:

            # individual s_embed and o_embed are extracted from the encoder
            s_embed = self.extract_roi_feat(srcs[-1], targets["unscaled_sub_boxes"])  
            o_embed = self.extract_roi_feat(srcs[-1], targets["unscaled_obj_boxes"])
            so_embed = self.so_linear(torch.cat([s_embed, o_embed], dim=1)) 
            query_embed = self.prepare_tag_query(so_embed, targets)
            #query_embed = query_embed.permute(1, 0, 2)
            
            tgt = torch.zeros_like(query_embed)
            reference_points = self.reference_points(query_embed).sigmoid()  
            init_reference_out = reference_points
            
            query_attn_mask = None

            hs, inter_references = self.decoder(
                tgt, reference_points, memory, spatial_shapes,
                valid_ratios, query_embed, mask_flatten, query_attn_mask) # hs: 6,2,500,288
            
            return hs, s_embed, o_embed
     
        # prepare input for decoder
        bs, _, c = memory.shape
        query_attn_mask = None

        query_embed, tgt = torch.split(query_embed, c, dim=1) # 500,C
            
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1) # 2,500,C
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        
        reference_points = self.reference_points(query_embed).sigmoid()  # 2,500,2
        
        if targets is not None and 'track_query_hs_embeds' in targets[0]:
            prev_hs_embed = torch.stack([t['track_query_hs_embeds'] for t in targets])
            prev_sub_boxes = torch.stack([t['track_query_sub_boxes'] for t in targets])
            prev_obj_boxes = torch.stack([t['track_query_obj_boxes'] for t in targets])
            
            prev_query_embed = torch.zeros_like(prev_hs_embed)
            
            prev_tgt = prev_hs_embed
            
            query_embed = torch.cat([prev_query_embed, query_embed], dim=1)
            tgt = torch.cat([prev_tgt, tgt], dim=1)
            
            prev_boxes = (prev_sub_boxes[..., :2] + prev_obj_boxes[..., :2]) / 2
            reference_points = torch.cat([prev_boxes, reference_points], dim=1)
        
        init_reference_out = reference_points
        
        hs, inter_references = self.decoder(
            tgt, reference_points, memory, spatial_shapes,
            valid_ratios, query_embed, mask_flatten, query_attn_mask) # hs: 6,2,500,288
        
        inter_references_out = inter_references

        return hs, memory, init_reference_out, inter_references_out
    
    
class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, padding_mask)
       
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, valid_ratios, pos=None, padding_mask=None):
        
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, src_padding_mask=None, query_attn_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), key_padding_mask=query_attn_mask)[0].transpose(0, 1)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, src_padding_mask, query_attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.sub_bbox_embed = None
        self.obj_bbox_embed = None
        
        self.sub_class_embed = None
        self.obj_class_embed = None
        
        self.verb_class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_valid_ratios,
                query_pos=None, src_padding_mask=None, query_attn_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_padding_mask, query_attn_mask)

            # hack implementation for iterative bounding box refinement
            if self.sub_bbox_embed is not None:
                tmp_s = self.sub_bbox_embed[lid](output)
                tmp_o = self.obj_bbox_embed[lid](output)
                
                if reference_points.shape[-1] == 4:
                    new_sub_reference_points = tmp_s + inverse_sigmoid(reference_points)
                    new_sub_reference_points = new_sub_reference_points.sigmoid()
                    
                    new_obj_reference_points = tmp_o + inverse_sigmoid(reference_points)
                    new_obj_reference_points = new_obj_reference_points.sigmoid()
                    
                else:
                    assert reference_points.shape[-1] == 2
                    new_sub_reference_points = tmp_s
                    new_sub_reference_points[..., :2] = tmp_s[..., :2] + inverse_sigmoid(reference_points)
                    new_sub_reference_points = new_sub_reference_points.sigmoid()
                    
                    new_obj_reference_points = tmp_o
                    new_obj_reference_points[..., :2] = tmp_o[..., :2] + inverse_sigmoid(reference_points)
                    new_obj_reference_points = new_obj_reference_points.sigmoid()
                
                new_reference_points = (new_sub_reference_points+new_obj_reference_points)/2
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
       
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        
        return output, reference_points
       
        
def build_deformable_transformer(args):
    num_feature_levels = args.num_feature_levels
    if args.multi_frame_attention:
        num_feature_levels *= 2
                
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        num_feature_levels=num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        multi_frame_attention_separate_encoder=args.multi_frame_attention_separate_encoder,
        stage=args.stage
    )