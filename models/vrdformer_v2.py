import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import build_backbone
from .matcher import build_matcher
from .base import VRDFormerBase, MLP
from util.misc import NestedTensor, accuracy, multi_label_acc, inverse_sigmoid


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class VRDFormer(nn.Module):
    def __init__(self, 
                 backbone, 
                 transformer, 
                 num_obj_class, 
                 num_verb_class, 
                 num_queries, 
                 num_feature_levels,
                 aux_loss=True, 
                 task="tagging",
                 deformable = False,
                 with_box_refine=False,
                 multi_frame_attention=False, 
                 multi_frame_encoding=False, 
                 merge_frame_features=False
        ):
        """ Initializes the model.
        Parameters:
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.merge_frame_features = merge_frame_features
        self.multi_frame_attention = multi_frame_attention
        self.multi_frame_encoding = multi_frame_encoding
        self.deformable = deformable
        self.with_box_refine = with_box_refine
        self.num_feature_levels = num_feature_levels
        
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.task = task
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        
        self.sub_class_embed = nn.Linear(hidden_dim, num_obj_class+1)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_class+1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_class+1)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        num_channels = backbone.num_channels[-3:]
        if num_feature_levels > 1:
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            num_backbone_outs = len(backbone.strides) - 1

            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channels = num_channels[i]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )])
        
        # prepare prediction head
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.sub_class_embed.bias.data = torch.ones_like(self.sub_class_embed.bias) * bias_value
        self.obj_class_embed.bias.data = torch.ones_like(self.obj_class_embed.bias) * bias_value
        self.verb_class_embed.bias.data = torch.ones_like(self.verb_class_embed.bias) * bias_value
        nn.init.constant_(self.sub_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.obj_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.obj_bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        
        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.sub_class_embed = _get_clones(self.sub_class_embed, num_pred)
            self.obj_class_embed = _get_clones(self.obj_class_embed, num_pred)
            self.verb_class_embed = _get_clones(self.verb_class_embed, num_pred)
            self.sub_bbox_embed = _get_clones(self.sub_bbox_embed, num_pred)
            self.obj_bbox_embed = _get_clones(self.obj_bbox_embed, num_pred)
            nn.init.constant_(self.sub_bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.obj_bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.sub_bbox_embed = self.sub_bbox_embed
            self.transformer.decoder.obj_bbox_embed = self.obj_bbox_embed
        else:
            nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.obj_bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.sub_class_embed = nn.ModuleList([self.sub_class_embed for _ in range(num_pred)])
            self.obj_class_embed = nn.ModuleList([self.obj_class_embed for _ in range(num_pred)])
            self.verb_class_embed = nn.ModuleList([self.verb_class_embed for _ in range(num_pred)])
            self.sub_bbox_embed = nn.ModuleList([self.sub_bbox_embed for _ in range(num_pred)])
            self.obj_bbox_embed = nn.ModuleList([self.obj_bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.sub_bbox_embed = None
            self.transformer.decoder.obj_bbox_embed = None
            
        if self.merge_frame_features:
            self.merge_features = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=1)
            self.merge_features = _get_clones(self.merge_features, num_feature_levels)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_sub_class, outputs_sub_coord, outputs_obj_class, outputs_obj_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_sub_logits': a, 'pred_sub_boxes': b, 'pred_obj_logits': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_sub_class[:-1], outputs_sub_coord[:-1], outputs_obj_class[:-1], outputs_obj_coord[:-1])]

    def forward(self, samples: NestedTensor, targets: list = None, prev_features=None): 
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        features, pos = self.backbone(samples)  # [2,2048,20,24], [2,256,20,24]
        features_all = features # 2, C, H, W
        features = features[-3:]
        
        if prev_features is None:
            prev_features = features
        else:
            prev_features = prev_features[-3:]
        
        src_list = []
        mask_list = []
        pos_list = []
        
        frame_features = [prev_features, features]  # 2,3,2,C,H,W
        if not self.multi_frame_attention:
            frame_features = [features]
            
        for frame, frame_feat in enumerate(frame_features):
            if self.multi_frame_attention and self.multi_frame_encoding:
                pos_list.extend([p[:, frame] for p in pos[-3:]])  # 3,2,c,H,W
            else:
                pos_list.extend(pos[-3:])
            for l, feat in enumerate(frame_feat): 
                src, mask = feat.decompose()
                
                if self.merge_frame_features:
                    prev_src, _ = prev_features[l].decompose()
                    src_list.append(self.merge_features[l](torch.cat([self.input_proj[l](src), 
                                                                      self.input_proj[l](prev_src)], dim=1)))
                else:
                    src_list.append(self.input_proj[l](src))

                mask_list.append(mask)
                assert mask is not None
            
            if self.num_feature_levels > len(frame_feat):
                _len_srcs = len(frame_feat)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        if self.merge_frame_features:
                            src = self.merge_features[l](torch.cat([self.input_proj[l](frame_feat[-1].tensors), self.input_proj[l](prev_features[-1].tensors)], dim=1))
                        else:
                            src = self.input_proj[l](frame_feat[-1].tensors)
                    else:
                        src = self.input_proj[l](src_list[-1])
           
                    _, m = frame_feat[0].decompose()
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]

                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    src_list.append(src)
                    mask_list.append(mask)
                    if self.multi_frame_attention and self.multi_frame_encoding:
                        pos_list.append(pos_l[:, frame])
                    else:
                        pos_list.append(pos_l)
                        
        
        query_embed = self.query_embed.weight
        
        hs, memory, init_reference, inter_references = self.transformer(
                                    src_list, mask_list, pos_list, 
                                    query_embed, 
                                    targets)
        if not self.deformable:
            memory = memory.transpose(0, 1)
            
        outputs_sub_classes, outputs_obj_classes, outputs_verb_classes = [], [], []
        outputs_sub_coords, outputs_obj_coords = [], []
        
        for lvl in range(hs.shape[0]):
            if self.deformable:
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
            
            outputs_sub_class = self.sub_class_embed[lvl](hs[lvl])
            outputs_obj_class = self.obj_class_embed[lvl](hs[lvl])
            outputs_verb_class = self.verb_class_embed[lvl](hs[lvl])
            
            tmp_s = self.sub_bbox_embed[lvl](hs[lvl])
            tmp_o = self.obj_bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp_s += reference
                tmp_o += reference
            else:
                assert reference.shape[-1] == 2
                tmp_s[..., :2] += reference
                tmp_o[..., :2] += reference
            
            outputs_sub_coord = tmp_s.sigmoid()
            outputs_obj_coord = tmp_o.sigmoid()
            outputs_sub_classes.append(outputs_sub_class)
            outputs_obj_classes.append(outputs_obj_class)
            outputs_verb_classes.append(outputs_verb_class)
            outputs_sub_coords.append(outputs_sub_coord)
            outputs_obj_coords.append(outputs_obj_coord)
        
        outputs_sub_class = torch.stack(outputs_sub_classes)
        outputs_obj_class = torch.stack(outputs_obj_classes)
        outputs_verb_class = torch.stack(outputs_verb_classes)
        
        outputs_sub_coord = torch.stack(outputs_sub_coords)
        outputs_obj_coord = torch.stack(outputs_obj_coords)
        
        out = {'pred_sub_logits': outputs_sub_class[-1],
               'pred_obj_logits': outputs_obj_class[-1],
               'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1],
               'pred_obj_boxes': outputs_obj_coord[-1],
               'hs_embed': hs[-1]}
           
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_sub_class, outputs_obj_class, outputs_verb_class,
                                                    outputs_sub_coord, outputs_obj_coord)

        offset = 0
        memory_slices = []
        batch_size, _, channels = memory.shape
        assert batch_size < 10
        for src in src_list:
            _, _, height, width = src.shape
            memory_slice = memory[:, offset:offset + height * width].permute(0, 2, 1).view(
                batch_size, channels, height, width)
            memory_slices.append(memory_slice)
            offset += height * width
        memory = memory_slices
        import pdb;pdb.set_trace()
        return out, targets, features_all, memory, hs
        
        
    @torch.jit.unused
    def _set_aux_loss(self, outputs_sub_class, outputs_obj_class, outputs_verb_class,
                            outputs_sub_coord, outputs_obj_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_sub_logits': a, 'pred_obj_logits': b, 'pred_verb_logits': c, 
                 'pred_sub_boxes': d, 'pred_obj_boxes': e} for a, b, c, d, e in 
                 zip(outputs_sub_class[:-1], outputs_obj_class[:-1], outputs_verb_class[:-1], 
                     outputs_sub_coord[:-1], outputs_obj_coord[:-1])]
        
        
class SetCriterion(nn.Module):
    def __init__(self, 
                 num_obj_class, 
                 matcher, 
                 weight_dict, 
                 eos_coef, 
                 losses,
                 track_query_false_positive_eos_weight
                ):
        super().__init__()
        self.num_obj_class = num_obj_class
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_class + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.track_query_false_positive_eos_weight = track_query_false_positive_eos_weight

    def loss_sub_labels(self, outputs, log=True):
        assert "pred_sub_logits" in outputs
        src_logits, sub_labels = outputs["pred_sub_logits"], outputs["label_sub_classes"]
        losses = {"loss_sub_ce": F.cross_entropy(src_logits, sub_labels)}
        if log:
            losses["sub_class_acc"] = accuracy(src_logits, sub_labels)[0]
        
        return losses

    def loss_obj_labels(self, outputs, log=True):
        assert "pred_obj_logits" in outputs
        src_logits, obj_labels = outputs["pred_obj_logits"], outputs["label_obj_classes"]
        losses = {"loss_obj_ce": F.cross_entropy(src_logits, obj_labels)}
        if log:
            losses["obj_class_acc"] = accuracy(src_logits, obj_labels)[0]
        return losses

    def loss_verb_labels(self, outputs, log=True):
        assert "pred_verb_logits" in outputs
        src_logits, verb_labels = outputs["pred_verb_logits"], outputs["label_verb_classes"]
        loss_verb_ce = F.binary_cross_entropy_with_logits(src_logits, verb_labels)
        losses = {"loss_verb_ce": loss_verb_ce}
        if log:
            losses["verb_class_acc"] = multi_label_acc(src_logits, verb_labels)
        return losses

    def get_loss(self, loss, outputs, **kwards):
        loss_map = {
                    "sub_labels": self.loss_sub_labels,
                    "obj_labels": self.loss_obj_labels, 
                    "verb_labels": self.loss_verb_labels
                    }
        assert loss in loss_map, f"do you really wnat to compute {loss} loss?"
        return loss_map[loss](outputs, **kwards)

    def forward(self, outputs):
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs))
        return losses


def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)
    if args.deformable:
        from .deformable_transformer import build_transformer
    else:
        from .transformer_v2 import build_transformer
    transformer = build_transformer(args)
    matcher = None
    #matcher = build_matcher(args) if args.task=="detection" else None

    num_obj_class = 80 if args.dataset == 'vidor' else 35
    num_verb_class = 42 if args.dataset == 'vidor' else 132  # multi-label

    model = VRDFormer(
        backbone,
        transformer,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        num_obj_class=num_obj_class,
        num_verb_class=num_verb_class,
        task=args.task,
        deformable=args.deformable,
        with_box_refine=args.with_box_refine,
        multi_frame_attention=args.multi_frame_attention,
        multi_frame_encoding=args.multi_frame_encoding,
        merge_frame_features=args.merge_frame_features
    )
    #track_query_false_positive_prob=args.track_query_false_positive_prob,
    #track_query_false_negative_prob=args.track_query_false_negative_prob,
    #track_query_noise=args.track_query_noise,
    weight_dict = {'loss_sub_ce': args.sub_loss_coef,
                   'loss_obj_ce': args.obj_loss_coef,
                   'loss_verb_ce': args.verb_loss_coef,}

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    losses = ["sub_labels", "obj_labels", "verb_labels"]
    

    criterion = SetCriterion(
        num_obj_class,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        track_query_false_positive_eos_weight=args.track_query_false_positive_eos_weight,
    )
    '''
    criterion = SetCriterionDet(
        num_obj_class,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        track_query_false_positive_eos_weight=args.track_query_false_positive_eos_weight,
        focal_loss=args.focal_loss,
        focal_alpha=args.focal_alpha)
    '''
    criterion.to(device)

    return model, criterion, weight_dict