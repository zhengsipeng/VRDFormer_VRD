import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import MLP
from util.misc import NestedTensor
from util.compute import accuracy, multi_label_acc
from util.misc import NestedTensor, sigmoid_focal_loss
from models.vrdformer import VRDFormer


class VRDFormer_S2(VRDFormer):
    def __init__(self, 
                 backbone, 
                 transformer, 
                 num_obj_classes, 
                 num_verb_classes, 
                 num_queries, 
                 num_feature_levels,
                 aux_loss=True, 
                 deformable = False,
                 with_box_refine=False,
                 overflow_boxes=False,
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
        super().__init__(backbone, transformer, num_obj_classes, num_verb_classes, num_queries, num_feature_levels,
                         aux_loss, deformable, with_box_refine, overflow_boxes, 
                         multi_frame_attention, multi_frame_encoding, merge_frame_features)

    def memory_update(self, outputs, targets, memory, is_eval=False):
        if memory is None:
            memory = {"sub_labels": {}, "obj_labels": {}, "verb_labels": {},
                      "s_embed": {}, "o_embed": {}, "rel_embed": {},
                      "frame_ids": {}}
        
        num_svo = targets["sub_boxes"].shape[0]
        s_embed = outputs["s_embed"][:num_svo]
        o_embed = outputs['o_embed'][:num_svo]
        hs_embed = outputs["rel_embed"][0][:num_svo]
        
        for i in range(targets["so_track_ids"].shape[0]):
            so_tid = targets['so_track_ids'][i]
            so_tid = "%s-%s"%(so_tid[0].cpu().item(), so_tid[1].cpu().item())

            if so_tid in memory["rel_embed"]:
                if is_eval:
                    memory['frame_ids'][so_tid].append(targets['frame_id'])
                    memory['rel_embed'][so_tid] = torch.cat([memory['rel_embed'][so_tid], hs_embed[i].unsqueeze(0).detach()])
                    memory['s_embed'][so_tid] = torch.cat([memory['s_embed'][so_tid], s_embed[i].unsqueeze(0).detach()])
                    memory['o_embed'][so_tid] = torch.cat([memory['o_embed'][so_tid], o_embed[i].unsqueeze(0).detach()])
                else:
                    memory['rel_embed'][so_tid] = torch.cat([memory['rel_embed'][so_tid], hs_embed[i].unsqueeze(0)])
                    memory['s_embed'][so_tid] = torch.cat([memory['s_embed'][so_tid], s_embed[i].unsqueeze(0)])
                    memory['o_embed'][so_tid] = torch.cat([memory['o_embed'][so_tid], o_embed[i].unsqueeze(0)])
            
                memory["verb_labels"][so_tid] = torch.cat([memory["verb_labels"][so_tid], targets["verb_labels"][i].unsqueeze(0)])  
            else:
                if is_eval:
                    memory['frame_ids'][so_tid] = [targets['frame_id']]       
                    memory["rel_embed"][so_tid] = hs_embed[i].unsqueeze(0).detach()
                    memory["s_embed"][so_tid] = s_embed[i].unsqueeze(0).detach()
                    memory["o_embed"][so_tid] = o_embed[i].unsqueeze(0).detach()     
                else:
                    memory["rel_embed"][so_tid] = hs_embed[i].unsqueeze(0)
                    memory["s_embed"][so_tid] = s_embed[i].unsqueeze(0)
                    memory["o_embed"][so_tid] = o_embed[i].unsqueeze(0)

                memory["verb_labels"][so_tid] = targets["verb_labels"][i].unsqueeze(0)
                memory['sub_labels'][so_tid] = targets["sub_labels"][i]
                memory['obj_labels'][so_tid] = targets['obj_labels'][i]
        
        return memory

    def relation_classifier(self, memory, gt=None, is_eval=False):
        if is_eval:
            verb_preds, verb_scores = [], []
            for gt_triplet in gt:
                so_id = "%d-%d"%(gt_triplet['subject_tid'], gt_triplet['object_tid'])  
                gt_tids = [i for i in range(gt_triplet['duration'][0], gt_triplet['duration'][1])]
                frame_ids = [memory['frame_ids'][so_id].index(gt_tid) for gt_tid in gt_tids]
                rel_embeds = memory['rel_embed'][so_id][frame_ids]
                verb_logits = self.verb_class_embed[-1](rel_embeds.mean(0, keepdim=True))
                verb_ind = verb_logits.argmax().item()
                verb_score = verb_logits.max().item()
                verb_preds.append(verb_ind)
                verb_scores.append(verb_score)
            return [verb_preds, verb_scores]
        
        rel_hs = memory["rel_embed"]
        labels = memory["verb_labels"]
        rel_embeds = []
        sub_labels, verb_labels, obj_labels = [], [], []
        for so_tid in rel_hs.keys():
            rel_embeds.append(rel_hs[so_tid].mean(0, keepdim=True) )  # mean pooling
            sub_labels.append(memory["sub_labels"][so_tid])
            obj_labels.append(memory["obj_labels"][so_tid])
            verb_labels.append(labels[so_tid][-1])

        rel_embeds = torch.cat(rel_embeds, dim=0)
        sub_labels = torch.stack(sub_labels)
        verb_labels = torch.stack(verb_labels)
        obj_labels = torch.stack(obj_labels)
        
        sub_logits = self.sub_class_embed[-1](rel_embeds)
        verb_logits = self.verb_class_embed[-1](rel_embeds)
        obj_logits = self.obj_class_embed[-1](rel_embeds)
        
        return sub_logits, verb_logits, obj_logits, sub_labels, verb_labels, obj_labels

    @torch.jit.unused
    def _set_aux_loss(self, outputs_sub_class, outputs_sub_coord, outputs_obj_class, outputs_obj_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_sub_logits': a, 'pred_sub_boxes': b, 'pred_obj_logits': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_sub_class[:-1], outputs_sub_coord[:-1], outputs_obj_class[:-1], outputs_obj_coord[:-1])]

    def forward(self, samples, targets, memory, eos=False, is_eval=False): 
        # eos: end_of_sequence
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)  
        
        features, pos = self.backbone(samples)
        features = features[-3:]

        src_list = []
        mask_list = []
        pos_list = []

        pos_list.extend(pos[-3:])
        for l, feat in enumerate(features): 
            src, mask = feat.decompose()
            
            src_list.append(self.input_proj[l](src))
            mask_list.append(mask)
            assert mask is not None
        
        if self.num_feature_levels > len(src_list):
            _len_srcs = len(src_list)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](src_list[-1])

                _, m = features[0].decompose()
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]

                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                src_list.append(src)
                mask_list.append(mask)
                pos_list.append(pos_l)
        
        query_embed = None  # initialize from gt boxes
        
        hs, s_embed, o_embed = self.transformer(
                                    src_list, 
                                    mask_list, 
                                    pos_list, 
                                    query_embed, 
                                    targets)
        
        hs = hs[-1]  # bs,num_queries,dim
        
        outputs_sub_coord = self.sub_bbox_embed[-1](hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed[-1](hs).sigmoid()
        
        out = {"rel_embed": hs, 
               's_embed': s_embed, 
               'o_embed': o_embed,
               'pred_sub_boxes': outputs_sub_coord[-1], 
               'pred_obj_boxes': outputs_obj_coord[-1]}
        
        memory = self.memory_update(out, targets, memory, is_eval=is_eval)

        if eos and not is_eval:
            memory["pred_sub_logits"], memory["pred_verb_logits"], memory["pred_obj_logits"], \
            memory["label_sub_classes"] , memory["label_verb_classes"], memory["label_obj_classes"] \
                = self.relation_classifier(memory)
                
        return memory
        

class SetCriterion(nn.Module):
    def __init__(self, 
                 num_obj_classes, 
                 num_verb_classes,
                 matcher, 
                 weight_dict, 
                 eos_coef, 
                 losses,
                 focal_loss, 
                 focal_alpha, 
                 focal_gamma,
                 track_query_false_positive_eos_weight
                ):
        super().__init__()
        self.num_obj_class = num_obj_classes
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.focal_loss = focal_loss
        empty_weight = torch.ones(self.num_obj_class + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def loss_labels(self, outputs, log=True):
        assert "pred_sub_logits" in outputs and "pred_obj_logits" in outputs
        
        losses = {}
        for role in ["sub", "obj"]:
            src_logits = outputs["pred_%s_logits"%role]

            target_classes = torch.as_tensor([t for t in outputs["%s_labels"%role].values()], device=src_logits.device)
            
            loss_ce = F.cross_entropy(src_logits, target_classes)
            
            if log:
                losses["%s_class_error"%role] = \
                            100 - accuracy(src_logits, target_classes)[0]

            losses['loss_ce_%s'%role] = loss_ce
        
        losses['loss_ce'] = (losses['loss_ce_sub']+losses['loss_ce_obj'])/2
        del losses['loss_ce_sub']
        del losses['loss_ce_obj']

        return losses

    def loss_labels_focal(self, outputs, log=True):
        assert "pred_sub_logits" in outputs and "pred_obj_logits" in outputs
        losses = {}
        for role in ["sub", "obj"]:
            src_logits = outputs['pred_%s_logits'%role]
            
            target_classes = torch.as_tensor([t for t in outputs["%s_labels"%role].values()], device=src_logits.device)
            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1] + 1], dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(1, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:,:-1]
            num_trajs = src_logits.shape[0]

            loss_ce = sigmoid_focal_loss(
                src_logits, target_classes_onehot, num_trajs,
                alpha=self.focal_alpha, gamma=self.focal_gamma)
            
            loss_ce *= src_logits.shape[0]

            losses['loss_ce_%s'%role] = loss_ce

            if log:
                # TODO this should probably be a separate loss, not hacked in this one here
                losses['%s_class_error'%role] = \
                        100 - accuracy(src_logits, target_classes)[0]
        
        losses['loss_ce'] = (losses['loss_ce_sub']+losses['loss_ce_obj'])/2
        del losses['loss_ce_sub']
        del losses['loss_ce_obj']    
        
        return losses
        

    def loss_verb_labels(self, outputs, log=True):
        assert "pred_verb_logits" in outputs
        
        src_logits = outputs["pred_verb_logits"]
        target_classes_v = outputs["label_verb_classes"]

        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_v, num_trajs=src_logits.shape[0],
            alpha=self.focal_alpha, gamma=self.focal_gamma
        )
        loss_ce *= src_logits.shape[0]
        losses = {'loss_ce_verb': loss_ce}
        

        if log:
            losses["verb_class_error"] = 100 - multi_label_acc(
                src_logits, target_classes_v)
        return losses

    def get_loss(self, loss, outputs, **kwards):
        loss_map = {
                    "labels": self.loss_labels_focal if self.focal_loss else self.loss_labels,
                    "verb_labels": self.loss_verb_labels
                    }
        assert loss in loss_map, f"do you really wnat to compute {loss} loss?"
        return loss_map[loss](outputs, **kwards)

    def forward(self, outputs):
        
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs))
        return losses

