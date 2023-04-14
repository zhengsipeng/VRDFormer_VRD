import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import MLP
from util.misc import accuracy, multi_label_acc


class VRDFormer(nn.Module):
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
        super().__init__()

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        
        self.sub_class_embed = nn.Linear(hidden_dim, num_obj_classes+1)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes+1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        #import pdb;pdb.set_trace()
        self.input_proj = nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.transformer.decoder.sbox_emb = None
        self.transformer.decoder.obox_emb = None

    def memory_update(self, outputs, targets, memory, is_eval=False):
        if memory is None:
            memory = {"label_sub": {}, "label_obj": {}, "label_verb": {},
                      "s_embed": {}, "o_embed": {}, "rel_embed": {},
                      "frame_ids": {}}
        
        num_svo = targets["sboxes"].shape[0]
        s_embed = outputs["s_embed"][:num_svo]
        o_embed = outputs['o_embed'][:num_svo]
        hs_embed = outputs["rel_embed"][0][:num_svo]
        
        for i in range(targets["so_track_ids"].shape[0]):
            so_tid = targets['so_track_ids'][i]
            so_tid = "%s-%s"%(so_tid[0].cpu().item(), so_tid[1].cpu().item())

            if so_tid not in memory["rel_embed"]:
                if not is_eval:
                    memory["rel_embed"][so_tid] = hs_embed[i].unsqueeze(0)
                    memory["s_embed"][so_tid] = s_embed[i].unsqueeze(0)
                    memory["o_embed"][so_tid] = o_embed[i].unsqueeze(0)
                else:
                    memory['frame_ids'][so_tid] = [targets['frame_id']]
                    
                    memory["rel_embed"][so_tid] = hs_embed[i].unsqueeze(0).detach()
                    memory["s_embed"][so_tid] = s_embed[i].unsqueeze(0).detach()
                    memory["o_embed"][so_tid] = o_embed[i].unsqueeze(0).detach()
                    
                memory["label_verb"][so_tid] = targets["verb_category_ids"][i].unsqueeze(0)
                memory['label_sub'][so_tid] = targets["sub_category_ids"][i]
                memory['label_obj'][so_tid] = targets['obj_category_ids'][i]
            else:
                if not is_eval:
                    memory['rel_embed'][so_tid] = torch.cat([memory['rel_embed'][so_tid], hs_embed[i].unsqueeze(0)])
                    memory['s_embed'][so_tid] = torch.cat([memory['s_embed'][so_tid], s_embed[i].unsqueeze(0)])
                    memory['o_embed'][so_tid] = torch.cat([memory['o_embed'][so_tid], o_embed[i].unsqueeze(0)])
                else:
                    memory['frame_ids'][so_tid].append(targets['frame_id'])
                    memory['rel_embed'][so_tid] = torch.cat([memory['rel_embed'][so_tid], hs_embed[i].unsqueeze(0).detach()])
                    memory['s_embed'][so_tid] = torch.cat([memory['s_embed'][so_tid], s_embed[i].unsqueeze(0).detach()])
                    memory['o_embed'][so_tid] = torch.cat([memory['o_embed'][so_tid], o_embed[i].unsqueeze(0).detach()])
                memory["label_verb"][so_tid] = torch.cat([memory["label_verb"][so_tid], targets["verb_category_ids"][i].unsqueeze(0)])  
               
        return memory

    def relation_classifier(self, memory, gt=None, is_eval=False):
        if is_eval:
            verb_preds, verb_scores = [], []
            for gt_triplet in gt:
                so_id = "%d-%d"%(gt_triplet['subject_tid'], gt_triplet['object_tid'])  
                gt_tids = [i for i in range(gt_triplet['duration'][0], gt_triplet['duration'][1])]
                frame_ids = [memory['frame_ids'][so_id].index(gt_tid) for gt_tid in gt_tids]
                rel_embeds = memory['rel_embed'][so_id][frame_ids]
                verb_logits = self.verb_class_embed(rel_embeds.mean(0, keepdim=True))
                verb_ind = verb_logits.argmax().item()
                verb_score = verb_logits.max().item()
                verb_preds.append(verb_ind)
                verb_scores.append(verb_score)
            return [verb_preds, verb_scores]
        
        rel_hs = memory["rel_embed"]
        labels = memory["label_verb"]
        rel_embeds = []
        sub_labels, verb_labels, obj_labels = [], [], []
        for so_tid in rel_hs.keys():
            rel_embeds.append(rel_hs[so_tid].mean(0, keepdim=True) )  # mean pooling
            sub_labels.append(memory["label_sub"][so_tid])
            obj_labels.append(memory["label_obj"][so_tid])
            verb_labels.append(labels[so_tid][-1])
        rel_embeds = torch.cat(rel_embeds, dim=0)
        sub_labels = torch.stack(sub_labels)
        verb_labels = torch.stack(verb_labels)
        obj_labels = torch.stack(obj_labels)
        
        sub_logits = self.sub_class_embed(rel_embeds)
        verb_logits = self.verb_class_embed(rel_embeds)
        obj_logits = self.obj_class_embed(rel_embeds)
     
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
  
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        query_embed = None  # initialize from gt boxes
  
        hs, _, s_embed, o_embed = self.transformer(
                                    self.input_proj(src), 
                                    mask, 
                                    pos[-1], 
                                    query_embed, 
                                    targets)
        hs = hs[-1]  # bs,num_queries,dim

        outputs_sub_coord = self.sub_bbox_embed(hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(hs).sigmoid()
        
        out = {"rel_embed": hs, 's_embed': s_embed, 'o_embed': o_embed,
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        
        memory = self.memory_update(out, targets, memory, is_eval=is_eval)
        
        if eos and not is_eval:
            memory["pred_sub_logits"], memory["pred_verb_logits"], memory["pred_obj_logits"], \
            memory["label_sub_classes"] , memory["label_verb_classes"], memory["label_obj_classes"] \
                = self.relation_classifier(memory)
                
        return memory
        

class SetCriterion(nn.Module):
    def __init__(self, 
                 num_obj_class, 
                 matcher, 
                 weight_dict, 
                 eos_coef, 
                 losses,
                 focal_loss, focal_alpha, focal_gamma,
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

    def loss_labels(self, outputs, log=True):
        assert "pred_sub_logits" in outputs and "pred_obj_logits" in outputs
        
        losses = {}
        for role in ["sub", "obj"]:
            src_logits = outputs["pred_%s_logits"%role]
            labels = outputs["label_%s_classes"%role]
            losses = {"loss_%s_ce"%role: F.cross_entropy(src_logits, labels)}
            if log:
                losses["%s_class_acc"%role] = accuracy(src_logits, labels)[0]
        
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
                    "labels": self.loss_labels,
                    "verb_labels": self.loss_verb_labels
                    }
        assert loss in loss_map, f"do you really wnat to compute {loss} loss?"
        return loss_map[loss](outputs, **kwards)

    def forward(self, outputs):
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs))
        return losses

