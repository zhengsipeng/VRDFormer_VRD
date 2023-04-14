import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from .matcher import HungarianMatcher
from .vrdformer import VRDFormer
from util.misc import (NestedTensor, is_dist_avail_and_initialized, get_world_size,
                       accuracy, multi_label_acc)


class TrackingBase(nn.Module):
    def __init__(self,
                 track_query_false_positive_prob: float = 0.0,
                 track_query_false_negative_prob: float = 0.0,
                 matcher: HungarianMatcher = None,
                 backprop_prev_frame=False):
        self._matcher = matcher
        self._track_query_false_positive_prob = track_query_false_positive_prob
        self._track_query_false_negative_prob = track_query_false_negative_prob
        self._backprop_prev_frame = backprop_prev_frame
        
        self._tracking = False
        
    def train(self, mode: bool = True):
        """Sets the module in train mode."""
        self._tracking = False
        return super().train(mode)
    
    def tracking(self):
        """Sets the module in tracking mode."""
        self.eval()
        self._tracking = True
        
    def add_track_queries_to_targets(self, targets, prev_indices, prev_out, add_false_pos=True):
        device = prev_out['pred_sub_boxes'].device
         
        min_prev_target_ind = min([len(prev_ind[1]) for prev_ind in prev_indices])
        num_prev_target_ind = 0
        if min_prev_target_ind:
            num_prev_target_ind = torch.randint(0, min_prev_target_ind + 1, (1,)).item()    
        if num_prev_target_ind < 1:
            num_prev_target_ind = 1
            
        num_prev_target_ind_for_fps = 0
        if num_prev_target_ind:
            num_prev_target_ind_for_fps = \
                torch.randint(int(math.ceil(self._track_query_false_positive_prob * num_prev_target_ind)) + 1, (1,)).item()
        #num_prev_target_ind_for_fps = 1
        for i, (target, prev_ind) in enumerate(zip(targets, prev_indices)):
            prev_out_ind, prev_target_ind = prev_ind
            
            # random subset
            if self._track_query_false_negative_prob:
                random_subset_mask = torch.randperm(len(prev_target_ind))[:num_prev_target_ind]
                prev_out_ind = prev_out_ind[random_subset_mask]
                prev_target_ind = prev_target_ind[random_subset_mask]
        
            # detected prev frame tracks
            prev_sub_track_ids = target['prev_target']['sub_track_ids'][prev_target_ind]
            prev_obj_track_ids = target['prev_target']['obj_track_ids'][prev_target_ind]
            
            # match track ids between frames
            target_sub_ind_match_matrix = prev_sub_track_ids.unsqueeze(dim=1).eq(target['sub_track_ids'])
            target_obj_ind_match_matrix = prev_obj_track_ids.unsqueeze(dim=1).eq(target['obj_track_ids'])
            target_ind_match_matrix = target_sub_ind_match_matrix * target_obj_ind_match_matrix
            target_ind_matching = target_ind_match_matrix.any(dim=1)
            target_ind_matched_idx = target_ind_match_matrix.nonzero()[:, 1]
            
            # index of prev frame detection in current frame box list
            target['track_query_match_ids'] = target_ind_matched_idx
            
            # random false positives
            if add_false_pos:
                prev_sub_boxes_matched = prev_out['pred_sub_boxes'][i, prev_out_ind[target_ind_matching]]
                prev_obj_boxes_matched = prev_out['pred_obj_boxes'][i, prev_out_ind[target_ind_matching]]
                assert prev_sub_boxes_matched.shape[1] == 4
                
                not_prev_out_ind = torch.arange(prev_out['pred_sub_boxes'].shape[1])
                not_prev_out_ind = [ ind.item() for ind in not_prev_out_ind
                                         if ind not in prev_out_ind]
        
                
                random_false_out_ind = []
                
                prev_target_ind_for_fps = torch.randperm(num_prev_target_ind)[:num_prev_target_ind_for_fps]
                
                for j in prev_target_ind_for_fps:
                    # if random.uniform(0, 1) < self._track_query_false_positive_prob:
                    prev_sub_boxes_unmatched = prev_out['pred_sub_boxes'][i, not_prev_out_ind]
                    prev_obj_boxes_unmatched = prev_out['pred_obj_boxes'][i, not_prev_out_ind]
                    
                    if len(prev_sub_boxes_matched) > j and len(prev_obj_boxes_matched) > j:
                        prev_sub_box_matched = prev_sub_boxes_matched[j]
                        prev_obj_box_matched = prev_obj_boxes_matched[j]
                        sub_box_weights = \
                            prev_sub_box_matched.unsqueeze(dim=0)[:, :2] - \
                            prev_sub_boxes_unmatched[:, :2]
                        obj_box_weights = \
                            prev_obj_box_matched.unsqueeze(dim=0)[:, :2] - \
                            prev_obj_boxes_unmatched[:, :2] 
                        box_weights = (sub_box_weights+obj_box_weights)/2    
                        box_weights = box_weights[:, 0] ** 2 + box_weights[:, 0] ** 2
                        box_weights = torch.sqrt(box_weights)
                        
                        random_false_out_idx = not_prev_out_ind.pop(
                            torch.multinomial(box_weights.cpu(), 1).item())
                    else:
                        random_false_out_idx = not_prev_out_ind.pop(torch.randperm(len(not_prev_out_ind))[0])
                    
                    random_false_out_ind.append(random_false_out_idx)
                
                prev_out_ind = torch.tensor(prev_out_ind.tolist() + random_false_out_ind).long()
                target_ind_matching = torch.cat([
                    target_ind_matching,
                    torch.tensor([False, ] * len(random_false_out_ind)).bool().to(device)
                ])
            
            # track query masks
            track_queries_mask = torch.ones_like(target_ind_matching).bool()
            track_queries_fal_pos_mask = torch.zeros_like(target_ind_matching).bool()
            track_queries_fal_pos_mask[~target_ind_matching] = True
            
            # set prev frame info
            target['track_query_hs_embeds'] = prev_out['hs_embed'][i, prev_out_ind]
            target['track_query_sub_boxes'] = prev_out['pred_sub_boxes'][i, prev_out_ind].detach()
            target['track_query_obj_boxes'] = prev_out['pred_obj_boxes'][i, prev_out_ind].detach()
            
            target['track_queries_mask'] = torch.cat([
                track_queries_mask,
                torch.tensor([False, ] * self.num_queries).to(device)
            ]).bool()

            target['track_queries_fal_pos_mask'] = torch.cat([
                track_queries_fal_pos_mask,
                torch.tensor([False, ] * self.num_queries).to(device)
            ]).bool()
                
    def forward(self, samples: NestedTensor, targets: list = None, prev_features=None):
        if targets is not None and not self._tracking:
            prev_targets = [target['prev_target'] for target in targets]
            # if self.training and random.uniform(0, 1) < 0.5:
            if self.training:
                backprop_context = torch.no_grad
                if self._backprop_prev_frame:
                    backprop_context = nullcontext
                
                with backprop_context():
                    prev_out, _, prev_features, _, _ = super().forward([t['prev_image'] for t in targets])
                    
                    prev_outputs_without_aux = {
                        k: v for k, v in prev_out.items() if 'aux_outputs' not in k}
                    prev_indices = self._matcher(prev_outputs_without_aux, prev_targets)
                    self.add_track_queries_to_targets(targets, prev_indices, prev_out)
            else:
                # if not training we do not add track queries and evaluate detection performance only.
                # tracking performance is evaluated by the actual tracking evaluation.
                for target in targets:
                    device = target['sub_boxes'].device
                    target['track_query_hs_embeds'] = torch.zeros(0, self.hidden_dim).float().to(device)
                    target['track_queries_mask'] = torch.zeros(self.num_queries).bool().to(device)
                    target['track_queries_fal_pos_mask'] = torch.zeros(self.num_queries).bool().to(device)
                    target['track_query_sub_boxes'] = torch.zeros(0, 4).to(device)
                    target['track_query_obj_boxes'] = torch.zeros(0, 4).to(device)
                    target['track_query_match_ids'] = torch.tensor([]).long().to(device)
         
            out, targets, features, memory, hs  = super().forward(samples, targets, prev_features, stage=2)

            return out, targets, features, memory, hs
        

class VRDFormerTracking(TrackingBase, VRDFormer):
    def __init__(self, tracking_kwargs, detr_kwargs):
        VRDFormer.__init__(self, **detr_kwargs)
        TrackingBase.__init__(self, **tracking_kwargs)
        
        
class SetCriterionTrack(nn.Module):
    def __init__(self, 
                 num_obj_classes, 
                 matcher, 
                 weight_dict, 
                 eos_coef, 
                 losses,
                 focal_loss, focal_alpha, focal_gamma,
                 track_query_false_positive_eos_weight,
                ):
        super().__init__()
        self.num_obj_classes = num_obj_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.tracking = True
        self.track_query_false_positive_eos_weight = track_query_false_positive_eos_weight
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
    def loss_labels(self, outputs, targets, indices, _, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_sub_logits" in outputs and "pred_obj_logits" in outputs
        
        losses = {}
        so_losses = []
        for role in ["sub", "obj"]:
            src_logits = outputs["pred_%s_logits"%role]
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["%s_labels"%role][J] for t, (_, J) in zip(targets, indices)])
            target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o
            
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2),
                                    target_classes,
                                    weight=self.empty_weight,
                                    reduction='none')
            if self.tracking and self.track_query_false_positive_eos_weight:
                for i, target in enumerate(targets):
                    if 'track_query_boxes' in target:
                        # remove no-object weighting for false track_queries
                        loss_ce[i, target['track_queries_fal_pos_mask']] *= 1 / self.eos_coef
                        # assign false track_queries to some object class for the final weighting
                        target_classes = target_classes.clone()
                        target_classes[i, target['track_queries_fal_pos_mask']] = 0
            loss_ce = loss_ce.sum() / self.empty_weight[target_classes].sum()
            
            if log:
                losses["%s_class_acc"%role] = accuracy(src_logits, target_classes_o)[0]
         
            so_losses.append(loss_ce)
            
        losses["loss_ce"] = sum(so_losses) / 2.

        import pdb;pdb.set_trace()  
    
        
        
        return losses
    
    def loss_labels_focal(self, outputs, targets, indices, num_trajs, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        
        assert "pred_sub_logits" in outputs and "pred_obj_logits" in outputs
        for role in ["sub", "obj"]:
            src_logits = outputs['pred_%s_logits'%role]
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["%s_labels"%role][J] for t, (_, J) in zip(targets, indices)])
            target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o
            
            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            target_classes_onehot = target_classes_onehot[:,:,:-1]
            import pdb;pdb.set_trace()
            loss_ce = sigmoid_focal_loss(
                src_logits, target_classes_onehot, num_trajs,
                alpha=self.focal_alpha, gamma=self.focal_gamma)
        
        loss_ce *= src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        
        return losses
      
    def loss_verb_labels(self, outputs, log=True):
        assert "pred_verb_logits" in outputs
        src_logits, verb_labels = outputs["pred_verb_logits"], outputs["label_verb_classes"]
        loss_verb_ce = F.binary_cross_entropy_with_logits(src_logits, verb_labels)
        losses = {"loss_verb_ce": loss_verb_ce}
        if log:
            losses["verb_class_acc"] = multi_label_acc(src_logits, verb_labels)
        return losses
    
    def get_loss(self, loss, outputs, targets, indices, num_trajs, **kwargs):
        loss_map = {
                    "labels": self.loss_labels_focal if self.focal_loss else self.loss_labels,
                    "verb_labels": self.loss_verb_labels,
                    #"sub_boxes": self.loss_sub_boxes,
                    #"obj_boxes": self.loss_obj_boxes,
                    #"cardinality": self.loss_cardinality
                    }
        assert loss in loss_map, f"do you really wnat to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_trajs, **kwargs)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # Retrieve the matching between the outputs of the last layer and the targets
        
        indices = self.matcher(outputs_without_aux, targets, stage=2)
        
        
        # Compute the average number of target boxes accross all nodes, 
        # for normalization purposes
        num_trajs = sum(len(t["sub_labels"]) for t in targets)
        num_trajs = torch.as_tensor(
            [num_trajs], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_trajs)
        num_trajs = torch.clamp(num_trajs / get_world_size(), min=1).item()    
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_trajs))
        
        # In case of auxiliary losses, we repeat this process with the
        # output of each intermediate layer.
        import pdb;pdb.set_trace()
        
        return losses
