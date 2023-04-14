import numpy as np
from .compute import viou, voc_ap
from collections import defaultdict, OrderedDict
    
    
def eval_tagging_scores(gt_relations, pred_relations, min_pred_num=0):
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    # ignore trajectories
    gt_triplets = set(tuple(r['triplet']) for r in gt_relations)
    pred_triplets = []
    hit_scores = []
    for r in pred_relations:
        triplet = tuple(r['triplet'])
        if not triplet in pred_triplets:
            pred_triplets.append(triplet)
            hit_scores.append(r['score'])
            
    hit_scores.extend([-np.inf]*(min_pred_num-len(hit_scores)))
    hit_scores = np.asarray(hit_scores)
    for i, t in enumerate(pred_triplets):
        if not t in gt_triplets:
            hit_scores[i] = -np.inf
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_triplets), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores


def eval_detection_scores(gt_relations, pred_relations, viou_threshold, allow_misalign=False):
    """
    allow_misalign: allow temporal misalignment between subject and object in the prediction;
                    this require 'duration' being replaced by 'sub_duration' and 'obj_duration'
    """
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    gt_detected = np.zeros((len(gt_relations),), dtype=bool)
    hit_scores = np.ones((max(len(pred_relations), len(gt_relations)),)) * -np.inf
    for pred_idx, pred_relation in enumerate(pred_relations):
        ov_max = -float('Inf')
        k_max = -1
        for gt_idx, gt_relation in enumerate(gt_relations):
            if not gt_detected[gt_idx] and tuple(pred_relation['triplet']) == tuple(gt_relation['triplet']):
                if allow_misalign and 'sub_duration' in pred_relation:
                    sub_duration = pred_relation['sub_duration']
                    obj_duration = pred_relation['obj_duration']
                else:
                    sub_duration = pred_relation['duration']
                    obj_duration = pred_relation['duration']
                    
                s_iou = viou(pred_relation['sub_traj'], sub_duration,
                        gt_relation['sub_traj'], gt_relation['duration'])
                o_iou = viou(pred_relation['obj_traj'], obj_duration,
                        gt_relation['obj_traj'], gt_relation['duration'])
                ov = min(s_iou, o_iou)

                if ov >= viou_threshold and ov > ov_max:
                    ov_max = ov
                    k_max = gt_idx
        if k_max >= 0:
            hit_scores[pred_idx] = pred_relation['score']
            gt_detected[k_max] = True
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_relations), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores


def evaluate_visual_relation(groundtruth, prediction, viou_threshold=0.5,
                             det_nreturns=[50, 100], tag_nreturns=[1, 5, 10], 
                             allow_misalign=False, verbose=True):
    """ evaluate visual relation detection and visual 
    relation tagging.
    """
    if allow_misalign:
        print('[warning] subject and object misalignment allowed (non-official support)')
    video_ap = dict()
    tot_scores = defaultdict(list)
    tot_tp = defaultdict(list)
    prec_at_n = defaultdict(list)
    tot_gt_relations = 0

    if verbose:
        print('[info] computing metric scores over {} videos...'.format(len(groundtruth)))

    for vid, gt_relations in groundtruth.items():
        if len(gt_relations)==0:
            continue
        if vid not in prediction or len(prediction[vid])==0:
            continue
        tot_gt_relations += len(gt_relations)
        predict_relations = prediction.get(vid, [])
   
        if 'sub_traj' in predict_relations[0]:
            # compute average precision and recalls in detection setting
            det_prec, det_rec, det_scores = eval_detection_scores(
                    gt_relations, predict_relations, viou_threshold, allow_misalign=allow_misalign)
            video_ap[vid] = voc_ap(det_rec, det_prec)
            tp = np.isfinite(det_scores)
            for nre in det_nreturns:
                cut_off = min(nre, det_scores.size)
                tot_scores[nre].append(det_scores[:cut_off])
                tot_tp[nre].append(tp[:cut_off])
        
        # compute precisions in tagging setting
        tag_prec, _, _ = eval_tagging_scores(gt_relations, predict_relations, max(tag_nreturns))
        for nre in tag_nreturns:
            prec_at_n[nre].append(tag_prec[nre-1])
        
    output = OrderedDict()
    # calculate mean ap for detection
    if len(video_ap)>0:
        output['detection mean AP'] = np.mean(list(video_ap.values()))
        # calculate recall for detection
        for nre in det_nreturns:
            scores = np.concatenate(tot_scores[nre])
            tps = np.concatenate(tot_tp[nre])
            sort_indices = np.argsort(scores)[::-1]
            tps = tps[sort_indices]
            cum_tp = np.cumsum(tps).astype(np.float32)
            rec = cum_tp / np.maximum(tot_gt_relations, np.finfo(np.float32).eps)
            output['detection recall@{}'.format(nre)] = rec[-1]
            
    # calculate mean precision for tagging
    for nre in tag_nreturns:
        output['tagging precision@{}'.format(nre)] = np.mean(prec_at_n[nre])
    
    return output


def evaluate(groundtruth, prediction, dataset):
    scores = dict()
    print('[info] evaluating overall setting')
    scores['overall'] = evaluate_visual_relation(groundtruth, prediction)

    for use_origin_zeroshot_eval in [False, True]:
        if use_origin_zeroshot_eval:
            print('[info] evaluating generalized zero-shot setting')
        else:
            print('[info] evaluating zero-shot setting')
            
        groundtruth = dict()
        zs_prediction = dict()
        
        for vid in dataset.video_ids:
            if vid not in prediction:
                continue
            gt_relations = dataset.get_relation_insts(vid)
            zs_gt_relations = []
            for r in gt_relations:
                if tuple(r['triplet']) in dataset.zeroshot_triplets:
                    zs_gt_relations.append(r)
            if len(zs_gt_relations) > 0:
                groundtruth[vid] = zs_gt_relations
                if use_origin_zeroshot_eval:
                    # origin zero-shot evaluation doesn't filter out non-zeroshot predictions
                    # in a video, which is the generalized zero-shot setting 
                    zs_prediction[vid] = prediction[vid]
                else:
                    zs_prediction[vid] = []
                    for r in prediction.get(vid, []):
                        if tuple(r['triplet']) in dataset.zeroshot_triplets:
                            zs_prediction[vid].append(r)
       
        if use_origin_zeroshot_eval:
            scores['generalized zero-shot'] = evaluate_visual_relation(groundtruth, zs_prediction)
        else:
            scores['zero-shot'] = evaluate_visual_relation(groundtruth, zs_prediction)

    return scores


def print_scores(scores, score_variance=None):
    for setting in scores.keys():
        print('[setting] {}'.format(setting))
        for metric in scores[setting].keys():
            if score_variance is not None:
                print('\t{}:\t{:.4f} \u00B1 {:.4f}'.format(metric, scores[setting][metric], score_variance[setting][metric]))
            else:
                print('\t{}:\t{:.4f}'.format(metric, scores[setting][metric]))