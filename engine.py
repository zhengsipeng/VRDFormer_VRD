import torch
import math
import datetime
import util.misc as utils
from util.misc import NestedTensor, target_to_cuda
from util.box_ops import debug_and_vis
from util.evaluate import evaluate

def is_loss_nan(loss):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    if math.isnan(loss):
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))


def train_tagging(model, criterion, data_loader, optimizer, device, epoch, args):
    
    #vis_iter_metrics = None
    #if visualizers:
    #    vis_iter_metrics = visualizers['iter_metrics']
        
    max_norm = args.clip_max_norm
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(
        args.vis_and_log_interval,
        delimiter="  ",
        vis=None,
        debug=args.debug)

    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('sub_class_acc', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))
    metric_logger.add_meter('obj_class_acc', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))
    metric_logger.add_meter('verb_class_acc', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))
    
    optimizer.zero_grad()
    
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, epoch, batch_size=args.batch_size*args.accumulate_steps)):
        if args.debug:
            debug_and_vis(args.datasets, samples, targets, i) 
        
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)  
        
        samples = samples.to(device)
        targets = [target_to_cuda(t) for t in targets[0]]
        
        # given annos of boxes, predicate class of {s,p,o} 
        # we don't use the rec-query and initialize static-query by given boxes
        
        memory = None
        for fid in range(args.seq_len): 
            cur_frame = samples.select_frame(fid)  # 1,3,H,W, 
            
            memory = model(cur_frame,
                           targets[fid], 
                           memory, 
                           eos=(fid+1)==args.seq_len
                        )  
       
        loss_dict = criterion(memory) 
        
        weight_dict = criterion.weight_dict
   
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        is_loss_nan(losses)
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        
        losses = losses / args.accumulate_steps
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        if (i+1) % args.accumulate_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(sub_class_acc=loss_dict_reduced['sub_class_acc'])
        metric_logger.update(obj_class_acc=loss_dict_reduced['obj_class_acc'])
        metric_logger.update(verb_class_acc=loss_dict_reduced['verb_class_acc'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"], 
                             lr_backbone=optimizer.param_groups[1]["lr"])

    if (i+1) % args.accumulate_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_detection(model, criterion, data_loader, optimizer, device, epoch, args):
    #vis_iter_metrics = None
    #if visualizers:
    #    vis_iter_metrics = visualizers['iter_metrics']
    
    max_norm = args.clip_max_norm
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(
        args.vis_and_log_interval,
        delimiter="  ",
        vis=None,
        debug=args.debug)

    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('sub_class_acc', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))
    metric_logger.add_meter('obj_class_acc', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))
    metric_logger.add_meter('verb_class_acc', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))
    
    optimizer.zero_grad()
    
    optimizer.zero_grad()
    
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, epoch, batch_size=args.batch_size*args.accumulate_steps)):
        if args.debug:
            debug_and_vis(args.datasets, samples, targets, i) 
        
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)  
        targets = [target_to_cuda(t) for t in targets[0]]
        
        # samples [2,3,H,W]
        # targets: [xxx, xxx]
        # 'boxes', 'labels', 'image_id', 'track_ids', 'area', 'iscrowd', 
        # 'orig_size', 'size', 'labels_ignore', 'area_ignore', 
        # 'iscrowd_ignore', 'boxes_ignore', 'track_ids_ignore', 
        # 'prev_image', 'prev_target'
        
        # in order to be able to modify targets inside the forward call we need
        # to pass it through as torch.nn.parallel.DistributedDataParallel only
        # passes copies
        outputs, targets, *_ = model(samples, targets)
        
        '''
        memory = None
        for fid in range(args.seq_len): 
            cur_frame = samples.select_frame(fid)  # 1,3,H,W, 
            
            memory = model(cur_frame,
                           targets[fid], 
                           memory, 
                           eos=(fid+1)==args.seq_len
                        )  
        '''
        import pdb;pdb.set_trace()
    
    

def evalGT_one_epoch(model, val_loader, device, epoch, args):
    model.eval()
    metric_logger = utils.MetricLogger(
        args.vis_and_log_interval,
        delimiter="  ",
        vis=None,
        debug=args.debug)
    
    with open("data/%s/action.txt"%args.dataset, "r") as f:
        action_list = [l.strip() for l in f.readlines()]
    action_dict = dict(zip(range(len(action_list)), action_list))
    val_dataset = val_loader.dataset
    groundtruth = dict()
    prediction = dict()
    for i, (samples, targets) in enumerate(metric_logger.log_every(val_loader, epoch)):
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)  
        
        samples = samples.to(device)
        targets = [target_to_cuda(t) for t in targets[0]]
        video_id = targets[0]['video_id']
        
        groundtruth[video_id] = targets[0]['groundtruth']
        frame_ids = [int(target['frame_id']) for target in targets]
        
        memory = None
        for fid in range(len(frame_ids)): 
            cur_frame = samples.select_frame(fid)  # 1,3,H,W, 
            memory = model(cur_frame,
                           targets[fid], 
                           memory, 
                           eos=(fid+1)==len(frame_ids),
                           is_eval=True
                        )  
        preds, scores = model.module.relation_classifier(memory, gt=targets[0]['groundtruth'], mode="evalGT")
        prediction[video_id] = []
        for j, pred in enumerate(preds):
            prediction[video_id].append({'triplet': (groundtruth[video_id][j]['triplet'][0], action_dict[pred], groundtruth[video_id][j]['triplet'][2]),
                                         'score': scores[j]
                                        })
        scores = evaluate(groundtruth, prediction, val_dataset)
        print("[info] Video %d"%i)
        print(scores)
    return scores
