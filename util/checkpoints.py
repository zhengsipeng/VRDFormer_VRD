import torch
from util.misc import is_main_process

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def resume_value_deformable(model_state_dict, checkpoint_state_dict, resume_shift_neuron=False):
    resume_state_dict = dict()
    for k, v in model_state_dict.items(): 
        # resume sub*** and obj*** 
        if k not in checkpoint_state_dict:
            checkpoint_k = k.replace("sub_", "").replace("obj_", "").replace("verb_", "")
            if "class_embed" in k:
                checkpoint_value = checkpoint_state_dict[checkpoint_k]
                cls_num_dict = v.shape[0] - checkpoint_value.shape[0]
                if cls_num_dict<=0:
                    resume_value = checkpoint_value[list(range(0, v.shape[0]))]
                else:
                    resume_value = torch.cat([checkpoint_value, 
                                                checkpoint_value[:cls_num_dict]], dim=0)
            elif "bbox_embed" in k:
                resume_value = checkpoint_state_dict[checkpoint_k]
            else:
                raise NotImplementedError(f"No rule for {k} with shape {v.shape}.")
                
            print(f"Load {k} {tuple(v.shape)} from resume model"
                    f"{tuple(checkpoint_value.shape)}.")

        # resume base parameters
        elif v.shape != checkpoint_state_dict[k].shape:
            checkpoint_value = checkpoint_state_dict[k]
            num_dims = len(checkpoint_value.shape)
            
            if 'norm' in k:
                resume_value = checkpoint_value.repeat(2)
            elif 'multihead_attn' in k or 'self_attn' in k:
                resume_value = checkpoint_value.repeat(num_dims * (2, ))
            elif 'reference_points' in k and checkpoint_value.shape[0] * 2 == v.shape[0]:
                import pdb;pdb.set_trace()
                resume_value = v
                resume_value[:2] = checkpoint_value.clone()
            elif 'linear1' in k or 'query_embed' in k:
                #resume_state_dict[k] = v
                #print(f'Load {k} {tuple(v.shape)} from scratch.')
                #continue
                if checkpoint_value.shape[1]*2== v.shape[1]:
                    # from hidden size 256 to 512
                    if checkpoint_value.shape[0]*2== v.shape[0]:
                        resume_value = checkpoint_value.repeat(2, 2)
                    else:
                        resume_value = checkpoint_value.repeat(1, 2)
                elif checkpoint_value.shape[0] > v.shape[0]:
                    resume_value = checkpoint_value[:v.shape[0]]
                elif checkpoint_value.shape[0] < v.shape[0]:
                    resume_value = v
            elif 'linear2' in k or 'input_proj' in k:
                import pdb;pdb.set_trace()
                resume_value = checkpoint_value.repeat((2,) + (num_dims - 1) * (1, ))
            else:
                raise NotImplementedError(f"No rule for {k} with shape {v.shape}.")
            
            print(f"Load {k} {tuple(v.shape)} from resume model "
                    f"{tuple(checkpoint_value.shape)}.")
        elif resume_shift_neuron and 'class_embed' in k:
            checkpoint_value = checkpoint_state_dict[k]
            # no-object class
            resume_value = checkpoint_value.clone()
            # no-object class
            # resume_value[:-2] = checkpoint_value[1:-1].clone()
            resume_value[:-1] = checkpoint_value[1:].clone()
            resume_value[-2] = checkpoint_value[0].clone()
            print(f"Load {k} {tuple(v.shape)} from resume model and "
                    "shift class embed neurons to start with label=0 at neuron=0.")
        else:
            resume_value = checkpoint_state_dict[k]

        resume_state_dict[k] = resume_value
    return resume_state_dict
        
        
def resume_value(model_state_dict, checkpoint_state_dict):
    resume_state_dict = dict()
    for k, v in model_state_dict.items(): 
        # resume sub*** and obj*** 
        if k not in checkpoint_state_dict:
            if "class_embed" in k:
                if "weight" in k:
                    checkpoint_k = "class_embed.weight"
                elif "bias" in k:
                    checkpoint_k = "class_embed.bias"
                checkpoint_value = checkpoint_state_dict[checkpoint_k]
                
                cls_num_dict = v.shape[0] - checkpoint_value.shape[0]
                if cls_num_dict<=0:
                    resume_value = checkpoint_value[list(range(0, v.shape[0]))]
                else:
                    resume_value = torch.cat([checkpoint_value, 
                                                checkpoint_value[:cls_num_dict]], dim=0)
            
            elif "bbox_embed" in k:
                checkpoint_k = k[4:].split(".layers")[0][:-1]+"layers"+k[4:].split(".layers")[1]
                checkpoint_value = checkpoint_state_dict[checkpoint_k]
                resume_value = checkpoint_value  
            elif 'input_proj' in k:
                if "weight" in k:
                    checkpoint_k = "input_proj.weight"
                elif "bias" in k:
                    checkpoint_k = "input_proj.bias"
                checkpoint_value = checkpoint_state_dict[checkpoint_k]  
                resume_value = checkpoint_value
            else:
                resume_state_dict[k] = v
                print(f'Load {k} {tuple(v.shape)} from scratch.')
                continue
                
            print(f"Load {k} {tuple(v.shape)} from resume model"
                    f"{tuple(checkpoint_value.shape)}.")

        # resume base parameters
        elif v.shape != checkpoint_state_dict[k].shape:
            checkpoint_value = checkpoint_state_dict[k]
            num_dims = len(checkpoint_value.shape)
            
            if 'norm' in k:
                resume_value = checkpoint_value.repeat(2)
            elif 'multihead_attn' in k or 'self_attn' in k:
                resume_value = checkpoint_value.repeat(num_dims * (2, ))
            elif 'linear1' in k or 'query_embed' in k:
                if checkpoint_value.shape[1]*2== v.shape[1]:
                    # from hidden size 256 to 512
                    if checkpoint_value.shape[0]*2== v.shape[0]:
                        resume_value = checkpoint_value.repeat(2, 2)
                    else:
                        resume_value = checkpoint_value.repeat(1, 2)
                elif checkpoint_value.shape[0] > v.shape[0]:
                    resume_value = checkpoint_value[:v.shape[0]]
                elif checkpoint_value.shape[0] < v.shape[0]:
                    resume_value = v
            else:
                raise NotImplementedError(f"No rule for {k} with shape {v.shape}.")
            
            print(f"Load {k} {tuple(v.shape)} from resume model "
                    f"{tuple(checkpoint_value.shape)}.")
        else:
            resume_value = checkpoint_state_dict[k]
   
        resume_state_dict[k] = resume_value
        
    return resume_state_dict
            
            
def param_initializer(args, model_without_ddp, optimizer, lr_scheduler):
    if args.resume:
        print("Loading checkpoint from %s"%args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    
    if args.pretrain:
        print("Loading pretrain weights from %s"%args.pretrain)
        if args.pretrain.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.pretrain, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.pretrain, map_location='cpu')

        model_state_dict = model_without_ddp.state_dict()
        checkpoint_state_dict = checkpoint['model']
        checkpoint_state_dict = {k.replace('detr.', ''): v for k, v in checkpoint['model'].items()}
        
        if args.deformable:
            resume_state_dict = resume_value_deformable(model_state_dict, 
                                                        checkpoint_state_dict, 
                                                        args.resume_shift_neuron)
        else:
            resume_state_dict = resume_value(model_state_dict, checkpoint_state_dict)
        
        
        model_without_ddp.load_state_dict(resume_state_dict)
        import pdb;pdb.set_trace()
        