import torch
from util.misc import is_main_process

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


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
        
        for k, v in checkpoint_state_dict.items():
            if k not in model_state_dict:
                print(f'Where is {k} {tuple(v.shape)}?')

        resume_state_dict = {}
        param_from_scratch = list()
        for k, v in model_state_dict.items():  
            if k not in checkpoint_state_dict:
                print(f'Load {k} {tuple(v.shape)} from scratch.')
                resume_value = v
                param_from_scratch.append(f'{k} {tuple(v.shape)}')
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
                    if args.deformable:
                        resume_state_dict[k] = v
                        print(f'Load {k} {tuple(v.shape)} from scratch.')
                        continue
                    else:
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
                    resume_value = checkpoint_value.repeat((2,) + (num_dims - 1) * (1, ))
                else:
                    raise NotImplementedError(f"No rule for {k} with shape {v.shape}.")
                
                print(f"Load {k} {tuple(v.shape)} from resume model "
                        f"{tuple(checkpoint_value.shape)}.")
            elif args.resume_shift_neuron and 'class_embed' in k:
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

        model_without_ddp.load_state_dict(resume_state_dict)
        