import os
import json
import time
import torch
import random
import datetime
import argparse
import numpy as np
import util.dist as dist
from pathlib import Path
from models import model_initializer
from datasets import dataloader_initializer
import util.misc as utils
from util.optim import optim_initializer
from util.evaluate import print_scores
from util.checkpoints import param_initializer, save_on_master


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)

    # Training hyper-parameters
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--lr_drop", default=40, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm")
    parser.add_argument("--accumulate_steps", default=1., type=float)
    parser.add_argument("--eval_skip", default=1, type=int, help='do evaluation every "eval_skip" frames')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--vis_and_log_interval", default=10, type=int)

    parser.add_argument("--schedule", default="linear_with_warmup", type=str, choices=("step", "multistep", "linear_with_warmup", "all_linear_with_warmup"))
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9998)
    parser.add_argument("--fraction_warmup_steps", default=0.01, type=float, help="Fraction of total number of steps")
    
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--roi_pool_type', default="avg", type=str)
    
    # Backbone
    parser.add_argument("--backbone", default="resnet101", type=str, help="Name of the convolutional backbone")
    parser.add_argument('--dilation', action='store_true', help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument("--position_embedding", default="sine", type=str, choices=("sine", "learned"), help="Type of positional embedding to use on top of the image features")

    # Transformer
    parser.add_argument("--enc_layers", default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument("--dec_layers", default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument("--dim_feedforward", default=2048, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument("--hidden_dim", default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument("--nheads", default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument("--num_queries", default=100, type=int, help="Number of object tokens")
    parser.add_argument("--pre_norm", action="store_true")
    
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false', help="Disables auxiliary decoding losses (loss at each layer)")

    # Run specific
    parser.add_argument("--eval", action="store_true", help="Only run evaluation")
    parser.add_argument("--eval_mode", default="evalGT")
    parser.add_argument("--output-dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--resume_shift_neuron", action="store_true")
    parser.add_argument("--pretrain", default="", help="load pretrain from checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    
    parser.add_argument("--num_workers", default=0, type=int)

    # Dataset Specific
    parser.add_argument("--dataset_config", default=None, required=True)

    # Distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument('--local_rank', default=-1, type=int) 
    
    # VidVRD
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--coco_path", type=str, default="")
    parser.add_argument("--eos_coef", default=0.1, type=float, help="Relative classification weight of the no-object class")
    parser.add_argument("--sub_loss_coef", default=0.8, type=float)
    parser.add_argument("--obj_loss_coef", default=0.8, type=float)
    parser.add_argument("--verb_loss_coef", default=5.0, type=float)
    parser.add_argument("--vidvrd_path", default="", type=str)
    parser.add_argument("--num_verb_classes", default=132, type=int)
    parser.add_argument("--num_obj_classes", default=35, type=int)
    parser.add_argument("--max_duration", default=24, type=int)
    parser.add_argument("--seq_len", default=8, type=int)
    parser.add_argument("--resolution", default="large", type=str)
    
    # Tracking
    parser.add_argument("--tracking", default=False)
    parser.add_argument("--track_prev_frame_range", default=0, type=int)
    parser.add_argument("--track_prev_frame_rnd_augs", default=0.01, type=float)
    parser.add_argument("--track_query_false_positive_prob", default=0.1, type=float)
    parser.add_argument("--track_query_false_negative_prob", default=0.4, type=float)
    parser.add_argument("--track_query_false_positive_eos_weight", default=True)
    parser.add_argument("--track_query_noise", default=0., type=float)
    parser.add_argument("--track_prev_prev_frame", default=False)
    parser.add_argument("--track_backprop_prev_frame", default=False)
    parser.add_argument("--track_attention", action="store_true")
    
    # Deformable
    parser.add_argument("--deformable", action="store_true")
    parser.add_argument("--num_feature_levels", default=1, type=int)
    parser.add_argument("--with_box_refine", default=False)
    parser.add_argument("--overflow_boxes", default=False)
    parser.add_argument("--multi_frame_attention", default=False)
    parser.add_argument("--multi_frame_encoding", default=False)
    parser.add_argument("--merge_frame_features", default=False)
    parser.add_argument("--multi_frame_attention_separate_encoder", default=False)
    parser.add_argument("--dec_n_points", default=4, type=int)
    parser.add_argument("--enc_n_points", default=4, type=int)
    parser.add_argument("--focal_loss", default=False)
    return parser


def main(args):
    # Init distributed mode
    dist.init_distributed_mode(args)
    
    # Update dataset specific configs
    if args.dataset_config is not None:
        # https://stackoverflow.com/a/16878364
        d = vars(args)
        with open(args.dataset_config, "r") as f:
            cfg = json.load(f)
        d.update(cfg)

    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    if args.debug:
        args.num_workers = 0
    if not args.deformable:
        assert args.num_feature_levels == 1
    print(args)

    if args.stage ==1:
        from engine import train_stage1 as train_one_epoch
    else:
        from engine import train_stage2 as train_one_epoch
        from engine import eval_stage2 as eval_one_epoch
    
    output_dir = Path(args.output_dir)
    if args.output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        json.dump(vars(args), open(output_dir / 'config.json', 'w'))
    
    # fix the seed for reproducibility
    seed = args.seed + dist.get_rank()  
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if int(torch.__version__.split(".")[1]) <= 8:  # for torch version<=1.8
        torch.set_deterministic(True)  
    
    model, model_without_ddp, criterion, n_parameters = model_initializer(args, device)
    
    #visualizers = build_visualizers(args, list(criterion.weight_dict.keys()))
    
    # Set up optimizers
    optimizer = optim_initializer(args, model_without_ddp)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    param_initializer(args, model_without_ddp, optimizer, lr_scheduler)
    data_loader_train, sampler_train, data_loader_val = dataloader_initializer(args)
    
    if args.eval:
        test_stats = eval_one_epoch(model, data_loader_val, device, 0, args)
        print(test_stats)
        return 
    
    print("Start training")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args)

        lr_scheduler.step()
        
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            
            for checkpoint_path in checkpoint_paths:
                save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        
        test_stats = eval_one_epoch(model, data_loader_val, device, epoch, args)
        print(test_stats)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("VRDFormer training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    main(args)