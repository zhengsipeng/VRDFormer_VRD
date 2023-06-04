import glob
import os
import json
import random
import pickle as pkl
import numpy as np
import torch
import torch.utils.data
from pathlib import Path
from . import video_transforms as T
from util.box_ops import box_cxcywh_to_xyxy
from datasets.dataset import VRDBase, make_video_transforms


class VidOR(VRDBase):
    def __init__(self, 
            dbname,
            image_set,
            data_dir, 
            max_duration,
            anno_file, 
            transforms, 
            trainval_imgset_file, 
            seq_len, 
            num_quries,
            stage=1,
            prev_frame=False, prev_frame_range=1, 
            prev_frame_rnd_augs=0.0, prev_prev_frame=False, debug=False
        ):
        num_verb_classes = 50
        super().__init__(dbname, image_set, data_dir, max_duration, anno_file, transforms, 
                         trainval_imgset_file, seq_len, num_quries, num_verb_classes,
                         stage, prev_frame, prev_frame_range, prev_frame_rnd_augs, prev_prev_frame, debug)

    def _check_anno(self, anno):
        assert 'version' in anno and anno['version']=='VERSION 1.0'
        return anno

    def get_action_predicates(self):
        print('[warning] VidOR._get_action_predicates() is deprecated.')
        actions = [
            'watch','bite','kiss','lick','smell','caress','knock','pat',
            'point_to','squeeze','hold','press','touch','hit','kick',
            'lift','throw','wave','carry','grab','release','pull',
            'push','hug','lean_on','ride','chase','get_on','get_off',
            'hold_hand_of','shake_hand_with','wave_hand_to','speak_to','shout_at','feed',
            'open','close','use','cut','clean','drive','play(instrument)',
        ]
        for action in actions:
            assert action in self.pred2pid
        return actions

    def get_spatial_predicates(self):
        return {'above', 'away', 'behind', 'beneath', 'in_front_of', 'next_to', 'toward', 'inside'}
    
    def get_interactive_predicates(self):
        return {
            'bite','kiss','lick','smell','caress','knock','pat',
            'squeeze','hold','press','touch','hit','kick',
            'lift','throw','wave','carry','grab','release','pull',
            'push','hug','lean_on','ride','chase','get_on','get_off',
            'hold_hand_of','shake_hand_with','feed',
            'open','close','use','cut','clean','drive','play(instrument)',
        }

    def get_verb_predicates(self):
        return {
            'watch','bite','kiss','lick','smell','caress','knock','pat',
            'point_to','squeeze','hold','press','touch','hit','kick',
            'lift','throw','wave','carry','grab','release','pull',
            'push','hug','lean_on','ride','chase','get_on','get_off',
            'hold_hand_of','shake_hand_with','wave_hand_to','speak_to','shout_at','feed',
            'open','close','use','cut','clean','drive','play(instrument)',
        }
    
    def get_human_classes(self):
        return {'adult', 'child', 'baby'}
    
    def get_animal_classes(self):
        return {'dog', 'cat', 'bird', 'duck', 'horse', 'elephant', 'fish', 'penguin', 'chicken', 
            'hamster/rat', 'sheep/goat', 'pig', 'cattle/cow', 'rabbit', 'turtle', 'tiger', 
            'panda', 'lion', 'kangaroo', 'camel', 'bear', 'crab', 'snake', 'squirrel',
            'leopard', 'stingray', 'crocodile'}
    
    
    

def build_dataset(image_set, args):
    
    root = Path(args.vidor_path)
    assert root.exists(), f'provided VidOR path {root} does not exist'

    dbname = args.dataset
    data_dir = args.vidor_path
    max_duration = args.max_duration
    anno_file = "data/metadata/%s_annotations.pkl"%dbname

    if image_set == 'train':
        trainval_imgset_file = "data/metadata/%s_%s_frames_stage%d.json"%(dbname, image_set, args.stage)
        prev_frame_rnd_augs = args.track_prev_frame_rnd_augs
        prev_frame_range=args.track_prev_frame_range
    else:
        trainval_imgset_file = "data/metadata/%s_%s_frames.json"%(dbname, image_set)
        prev_frame_rnd_augs = 0.0
        prev_frame_range = 1
        
    transforms = make_video_transforms(image_set, 
                                       args.cautious, 
                                       args.by_ratio, 
                                       args.resolution,
                                       overflow_boxes=args.overflow_boxes)

    dataset = VidOR(
        dbname,
        image_set,
        data_dir,
        max_duration,
        anno_file,
        transforms=transforms,
        trainval_imgset_file=trainval_imgset_file, 
        seq_len=args.seq_len,
        num_quries=args.num_queries,
        stage=args.stage,
        prev_frame=args.tracking,
        prev_frame_range=prev_frame_range,
        prev_frame_rnd_augs=prev_frame_rnd_augs, 
        prev_prev_frame=args.track_prev_prev_frame,
        debug=args.debug
    )

        
    return dataset