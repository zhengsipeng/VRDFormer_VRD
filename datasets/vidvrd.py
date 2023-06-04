import glob
import os
import json
import random
import pickle as pkl
import numpy as np
import torch
import torch.utils.data
from pathlib import Path
from util.box_ops import box_cxcywh_to_xyxy
from datasets.dataset import VRDBase, make_video_transforms


class VidVRD(VRDBase):
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
            prev_frame=False, prev_frame_range=1, prev_frame_rnd_augs=0.0, prev_prev_frame=False, debug=False
        ):
        num_verb_classes = 132
        super().__init__(dbname, image_set, data_dir, max_duration, anno_file, transforms, 
                         trainval_imgset_file, seq_len, num_quries, num_verb_classes,
                         stage, prev_frame, prev_frame_range, prev_frame_rnd_augs, prev_prev_frame, debug)

    def _check_anno(self, anno):
        assert 'version' not in anno
        return anno

    def fid2int(self, frame_ids):
        # convert frame str ids into int its
        frame_ints = [int(fid.split("-")[-1]) for fid in frame_ids]
        return torch.as_tensor(frame_ints)

    def get_seq_targets(self, video_id, seq_fids, w, h):
        targets = []
        for seq_fid in seq_fids:
            anno = self.annotations[video_id][seq_fid]
            target = self.prepare(anno, w, h)
            target["frame_id"] = seq_fid
            target["svo_ids"] = torch.arange(len(target['vclss']))
            targets.append(target)
        return targets

    def get_frame_data(self, video_id, frame_id):
        img_path = os.path.join(self.data_dir, "images", video_id, "%06d.jpg"%(int(frame_id)+1))
        imgs = [self.read_frame(img_path)]
        w, h = imgs[0].size
        targets = self.get_video_gt(video_id, [frame_id], w, h)
        if self._transforms is not None:
            imgs, targets = self.transforms(imgs, targets)

            for i, gt in enumerate(targets):
                h, w = gt["size"]
                img_resize = torch.as_tensor([w, h, w, h])
                targets[i]["unscaled_sub_boxes"] = box_cxcywh_to_xyxy(targets[i]["sub_boxes"]) * img_resize
                targets[i]["unscaled_obj_boxes"] = box_cxcywh_to_xyxy(targets[i]["obj_boxes"]) * img_resize
        imgs = torch.stack(imgs)

        return imgs, targets

    
def build_dataset(image_set, args):
    root = Path(args.vidvrd_path)
    assert root.exists(), f'provided VidVRD path {root} does not exist'

    dbname = args.dataset
    data_dir = args.vidvrd_path
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
    
    dataset = VidVRD(
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

