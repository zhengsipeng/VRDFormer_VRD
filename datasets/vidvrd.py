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
from datasets.dataset import VRDBase


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
            num_verb_classes,
            stage=1,
            prev_frame=False, prev_frame_range=1, prev_frame_rnd_augs=0.0, prev_prev_frame=False
        ):
        super().__init__(dbname, image_set, data_dir, anno_file, num_quries, num_verb_classes)
        self.dbname = dbname
        self.image_set = image_set
        self.data_dir = data_dir
        self.max_duration = max_duration
        self.anno_file = anno_file
        self.stage = stage
        self._prev_frame = prev_frame
        self._prev_frame_range = prev_frame_range
        self._prev_frame_rnd_augs = prev_frame_rnd_augs
        self._prev_prev_frame = prev_prev_frame
        
        with open(trainval_imgset_file, "r") as f:
            data = json.load(f)
        
        print('[info] loading processed annotations...')
        with open(self.anno_file, "rb") as f:
            self.annotations = pkl.load(f)
        split = "train" if self.image_set=="train" else "val"
        self.video_ids = [vid.split(".")[0] for vid in os.listdir(self.data_dir+"/annotations/%s"%split)]
        
        if self.image_set == "train":
            self.train_begin_fids, self.max_durations = data["train_begin_fids"], data["durations"]
        else:
            if self.normalize_coords:
                print('[info] bounding boxes are normalized to [0, 1]')
            #import pdb;pdb.set_trace()
            self.val_frame_meta = data
            self.idx2vid = dict(zip(range(len(self.val_frame_meta)), self.val_frame_meta.keys()))    
        
        self.clip_duration = [7,15,31] #clip_duration  # [7,15,31]
        self.seq_len = seq_len  # default: 8
        self.transforms = transforms

    def __len__(self):
        return len(self.train_begin_fids) if self.image_set=="train" else len(self.val_frame_meta) # video num

    def _get_anno_files(self, split):
        anno_files = glob.glob(os.path.join(self.data_dir, 'annotations/%s/*.json'%split))
        assert len(anno_files)>0, 'No annotation file found. Please check if the directory is correct.'
        return anno_files
    
    def parse_frame_name(self, frame_name):
        video_id, begin_fid = frame_name.split("-")[-2:]
        return "", video_id, begin_fid

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
        if self.transforms is not None:
            imgs, targets = self.transforms(imgs, targets)

            for i, gt in enumerate(targets):
                h, w = gt["size"]
                img_resize = torch.as_tensor([w, h, w, h])
                targets[i]["unscaled_sub_boxes"] = box_cxcywh_to_xyxy(targets[i]["sub_boxes"]) * img_resize
                targets[i]["unscaled_obj_boxes"] = box_cxcywh_to_xyxy(targets[i]["obj_boxes"]) * img_resize
        imgs = torch.stack(imgs)

        return imgs, targets

    def prepare_data_stage1(self, index):
        begin_frame = self.train_begin_fids[index]
        subid, video_id, begin_fid = self.parse_frame_name(begin_frame)
        frame_id = int(begin_fid)
        img = self.read_video_frame(video_id, frame_id)
        target = self.get_video_gt(video_id, [frame_id], img)
        img, target = self.transforms(img, target)
        
        img = img.squeeze(1)
        assert len(img.shape)==3 and img.shape[0]==3
        target = target[0]
        
        if self._prev_frame:
            prev_frame_id = random.randint(
                frame_id, frame_id+min(self._prev_frame_range, self.max_durations[index]-1)
            )
            prev_img = self.read_video_frame(video_id, prev_frame_id)
            prev_target = self.get_video_gt(video_id, [prev_frame_id], prev_img)
            prev_img, prev_target = self.transforms(prev_img, prev_target)
            prev_img = prev_img.squeeze(1)
            assert len(prev_img.shape)==3 and prev_img.shape[0]==3
            target[f'prev_image'] = prev_img
            target[f'prev_target'] = prev_target[0]
                
        return img, target
            
    def prepare_data_stage2(self, index):
        if self.image_set=="train":
            # vidvrd: ILSVRC2015_train_00729000_0; vidor: 0000_2401075277_12
            begin_frame = self.train_begin_fids[index]
            _, video_id, begin_fid = self.parse_frame_name(begin_frame)

            begin_fid = int(begin_fid)
            end_fid = begin_fid + self.max_durations[index]  # end_fid is not included
            
            clip_imgs = self.read_video_clip(video_id, begin_fid, end_fid)
            clip_fids = list(range(begin_fid, end_fid)) 
            if len(clip_fids)<8:
                clip_fids = clip_fids+clip_fids
            random.shuffle(clip_fids)
            # we randomly select 8 frames from [begin_frameid, begin_frameid+duration]
            seq_fids = sorted(clip_fids[:self.seq_len])
            imgs = clip_imgs[ [fid-begin_fid for fid in seq_fids] ] # num_frame,H,W,3 #.permute(1,2,0,3)
        else:
            video_id = self.idx2vid[index]      
            groundtruth = self.get_relation_insts(video_id)

            val_pos_frames= np.asarray(self.val_frame_meta[video_id], dtype=int)
            frame_count = len(val_pos_frames)
            seq_fids = np.argwhere(val_pos_frames).reshape(-1)
            clip_imgs = self.read_video_clip(video_id, 0, frame_count)
            imgs = clip_imgs[seq_fids]
        
        targets = self.get_video_gt(video_id, seq_fids, imgs)  
        
        imgs, targets = self.transforms(imgs, targets)  # 3,t,h,w
        #import pdb;pdb.set_trace()
        if self.image_set!="train":
            targets[0]['video_id'] = video_id
            targets[0]['groundtruth'] = groundtruth
            
        for i, gt in enumerate(targets):
            h, w = gt["size"]
            img_resize = torch.as_tensor([w, h, w, h])
            targets[i]["unscaled_sub_boxes"] = box_cxcywh_to_xyxy(targets[i]["sub_boxes"]) * img_resize
            targets[i]["unscaled_oub_boxes"] = box_cxcywh_to_xyxy(targets[i]["obj_boxes"]) * img_resize
        
        return imgs, targets


def make_video_transforms(split, cautious=True, by_ratio=False, resolution="large", overflow_boxes=False):
    """
    :param cautious: whether to preserve bounding box annotations in the spatial random crop
    :return: composition of spatial data transforms to be applied to every frame of a video
    """
    normalize = T.Compose([
         T.ToTensor(), 
         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    horizontal = [] if cautious else [T.RandomHorizontalFlip()]
    
    if resolution=="small":
        scales = [288, 320, 352, 384, 416, 448, 480]
        max_size = 800
        resizes = [240, 300, 360]
        crop = 240
    elif resolution=="middle":
        scales = [384, 416, 448, 480, 512, 544, 576, 608, 640]
        max_size = 1000
        resizes = [300, 400, 500]
        crop = 300
    else:
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        max_size = 1333
        resizes = [400, 500, 600]
        crop = 384

    test_size = [800]
    scale = [0.8, 1.0]
    
    if split == "train":
        transforms = horizontal + [
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose(
                    [
                        T.RandomResize(resizes),
                        T.RandomSizeCrop(crop, max_size, scale, 
                                         respect_boxes=cautious, 
                                         by_ratio=by_ratio,
                                         overflow_boxes=overflow_boxes),
                        T.RandomResize(scales, max_size=max_size),
                    ]
                ),
            ),
            normalize,
        ]

    else:
        transforms = [T.RandomResize(test_size, max_size=max_size), normalize]

    return T.Compose(transforms)

    
def build_dataset(image_set, args):
    root = Path(args.vidvrd_path)
    assert root.exists(), f'provided VidVRD path {root} does not exist'

    dbname = args.dataset
    data_dir = args.vidvrd_path
    max_duration = args.max_duration
    anno_file = "data/metadata/%s_annotations.pkl"%dbname
    trainval_imgset_file = "data/metadata/%s_%s_frames_v2.json"%(dbname, image_set)

    if image_set == 'train':
        prev_frame_rnd_augs = args.track_prev_frame_rnd_augs
        prev_frame_range=args.track_prev_frame_range
    else:
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
        num_verb_classes=args.num_verb_classes,
        stage=args.stage,
        prev_frame=args.tracking,
        prev_frame_range=prev_frame_range,
        prev_frame_rnd_augs=prev_frame_rnd_augs, 
        prev_prev_frame=args.track_prev_prev_frame
    )

        
    return dataset

