import os
import random
import pickle as pkl
import numpy as np
import torch
import json
import glob
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from util.box_ops import box_cxcywh_to_xyxy
from . import video_transforms as T


class ConvertCocoPolysToMask(object):
    # Prepare annotations for a frame
    def __init__(self, num_queries, num_verb_class):
        self.num_queries = num_queries
        self.num_verb_class = num_verb_class

    def get_one_hot(self, labels):
        index = torch.tensor(labels)  # tensor
        onehot_label = torch.eye(self.num_verb_class)[index]
        onehot_label = onehot_label.sum(0)
        return onehot_label
        
    def __call__(self, imgs, anno):
        h, w = imgs.shape[1:3]
   
        sboxes = torch.as_tensor(anno["sub_boxes"], dtype=torch.float32).reshape(-1, 4)
        oboxes = torch.as_tensor(anno["obj_boxes"], dtype=torch.float32).reshape(-1, 4) 
  
        sboxes[:, 0::2].clamp_(min=0, max=w)
        sboxes[:, 1::2].clamp_(min=0, max=h)
        oboxes[:, 0::2].clamp_(min=0, max=w)
        oboxes[:, 1::2].clamp_(min=0, max=h)
   
        keep = (sboxes[:, 3] > sboxes[:, 1]) & (sboxes[:, 2] > sboxes[:, 0]) \
                & (oboxes[:, 3] > oboxes[:, 1]) & (oboxes[:, 2] > oboxes[:, 0])
        
        sboxes = sboxes[keep]
        oboxes = oboxes[keep]
 
        _sboxes = sboxes.reshape(-1, 2, 2)
        sarea = (_sboxes[:, 1, :] - _sboxes[:, 0, :]).prod(dim=1)
        _oboxes = oboxes.reshape(-1, 2, 2)
        oarea = (_oboxes[:, 1, :] - _oboxes[:, 0, :]).prod(dim=1)
        assert sboxes.shape[0] == oboxes.shape[0]

        sclss = torch.as_tensor(anno["sub_labels"])[keep]  # not -1 in VidVRD and VidOR
        oclss = torch.as_tensor(anno["obj_labels"])[keep]
        
        so_track_ids = torch.as_tensor(anno["so_track_ids"])[keep]
        so_track_ids[:, 0]
       
        raw_vclss = anno["verb_labels"]
        vclss = [self.get_one_hot(raw_vclss[i]) for i, flag in enumerate(keep) if flag]
        vclss = torch.stack(vclss)

        target = {"sub_boxes": sboxes, "obj_boxes": oboxes, 
                  "sub_area": sarea, "obj_area": oarea,
                  "sub_labels": sclss, "obj_labels": oclss, "verb_labels": vclss, 
                  "raw_verb_labels": raw_vclss, "so_track_ids": so_track_ids, 
                  "sub_track_ids": so_track_ids[:, 0], "obj_track_ids": so_track_ids[:, 1]
                  }
        
        for k in target.keys():
            target[k] = target[k][:self.num_queries]
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["num_inst"] = len(anno["verb_labels"])

        return target


class VRDBase(Dataset):
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
            stage,
            prev_frame, prev_frame_range, prev_frame_rnd_augs, prev_prev_frame, debug=False
        ):
        super().__init__()
        self.dbname = dbname
        self.image_set = image_set
        self.data_dir = data_dir
        self.anno_file = anno_file
        self.prepare = ConvertCocoPolysToMask(num_quries, num_verb_classes)
        self.normalize_coords = True
        
        self.max_duration = max_duration
        self.stage = stage
        self._prev_frame = prev_frame
        self._prev_frame_range = prev_frame_range
        self._prev_frame_rnd_augs = prev_frame_rnd_augs
        self._prev_prev_frame = prev_prev_frame
        
        with open(trainval_imgset_file, "r") as f:
            data = json.load(f)
        
        if not debug:
            self.load_raw_annotations(image_set) 
        
        print('[info] loading processed annotations...')
        with open(self.anno_file, "rb") as f:
            self.annotations = pkl.load(f)

        self.video_ids = [vid.split(".")[0] for vid in os.listdir(self.data_dir+"/annotations/%s"%self.image_set)]
    
        if self.image_set == "train":
            self.train_begin_fids, self.max_durations = data["train_begin_fids"], data["durations"]
        else:
            if self.normalize_coords:
                print('[info] bounding boxes are normalized to [0, 1]')
    
            self.val_frame_meta = data
            self.idx2vid = dict(zip(range(len(self.val_frame_meta)), self.val_frame_meta.keys()))    
            
        self.clip_duration = [7,15,31] #clip_duration  # [7,15,31]
        self.seq_len = seq_len  # default: 8
        self.transforms = transforms
    
    def __len__(self):
        return len(self.train_begin_fids) if self.image_set=="train" else len(self.val_frame_meta) # video num
    
    def __getitem__(self, index):   
        if self.stage==1:
            return self.prepare_data_stage1(index)
        else:
            return self.prepare_data_stage2(index)
        
    def get_anno_files(self, split):
        anno_files = glob.glob(os.path.join(self.data_dir, 'annotations/%s/*.json'%split))
        assert len(anno_files)>0, 'No annotation file found. Please check if the directory is correct.'
        return anno_files
    
    def get_video_gt(self, video_id, seq_fids, img):
        targets = []
    
        img = img[: 1]

        for seq_fid in seq_fids:
            anno = self.annotations[video_id]["frame_annos"][seq_fid]

            target = self.prepare(img, anno)
            target['video_id'] = video_id
            target["frame_id"] = int(seq_fid)
            target["inst_ids"] = torch.arange(len(target['verb_labels']))
            targets.append(target)
            
        return targets
    
    def getitem_from_id(self, video_id, frame_ids):
        video_path = os.path.join(self.data_dir, "videos", video_id+".mp4")
        vr = VideoReader(video_path, ctx=cpu(0))
        img_arrays = vr.get_batch(frame_ids).asnumpy()  # t,h,w,3
        
        targets = self.get_video_gt(video_id, frame_ids, img_arrays)
        
        img, targets = self.transforms(img_arrays, targets)
        if len(targets) > 2:
            return img, targets

        target = targets[1]
        target[f'prev_target'] = targets[0]
        target[f'prev_image'] = img[:, 0]

        img = img[:, 1]
 
        assert len(img.shape)==3 and img.shape[0]==3
       
        return img, target      
    
    def prepare_data_stage1(self, index):
        begin_frame = self.train_begin_fids[index]
        _, video_id, begin_fid = self.parse_frame_name(begin_frame)
        frame_id = int(begin_fid)
        post_frame_id = random.randint(
                frame_id, frame_id+min(self._prev_frame_range, self.max_durations[index]-1)
            )
        img, target = self.getitem_from_id(video_id, [frame_id, post_frame_id])
       
        assert target[f'prev_image'].shape == img.shape

        return img, target
    
    def prepare_data_stage2(self, index):
        
        if self.image_set=="train":
            begin_frame = self.train_begin_fids[index]
            _, video_id, begin_fid = self.parse_frame_name(begin_frame)

            begin_fid = int(begin_fid)
            end_fid = begin_fid + self.max_durations[index]  # end_fid is not included
            frame_ids =  list(range(begin_fid, end_fid)) 
            frame_ids = frame_ids+frame_ids if len(frame_ids)<8 else frame_ids
            seq_fids = sorted(random.sample(frame_ids, self.seq_len))
            
            img, target = self.getitem_from_id(video_id, seq_fids) # num_frame,H,W,3 #.permute(1,2,0,3)
        else:
            video_id = self.idx2vid[index]      
            groundtruth = self.get_relation_insts(video_id)
            
            val_pos_frames= np.asarray(self.val_frame_meta[video_id], dtype=int)
            seq_fids = np.argwhere(val_pos_frames).reshape(-1)
            img, target = self.getitem_from_id(video_id, seq_fids)

            target[0]['video_id'] = video_id
            target[0]['groundtruth'] = groundtruth
        
        for i, gt in enumerate(target):
            h, w = gt["size"]
            img_resize = torch.as_tensor([w, h, w, h])
            target[i]["unscaled_sub_boxes"] = box_cxcywh_to_xyxy(target[i]["sub_boxes"]) * img_resize
            target[i]["unscaled_obj_boxes"] = box_cxcywh_to_xyxy(target[i]["obj_boxes"]) * img_resize
        
        return img, target

    def get_triplets(self):
        triplets = set()
        
        for vid in self.video_ids:
            insts = self.get_relation_insts(vid, no_traj=True)
            triplets.update(inst['triplet'] for inst in insts)
        return triplets
    
    def parse_frame_name(self, frame_name):
        video_id, begin_fid = frame_name.split("-")[-2:]
        return "", video_id, begin_fid
    
    # ==========
    # Test
    # ==========
    def load_raw_annotations(self, split):
        print('[info] loading raw annotations...')
        so = set()
        pred = set()
        self.split_index = defaultdict(list)
        self.raw_annos = dict()
        
        anno_files = self.get_anno_files(split)

        annos = dict()
        for path in tqdm(anno_files):
            with open(path, 'r') as fin:
                anno = json.load(fin)
                anno = self._check_anno(anno)
            annos[anno['video_id']] = anno
        for vid, anno in annos.items():
            self.split_index[split].append(vid)
            for obj in anno['subject/objects']:
                so.add(obj['category'])
            for rel in anno['relation_instances']:
                pred.add(rel['predicate'])
            if self.normalize_coords and 'trajectories' in anno:
                for frame in anno['trajectories']:
                    for roi in frame:
                        roi['bbox']['xmin'] /= anno['width']
                        roi['bbox']['ymin'] /= anno['height']
                        roi['bbox']['xmax'] /= anno['width']
                        roi['bbox']['ymax'] /= anno['height']
        self.raw_annos.update(annos)

        # build index for subject/object and predicate
        so = sorted(so)
        pred = sorted(pred)
        self.soid2so = dict()
        self.so2soid = dict()
        self.pid2pred = dict()
        self.pred2pid = dict()
        for i, name in enumerate(so):
            self.soid2so[i] = name
            self.so2soid[name] = i
        for i, name in enumerate(pred):
            self.pid2pred[i] = name
            self.pred2pid[name] = i
            
    def get_raw_anno(self, vid):
        """get raw annotation for a video
        """
        return self.raw_annos[vid]

    def get_object_insts(self, vid):
        """
        get the object instances (trajectories) labeled in a video
        """
        anno = self.get_raw_anno(vid)
        object_insts = []
        tid2cls = dict()
        for item in anno['subject/objects']:
            tid2cls[item['tid']] = item['category']
        traj = defaultdict(dict)
        for fid, frame in enumerate(anno['trajectories']):
            for roi in frame:
                traj[roi['tid']][str(fid)] = (roi['bbox']['xmin'],
                                            roi['bbox']['ymin'],
                                            roi['bbox']['xmax'],
                                            roi['bbox']['ymax'])
        for tid in traj:
            object_insts.append({
                'tid': tid,
                'category': tid2cls[tid],
                'trajectory': traj[tid]
            })
        return object_insts
    
    def get_action_insts(self, vid):
        """
        get the action instances labeled in a video
        """
        anno = self.get_raw_anno(vid)
        action_insts = []
        actions = self._get_action_predicates()
        for each_ins in anno['relation_instances']:
            if each_ins['predicate'] in actions:
                begin_fid = each_ins['begin_fid']
                end_fid = each_ins['end_fid']
                each_ins_trajectory = []
                for each_traj in anno['trajectories'][begin_fid:end_fid]:
                    for each_traj_obj in each_traj:
                        if each_traj_obj['tid'] == each_ins['subject_tid']:
                            each_traj_frame = (
                                each_traj_obj['bbox']['xmin'],
                                each_traj_obj['bbox']['ymin'],
                                each_traj_obj['bbox']['xmax'],
                                each_traj_obj['bbox']['ymax']
                            )
                            each_ins_trajectory.append(each_traj_frame)
                each_ins_action = {
                    "category": each_ins['predicate'],
                    "duration": (begin_fid, end_fid),
                    "trajectory": each_ins_trajectory
                }
                action_insts.append(each_ins_action)
                
        return action_insts
    
    def get_relation_insts(self, vid, no_traj=False):
        """get the visual relation instances labeled in a video,
        no_traj=True will not include trajectories, which is
        faster.
        """
        anno = self.get_raw_anno(vid)
        sub_objs = dict()
        for so in anno['subject/objects']:
            sub_objs[so['tid']] = so['category']
        if not no_traj:
            trajs = []
            for frame in anno['trajectories']:
                bboxes = dict()
                for bbox in frame:
                    bboxes[bbox['tid']] = (bbox['bbox']['xmin'],
                                        bbox['bbox']['ymin'],
                                        bbox['bbox']['xmax'],
                                        bbox['bbox']['ymax'])
                trajs.append(bboxes)
        relation_insts = []
        for anno_inst in anno['relation_instances']:
            inst = dict()
            inst['triplet'] = (sub_objs[anno_inst['subject_tid']],
                            anno_inst['predicate'],
                            sub_objs[anno_inst['object_tid']])
            inst['subject_tid'] = anno_inst['subject_tid']
            inst['object_tid'] = anno_inst['object_tid']
            inst['duration'] = (anno_inst['begin_fid'], anno_inst['end_fid'])
            if not no_traj:
                inst['sub_traj'] = [bboxes[anno_inst['subject_tid']] for bboxes in
                        trajs[inst['duration'][0]: inst['duration'][1]]]
                inst['obj_traj'] = [bboxes[anno_inst['object_tid']] for bboxes in
                        trajs[inst['duration'][0]: inst['duration'][1]]]
            relation_insts.append(inst)
            
        return relation_insts

    # Backup
    def fid2int(self, frame_ids):
        raise NotImplementedError

    def is_cls_mismatch(self, video_id, begin_fid, end_fid):
        begin_anno = self.annotations[video_id]["frame_annos"][begin_fid]
        end_anno = self.annotations[video_id]['frame_annos'][end_fid]
        begin_so_ids, end_so_ids = begin_anno['so_track_ids'], end_anno['so_track_ids']
        assert len(begin_so_ids)==len(end_so_ids)
        for i, begin_so_id in enumerate(begin_so_ids):
            import pdb;pdb.set_trace()
            
            
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
            normalize
        ]

    else:
        transforms = [T.RandomResize(test_size, max_size=max_size), normalize]

    return T.Compose(transforms)