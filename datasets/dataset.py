import os
import numpy as np
import torch
import json
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import decord


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

    def __call__(self, anno, w, h):
        sboxes = torch.as_tensor(anno["sboxes"], dtype=torch.float32).reshape(-1, 4)
        oboxes = torch.as_tensor(anno["oboxes"], dtype=torch.float32).reshape(-1, 4) 
        
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

        sclss = torch.as_tensor(anno["sclss"])[keep]
        oclss = torch.as_tensor(anno["oclss"])[keep]

        so_traj_ids = torch.as_tensor(anno["so_traj_ids"])[keep]
        
        raw_vclss = anno["vclss"]
        vclss = [self.get_one_hot(raw_vclss[i]) for i, flag in enumerate(keep) if flag]
        vclss = torch.stack(vclss)
        target = {"sboxes": sboxes, "oboxes": oboxes, "sarea": sarea, "oarea": oarea,
                  "sclss": sclss, "oclss": oclss, "so_traj_ids": so_traj_ids, 
                  "vclss": vclss, "raw_vclss": raw_vclss}
        
        for k in target.keys():
            target[k] = target[k][:self.num_queries]
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["num_svo"] = len(anno["vclss"])

        return target


class VRDBase(Dataset):
    def __init__(self, 
            dbname, 
            image_set,
            data_dir, 
            anno_file,
            num_quries, 
            num_verb_class
        ):
        super().__init__()
        self.dbname = dbname
        self.image_set = image_set
        self.data_dir = data_dir
        self.anno_file = anno_file
        self.prepare = ConvertCocoPolysToMask(num_quries, num_verb_class)
        self.normalize_coords = True
        self.load_raw_annotations(image_set)
        
    def parse_frame_name(self, frame_name):
        raise NotImplementedError

    def fid2int(self, frame_ids):
        raise NotImplementedError

    def read_video_clip(self, video_id, begin_fid, end_fid):
        video_path = os.path.join(self.data_dir, "videos", video_id+'.mp4')
        vr = VideoReader(video_path, ctx=cpu(0))
        frame_idx = np.linspace(begin_fid, end_fid-1, num=end_fid-begin_fid).astype(int)
        img_arrays = vr.get_batch(frame_idx).asnumpy()
        #img_arrays = img_arrays.permute(0, 3, 1, 2)
        return img_arrays

    def is_cls_mismatch(self, video_id, begin_fid, end_fid):
        begin_anno = self.annotations[video_id]["frame_annos"][begin_fid]
        end_anno = self.annotations[video_id]['frame_annos'][end_fid]
        begin_so_ids, end_so_ids = begin_anno['so_traj_ids'], end_anno['so_traj_ids']
        assert len(begin_so_ids)==len(end_so_ids)
        for i, begin_so_id in enumerate(begin_so_ids):
            import pdb;pdb.set_trace()

    def get_video_gt(self, video_id, seq_fids, w, h):
        targets = []
        for seq_fid in seq_fids:
            if self.dbname == "vidvrd":
                anno = self.annotations[video_id]["frame_annos"][seq_fid]
            else:
                anno = self.annotations[seq_fid]
            target = self.prepare(anno, w, h)
            target['video_id'] = video_id
            target["frame_id"] = int(seq_fid)
            target["svo_ids"] = torch.arange(len(target['vclss']))
            targets.append(target)
            
        return targets

    def get_triplets(self):
        triplets = set()
        
        for vid in self.video_ids:
            insts = self.get_relation_insts(vid, no_traj=True)
            triplets.update(inst['triplet'] for inst in insts)
        return triplets
    
    def __getitem__(self, index):
        return index

    # ==========
    # Test
    # ==========
    def _check_anno(self, anno):
        assert 'version' not in anno
        return anno

    def load_raw_annotations(self, split):
        print('[info] loading raw annotations...')
        so = set()
        pred = set()
        self.split_index = defaultdict(list)
        self.raw_annos = dict()
        
        anno_files = self._get_anno_files(split)
         
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
    
    def _get_anno_files(self, split):
        raise NotImplementedError