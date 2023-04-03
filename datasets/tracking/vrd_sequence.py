# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
VRD sequence dataset.
"""

import json
import os.path as osp
from typing import List

import torch

from ..vrd_dataset import make_vrd_transforms
from ..vrd_dataset import VRDBase
from ..transforms import Compose
from ...utils.box_ops import box_cxcywh_to_xyxy


class VRDSequence(VRDBase):
    """Multiple Object Tracking for VRD Dataset including VIDVRD and VidOR.
    This dataloader is designed so that it can handle only one sequence,
    if more have to be handled one should inherit from this class.
    """

    def __init__(self, dbname, video_name, dets, ann_file, root_dir, num_queries, num_verb_classes,
                    img_transform=None, task="tagging"):
        """
        Args:
            video_name (str): "ILSVRC****" of vidvrd and "0000/***" of vidor
            dets: PRIVATE/PUBLIC/ANNO
            vis_threshold (float): Threshold of visibility of persons
                                   above which they are selected
        """
        super().__init__(dbname, ann_file, num_queries, num_verb_classes)
        
        self.split = "validation" if self.dbname == "vidor" else "testing"
        self.dets = dets
        self.data_dir = root_dir
        self.video_name = video_name
        self.transforms = Compose(make_vrd_transforms('val', img_transform))

        gts, imgpaths, end_rinst_perframe, trackpair_to_rinst = self.read_gts(video_name, task)
        self.gts = gts
        self.imgpaths = imgpaths
        self.end_rinst_perframe = end_rinst_perframe
        self.trackpair_to_rinst = trackpair_to_rinst

    def __len__(self) -> int:
        """sequence length"""
        return len(self.imgpaths)

    def read_gts(self, video_name, task="tagging"):
        with open(osp.join(self.data_dir, self.dbname, "annotation", self.split, video_name), "r") as f:
            raw_gt = json.load(f)
        num_frames = raw_gt["frame_count"]
        width = raw_gt["width"]
        height = raw_gt["height"]

        video_id = video_name.split(".")[0]
        frame_ids = list(range(1, num_frames+1))
        img_paths = [osp.join(self.data_dir, self.dbname, "images", video_name, "%06d.jpg"%fid) for fid in frame_ids]
        
        if task != "tagging":
            return 0, img_paths, 0, 0

        gts = self.get_video_gts(video_id, frame_ids, width, height)
        
        end_rinst_perframe = dict()
        sotid_to_rinst = dict()
        for rid, rinst in enumerate(raw_gt["relation_instances"]):
            bfid, efid = int(rinst["begin_fid"]), int(rinst["end_fid"])-1
            stid, otid = rinst["subject_tid"], rinst["object_tid"]

            if bfid not in end_rinst_perframe.keys():
                end_rinst_perframe[bfid] = [rid]
            else:
                end_rinst_perframe[bfid].append(rid)
            if efid not in end_rinst_perframe.keys():
                end_rinst_perframe[efid] = [efid]
            else:
                end_rinst_perframe[efid].append(rid)

            if (stid, otid) not in sotid_to_rinst.keys():
                sotid_to_rinst[(stid, otid)] = [rid]
            else:
                sotid_to_rinst[(stid, otid)].append(rid)

        # generate mapping from trackpair(id) to relation instance(id) at each frame
        trackpair_to_rinst = dict()
        for fid, so_traj_ids in enumerate(self.annotations["so_traj_ids"]):
            trackpair_ids = []
            for so_traj_id in so_traj_ids:
                rid = sotid_to_rinst[tuple(so_traj_id)]
                trackpair_ids.append(rid)

            trackpair_to_rinst[fid] = trackpair_ids

        return gts, img_paths, end_rinst_perframe, trackpair_to_rinst

    def __getitem__(self, idx: int) -> dict:
        """Return the ith image converted to blob
        For visual relation tagging, the sequence will also provide the bounding boxes as well as triplet IDs
        For visual relation detection, the sequence will only provide the frames
        trackpair: tracklet pair for shot; rinst: relation instance for short
        """
        img = self.read_single_image(self.imgpaths[idx])
        width_orig, height_orig = img.size

        if self.task != "tagging":
            if self.transforms is not None:
                assert 1==0
            sample = {"img": img, }

        gt = self.gts[idx]
        
        if self.transforms is not None:
            img, gt = self.transforms(img, gt)
            h, w = gt["size"]
            img_resize = torch.as_tensor([w, h, w, h])
            gt["unscaled_sboxes"] = box_cxcywh_to_xyxy(gt["sboxes"]) * img_resize
            gt["unscaled_oboxes"] = box_cxcywh_to_xyxy(gt["oboxes"]) * img_resize

        width, height = img.size(2)

        sample = {}
        sample['img'] = img
        sample['orig_size'] = torch.as_tensor([int(height_orig), int(width_orig)])
        sample['size'] = torch.as_tensor([int(height), int(width)])

        if self.task == "tagging":
            sample['gt'] = gt
            sample['end_rinst_perframe'] = self.end_rinst_perframe[idx]  # ids of end rinst at each frame
            sample['trackpair_to_rinst'] = self.trackpair_to_rinst[idx]  # mapping of each trackpair to rinst

        return sample