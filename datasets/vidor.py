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
            num_verb_class
        ):
        super().__init__(dbname, image_set, data_dir, anno_file, num_quries, num_verb_class)
        self.dbname = dbname
        self.image_set = image_set
        self.data_dir = data_dir
        self.max_duration = max_duration
        self.anno_file = anno_file
        
        self.anno_files = dict()
            for af in glob.glob(data_dir+"/vidor/annotations/*"):
                video_id = af.split("/")[-1].split(".")[0]
                self.anno_files[video_id] = af
                
        if self.image_set == "train":
            with open(self.anno_file, "rb") as f:
                self.annotations = pkl.load(f)
        else:

        with open(trainval_imgset_file, "r") as f:
            data = json.load(f)
            
        if self.image_set == "train":
            self.train_begin_fids, self.max_durations = data["train_begin_fids"], data["durations"]
        else:
            self.val_valid_frames = data
            self.idx2vid = dict(zip(range(len(self.val_valid_frames)), self.val_valid_frames.keys()))    
        
        self.clip_duration = [7,15,31] #clip_duration  # [7,15,31]
        self.seq_len = seq_len  # default: 8
        self.transforms = transforms

    def __len__(self):
        return len(self.train_begin_fids) if self.image_set=="train" else len(self.val_valid_frames) # video num

    def parse_frame_name(self, frame_name):
        subid, video_id, begin_fid = frame_name.split("-")[-3:]
        return subid, video_id, begin_fid

    def fid2int(self, frame_ids):
        # convert frame str ids into int its
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
                targets[i]["unscaled_sboxes"] = box_cxcywh_to_xyxy(targets[i]["sboxes"]) * img_resize
                targets[i]["unscaled_oboxes"] = box_cxcywh_to_xyxy(targets[i]["oboxes"]) * img_resize
        imgs = torch.stack(imgs)
        return imgs, targets

    def __getitem__(self, index):   
        if self.image_set=="train":
            # vidvrd: ILSVRC2015_train_00729000_0; vidor: 0000_2401075277_12
            begin_frame = self.train_begin_fids[index]
            subid, video_id, begin_fid = self.parse_frame_name(begin_frame)
            begin_fid = int(begin_fid)
            
            #duration = self.max_duration
            #if self.max_durations[index] < 8:
            #    duration = self.max_durations[index]
            #elif duration > self.max_durations[index]:  # 7, 15, 31
            #    duration = 8

            end_fid = begin_fid + self.max_durations[index]  # end_fid is not included
            
            clip_imgs = self.read_video_clip(video_id, begin_fid, end_fid)
            clip_fids = list(range(begin_fid, end_fid)) 
            if len(clip_fids)<8:
                clip_fids = clip_fids+clip_fids
            random.shuffle(clip_fids)
            # we randomly select 8 frames from [begin_frameid, begin_frameid+duration]
            seq_fids = sorted(clip_fids[:self.seq_len])
            imgs = clip_imgs[ [fid-begin_fid for fid in seq_fids] ] # num_frame,H,W,3 #.permute(1,2,0,3)
            if imgs.shape[0]!=8:
                import pdb;pdb.set_trace()
        else:
            video_id = self.idx2vid[index]
            valid_frame_list = np.asarray(self.val_valid_frames[video_id], dtype=int)
            frame_count = len(valid_frame_list)
            seq_fids = np.argwhere(valid_frame_list).reshape(-1)
            clip_imgs = self.read_video_clip(video_id, 0, frame_count)
            imgs = clip_imgs[seq_fids]
            #if self.dbname == "vidvrd":
            #    rel_tag_targets = self.annotations[video_id]["rel_tag_insts"]
            #return video_id, rel_tag_targets, self.fid2int(self.val_frame_dict[video_id])

        h, w = imgs.shape[1:3]
        targets = self.get_video_gt(video_id, seq_fids, w, h)
        
        imgs, targets = self.transforms(imgs, targets)
        for i, gt in enumerate(targets):
            h, w = gt["size"]
            img_resize = torch.as_tensor([w, h, w, h])
            targets[i]["unscaled_sboxes"] = box_cxcywh_to_xyxy(targets[i]["sboxes"]) * img_resize
            targets[i]["unscaled_oboxes"] = box_cxcywh_to_xyxy(targets[i]["oboxes"]) * img_resize
        
        
        """
        targets is a list (len=num_frame), each element is a dict as follows:
            'sboxes', 'oboxes': tensor([[0.8129, 0.5718, 0.1405, 0.1302], [0.4376, 0.5466, 0.2332, 0.1834]])
            'sarea', 'oarea': tensor([23082.4492,  9873.2051])
            'sclss', 'oclss': tensor([20, 28])
            'so_traj_ids': tensor([[0, 1], [1, 0]])
            'vclss': 2, num_vclss (one-hot vector)
            'raw_vclss': [[88, 72, 0, 65], [68, 56, 111]]  
            'orig_size': tensor([ 720, 1280])S
            'size': tensor([614, 879])
            'num_svo': 2
            'svo_ids': tensor([0, 1])
            'unscaled_sboxes', 'unscaled_oboxes':  tensor([[282.1861, 279.3351, 487.1930, 391.9286], [652.7425, 311.1363, 776.2614, 391.0691]])
        """
        return imgs, targets

    # ========
    # Test
    # ========
    def _get_anno_files(self, split):
        anno_files = glob.glob(os.path.join(self.anno_rpath, '{}/*.json'.format(split)))
        assert len(anno_files)>0, 'No annotation file found. Please check if the directory is correct.'
        return anno_files


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


def make_video_transforms(split, debug=False, cautious=True, by_ratio=False, resolution="large"):
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
        scales = [288, 320, 352, 384, 416, 448, 480]
        max_size = 800
        resizes = [240, 300, 360]
        crop = 240
    else:
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        max_size = 1333
        resizes = [400, 500, 600]
        crop = 384
    test_size = [800]
    scale = [0.8, 1.0]
    if debug:
        return T.Compose(
            horizontal
            + [
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose(
                        [
                            T.RandomResize(resizes),
                            T.RandomSizeCrop(crop, max_size, scale, respect_boxes=cautious, by_ratio=by_ratio),
                            T.RandomResize(scales, max_size=max_size),
                        ]
                    ),
                ),
                T.ToTensor(),
            ]
        )
    if split == "train":
        return T.Compose(
            horizontal
            + [
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose(
                        [
                            T.RandomResize(resizes),
                            T.RandomSizeCrop(crop, max_size, scale, respect_boxes=cautious, by_ratio=by_ratio),
                            T.RandomResize(scales, max_size=max_size),
                        ]
                    ),
                ),
                normalize,
            ]
        )
    else:
        return T.Compose([T.RandomResize(test_size, max_size=max_size), normalize])

    
def build_tagging(image_set, args):
    root = Path(args.vidvrd_path)
    assert root.exists(), f'provided VidVRD path {root} does not exist'

    dbname = args.dataset
    data_dir = args.vidvrd_path
    max_duration = args.max_duration
    anno_file = "data/metadata/%s_annotations.pkl"%dbname
    trainval_imgset_file = "data/metadata/%s_%s_frames_v2.json"%(dbname, image_set)

    transforms = make_video_transforms(image_set, args.debug, args.cautious, args.by_ratio, args.resolution)

    dataset = VRDTagging(
        dbname,
        image_set,
        data_dir,
        max_duration,
        anno_file,
        transforms=transforms,
        trainval_imgset_file=trainval_imgset_file, 
        seq_len=args.seq_len,
        num_quries=args.num_queries,
        num_verb_class=args.num_verb_class)

    return dataset

