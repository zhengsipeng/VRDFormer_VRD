import sys
import json
import glob
import numpy as np
import pickle as pkl
from tqdm import tqdm


def is_cls_match(begin_fid, duration, metadata):
    try:
        s_meta = metadata[begin_fid]
    except:
        import pdb;pdb.set_trace()
    e_meta = metadata[begin_fid+duration-1]
    s_so_trajs, e_so_trajs = s_meta['so_traj_ids'], e_meta['so_traj_ids']
    s_vclss, e_vclss = s_meta['vclss'], e_meta['vclss']
    if len(s_so_trajs)!=len(e_so_trajs):
        #import pdb;pdb.set_trace()
        return False
    for i, s_so_traj in enumerate(s_so_trajs):
        e_so_traj = e_so_trajs[i]
        if s_so_traj[0]!=e_so_traj[0] or s_so_traj[1]!=e_so_traj[1]:
            return False
    for i, s_v in enumerate(s_vclss):
        str_s_v = ' '.join(str(j) for j in s_v)
        str_e_v = ' '.join(str(j) for j in e_vclss[i])
        if str_s_v != str_e_v:
            return False
    return True


def get_trainval_frameids(root_dir, dbname, split, timestep, min_max_duration=32):
    postfix = ""
    anno_files = []
    if dbname == "vidor_part":
        dbname, postfix = dbname.split("_")

    if dbname == "vidor":
        anno_files += glob.glob("%s/vidor/annotations/%s/**/*.json"%(root_dir, split))
    else:
        anno_files += glob.glob("%s/vidvrd/annotations/%s/*.json"%(root_dir, split))
    print(len(anno_files))
    
    with open("metadata/%s_annotations.pkl"%dbname, "rb") as f:
        meta_anno = pkl.load(f)

    if "val" in split:
        val_pos_frames = dict()
        for anno_f in tqdm(anno_files):
            if "vidor" in dbname:
                subid, vid = anno_f.split(".")[0].split("/")[-2:]
            else:
                vid = anno_f.split("/")[-1].split(".")[0]
            
            with open(anno_f, "r") as f:
                data = json.load(f)
            pos_fids = np.zeros(data["frame_count"])
            for rel in data["relation_instances"]:
                for idx in range(rel["begin_fid"], rel["end_fid"]):
                    pos_fids[idx] = 1
            
            pos_fids = pos_fids.tolist()
            val_pos_frames[vid] = pos_fids
        with open("metadata/%s_val_frames_v2.json"%dbname, "w") as f:
            json.dump(val_pos_frames, f)
        return
    
    fail_fnum = 0
    succ_fnum = 0
    train_begin_fids, durations, total_fnum, video_num = [], [], 0, 0
    for anno_f in tqdm(anno_files):
        video_num += 1
        if postfix!="" and video_num>1000:  # vidor_part
            break
        if "vidor" in dbname:
            subid, vid = anno_f.split(".")[0].split("/")[-2:]
        else:
            vid = anno_f.split("/")[-1].split(".")[0]
        
        with open(anno_f, "r") as f:
            data = json.load(f)

        fnum = data["frame_count"]
        total_fnum += fnum
        pos_fids = np.zeros(fnum)
        
        for rel in data["relation_instances"]:
            for idx in range(rel["begin_fid"], rel["end_fid"]):
                pos_fids[idx] = 1
        
        selected_fnum = 0
        for idx in range(0, fnum, timestep):
            # min_max_duration means how long the positive frames are begining from current frame
            max_duration = 24
            duration = min_max_duration
            for i in range(0, min_max_duration):
                if idx+i >= fnum or pos_fids[idx+i] == 0:
                    duration = i
                    break
            if duration < 4:
                continue
            for duration in range(duration, 0, -1):
                if idx+duration>=fnum or pos_fids[idx+duration-1]==0 :
                    continue
                if idx+duration<fnum and pos_fids[idx+duration-1]!=0 and is_cls_match(idx, duration, meta_anno[vid]["frame_annos"]):
                    break
            if duration < 4:
                fail_fnum += 1
                continue
            else:
                succ_fnum +=1
            
            if "vidor" in dbname:
                sampled_frame = subid + "-" + vid + "-" + str(idx)
            else:
                sampled_frame = vid + "-" + str(idx)
            
            train_begin_fids.append(sampled_frame)
            durations.append(duration)
            selected_fnum += 1

    print(total_fnum, "Num of sampled frames: %d"%len(train_begin_fids))
    print(succ_fnum, fail_fnum)
    postfix = "_"+postfix if postfix != "" else postfix
    
    with open("metadata/%s_%s_frames%s_v2.json"%(dbname.split("_")[0], split, postfix), "w") as f:
        json.dump({"%s_begin_fids"%split: train_begin_fids, "durations": durations}, f)


def generate_vrd_annos(root_dir, dbname):
    postfix = ""
    if dbname == "vidor_part":
        dbname, postfix = dbname.split("_")

    with open("%s/action.txt"%dbname, "r") as f:
        action_list = [l.strip() for l in f.readlines()]
    action_dict = dict(zip(action_list, range(len(action_list))))
    with open("%s/obj.txt"%dbname, "r") as f:
        obj_list  = [l.strip() for l in f.readlines()]
    obj_dict = dict(zip(obj_list, range(len(obj_list))))

    if dbname == "vidor":
        anno_files = []
        for split in ["training", "validation"]:
            anno_files += glob.glob("%s/vidor/annotations/%s/**/*.json"%(root_dir, split))
    else:
        anno_files = []
        for split in ["train", "val"]:
            anno_files += glob.glob("%s/vidvrd/annotations/%s/*.json"%(root_dir, split))

    vrd_anno = dict()
    video_num = 0
    for anno_file in tqdm(anno_files):
        video_num += 1
        if postfix != "" and video_num > 1000:
            break

        if dbname == "vidvrd":
            vid = anno_file.split(".")[0].split("/")[-1]
        else:
            subid, vid = anno_file.split(".")[0].split("/")[-2: ]
            vid = subid + "-" + vid

        with open(anno_file, "r") as f:
            data = json.load(f)
        rels = data["relation_instances"]
        print(rels)
        trajectories = data["trajectories"]
        traj_objs = data["subject/objects"]
        #fnum = data["frame_count"]
      
        each_frame_dict = dict()
        rel_tag_uids = dict()  # dict of rel_uid (corresponding so_traj and begin/end fid) for each frame

        for rel in rels:
            begin, end = rel["begin_fid"], rel["end_fid"]
            stid, otid = rel["subject_tid"], rel["object_tid"]
            pclass = rel["predicate"]
            pclassid = action_dict[pclass] 
            
            for fid in range(begin, end):
                if fid not in each_frame_dict.keys():
                    each_frame_dict[fid] = {}
                
                rel_uid = "%s-%s-%d-%d"%(stid, otid, begin, end)
                if rel_uid not in rel_tag_uids.keys():
                    rel_tag_uids[rel_uid] = [pclassid]
                if pclassid not in rel_tag_uids[rel_uid]:
                    rel_tag_uids[rel_uid].append(pclassid)

                so_id = "%s-%s"%(stid, otid)
                if so_id not in each_frame_dict[fid].keys():
                    each_frame_dict[fid][so_id] = [pclassid]
                if pclassid not in each_frame_dict[fid][so_id]:
                    each_frame_dict[fid][so_id].append(pclassid)
        
        for fid, each_frame in each_frame_dict.items():
            so_traj_ids = []
            sclss, oclss, vclss = [], [], []
            sboxes, oboxes = [], []
            for so, v in each_frame.items():
                stid, otid = int(so.split("-")[0]), int(so.split("-")[1])
                so_traj_ids.append([stid, otid])
                
                scls, ocls = None, None
                for traj in traj_objs:
                    if traj["tid"] == stid:
                        scls = traj["category"]
                    if traj["tid"] == otid:
                        ocls = traj["category"]
                assert scls and ocls

                sclsid, oclsid = obj_dict[scls], obj_dict[ocls]
                sclss.append(sclsid)
                oclss.append(oclsid)
                vclss.append(v)
                boxes = trajectories[fid]

                for box in boxes:
                    if box["tid"] == stid:
                        sboxes.append([box["bbox"]["xmin"], box["bbox"]["ymin"], box["bbox"]["xmax"], box["bbox"]["ymax"]])
                    if box["tid"] == otid:
                        oboxes.append([box["bbox"]["xmin"], box["bbox"]["ymin"], box["bbox"]["xmax"], box["bbox"]["ymax"]])
               
            assert len(sboxes) == len(sclss)
            assert len(list(set(vclss[0]))) >= len(vclss[0])
            
            each_frame_dict[fid] = {"sclss": sclss, "oclss": oclss, "vclss": vclss,
                                    "so_traj_ids": so_traj_ids, 
                                    "sboxes": sboxes, "oboxes": oboxes}
        #import pdb;pdb.set_trace()
        #print(vid)
        vrd_anno[vid] = {"frame_annos": each_frame_dict, "rel_tag_uids": rel_tag_uids}

    with open("metadata/%s_annotations.pkl"%dbname, "wb") as f:
        pkl.dump(vrd_anno, f)


def check_annotation(dbname):
    min_max_duration = 24 if dbname == "vidvrd" else 32
    with open("metadata/%s_train_frames.json"%dbname, "r") as f:
        data = json.load(f)
    with open("metadata/%s_annotations.pkl"%dbname, "rb") as f:
        annotations = pkl.load(f)

    for frame_name in tqdm(data["train_begin_fids"]):
        for i in range(min_max_duration):
            subid, video_id, fid = frame_name.split("-")[-3:]
            fid = int(fid)
            #import pdb;pdb.set_trace()
            anno = annotations[subid+"-"+video_id]["frame_annos"][fid+i]
            if len(anno) == 0:
                print(anno)
                print(frame_name)
                assert 1==0
    import pdb;pdb.set_trace()


def get_max_numpair_frame(dbname):
    with open("/data3/zsp/data/%s/vrd_annotation.pkl"%dbname, "rb") as f:
        annos = pkl.load(f)
    max_num = 0
    num_20 = 100
    for vid in annos.keys():
        each_frame_dict = annos[vid]
        for fid, fanno in each_frame_dict.items():
            num_svo = len(fanno["vclss"])
            if num_svo > 100:
                num_20 += 1
                print(num_20)
            if num_svo > max_num:
                max_num = num_svo
                print("max_num", max_num)
            break
          

if __name__ == "__main__":
    root_dir = "/data2/zsp/data/"
    func = 0
    dbname = "vidvrd" # vidvrd, vidor, vidor_part 
    #check_annotation(dbname)
    if func == 0:
        split, timestep, min_max_duration = sys.argv[1:]
        # vidvrd: train/val;  vidor: training/validation
        get_trainval_frameids(root_dir, dbname, split, int(timestep), int(min_max_duration))  
        # python prepare_annotation.py vidvrd val 1 24
        # 67774:  vidvird, 1, 24
        # 367,200(7,145,644 in total): vidor, 16,32
        # 367,200(7,145,644 in total): vidor, 33,32
        
        #vis_val_idxs("vidvrd")
    else:
        generate_vrd_annos(root_dir, dbname)