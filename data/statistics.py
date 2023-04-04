import sys
import glob
import json
from tqdm import tqdm


def static_cls_types(root_dir):
    json_files = []
    for split in ["train", "val"]:
        json_files += glob.glob("%s/vidor/annotations/%s/**/*.json"%(root_dir, split)) 
    print(len(json_files))
    
    subject_list, object_list, rel_list = [], [], []
    max_obj, max_rel = 0, 0
    for jsonfile in tqdm(json_files):
        with open(jsonfile, "r") as f:
            data = json.load(f)
        so_list = data["subject/objects"]
        rels = data["relation_instances"]
        trajs = data["trajectories"]

        if len(rels) > max_rel:
            max_rel = len(rels)

        for each_frame in trajs:
            if len(each_frame) > max_obj:
                max_obj = len(each_frame)
                print("Max obj: %d, Max rel: %d"%(max_obj, max_rel))

        for rel in rels:
            sid = rel["subject_tid"]
            oid  = rel["object_tid"]
            pclass = rel["predicate"]
            scls, ocls = so_list[sid]["category"], so_list[oid]["category"]
            if scls not in subject_list:
                subject_list.append(scls)
            if ocls not in object_list:
                object_list.append(ocls)
            if (scls, pclass, ocls) not in rel_list:
                rel_list.append((scls, pclass, ocls))
 
    with open("metadata/vidor_subject.txt", "w") as f:
        for subject in subject_list:
            f.writelines(subject+"\n")
    with open("metadata/vidor_object.txt", "w") as f:
        for obj in object_list:
            f.writelines(obj+"\n")
    with open("metadata/vidor_relation.txt", "w") as f:
        for rel in rel_list:
            s,p,o = rel
            f.writelines(s+"-"+p+"-"+o+"\n")


def is_rel_multilabel(root_dir, dbname):
    json_files = []
    if dbname == "vidor":
        for split in ["train", "val"]:
            json_files += glob.glob("%s/vidor/annotations/%s/**/*.json"%(root_dir, split))
    else:
        for split in ["train", "test"]:
            json_files += glob.glob("%s/vidvrd/annotations/%s/*.json"%(root_dir, split))
   
    for jsonfile in tqdm(json_files):
        with open(jsonfile, "r") as f:
            data = json.load(f)
        
        rels = data["relation_instances"]
        so_pair_dict = {}
      
        for rel in rels:
            sid = rel["subject_tid"]
            oid  = rel["object_tid"]
            begin_fid = rel["begin_fid"]
            end_fid = rel["end_fid"]
            pclass = rel["predicate"]

            if (sid, oid) in so_pair_dict.keys():
                for [p, begin_fid, end_fid] in so_pair_dict[(sid, oid)]:
                    if p == pclass or begin_fid >= end_fid or end_fid <= begin_fid:
                        continue
                    
                    print(sid, p, oid, begin_fid, end_fid)
                    print(sid, pclass, oid, begin_fid, end_fid)
                    assert 1==0
                so_pair_dict[(sid, oid)].append([pclass, begin_fid, end_fid])
            so_pair_dict[(sid, oid)] = [[pclass, begin_fid, end_fid]]
            

def get_anno_files(root_dir, dbname):
    anno_files = []
    if dbname == "vidor":
        for split in ["train", "val"]:
            anno_files += glob.glob("%s/vidor/annotations/%s/**/*.json"%(root_dir, split))
    else:
        for split in ["train", "test"]:
            anno_files += glob.glob("%s/vidvrd/annotations/%s/*.json"%(root_dir, split))
    print(len(anno_files))
    return anno_files


def get_minmax_action_len(root_dir, dbname):
    anno_files = get_anno_files(root_dir, dbname)
    
    minlen, maxlen = 3000, 0
    minnum, maxnum = 0, 0
    for anno_f in tqdm(anno_files):
        with open(anno_f, "r") as f:
            data = json.load(f)
        rels = data["relation_instances"]
        for rel in rels:
            duration = rel["end_fid"] - rel["begin_fid"]
            if duration > maxlen:
                maxlen = duration
            if duration < minlen:
                minlen = duration
            if duration < 24:
                minnum += 1
                print(minnum, maxnum)
            else:
                maxnum += 1
    print("MinMax: %d/%d"%(minlen, maxlen))


def get_minmax_wh_size(root_dir, dbname):
    anno_files = get_anno_files(root_dir, dbname)
    max_h, max_w = 0, 0
    for anno_f in tqdm(anno_files):
        with open(anno_f, "r") as f:
            data = json.load(f)
        width, height = data["width"], data["height"]
        if width > max_w:
            max_w = width
        if height > max_h:
            max_h = height
    print("Max Width and Height: %d/%d"%(max_w, max_h))


def cal_num_val_rinst(root_dir, dbname):
    if dbname == "vidor":
        anno_files = glob.glob("%s/vidor/annotations/val/**/*.json"%root_dir)
    else:
        anno_files = glob.glob("%s/vidvrd/annotations/test/*.json"%root_dir)
    num_rinsts = 0
    for anno_file in tqdm(anno_files):
        with open(anno_file, "r") as f:
            anno = json.load(f)
        num_rinsts += len(anno["relation_instances"])
    print(num_rinsts)


def cal_max_rinst_len(root_dir):
    anno_files = glob.glob("%s/vidor/annotations/val/**/*.json"%root_dir)
    min_num = 0
    total_num = 0
    for anno_file in tqdm(anno_files):
        with open(anno_file, "r") as f:
            anno = json.load(f)
        rinsts = anno["relation_instances"]
        for rinst in rinsts:
            total_num += 1
            begin_fid = rinst["begin_fid"]
            end_fid = rinst["end_fid"]
            duration = end_fid - begin_fid
            if duration<200 and duration>=100:
                min_num += 1
                print(duration, min_num, total_num)

    print(total_num)
    # 30142
    # <10 frame: 967
    # 10~100 frame:13761
    # 100~200 frame:6263
    # >200 frame:9117


def get_max_numpair_frame(root_dir, dbname):
    with open("%s/%s/vrd_annotations.json"%(root_dir, dbname), "rb") as f:
        annos = json.load(f)
    max_num = 0
    num_20 = 100
    for vid in annos.keys():
        each_framerame_dict = annos[vid]
        for fid, fanno in each_framerame_dict.items():
            num_svo = len(fanno["verb_classes"])
            if num_svo > 100:
                num_20 += 1
                print(num_20)
            if num_svo > max_num:
                max_num = num_svo
                print("max_num", max_num)
            break


if __name__ == "__main__":
    func = int(sys.argv[1])
    dbname = sys.argv[2]
    root_dir = "/data4/datasets/"
    if func == 1:
        static_cls_types(root_dir)
    elif func == 2:
        is_rel_multilabel(root_dir, dbname)  # Yes for both datasets
    elif func == 3:
        get_minmax_action_len(root_dir, dbname)  # vidvrd: 30/1200;  vidor: 3/5395(recommend: 16)
    elif func == 4:
        get_minmax_wh_size(root_dir, dbname)  # W/H VIDOR: 640/1138; VIDVRD: 1920/1080
    elif func == 5:
        cal_num_val_rinst(root_dir, dbname)
    elif func == 6:
        cal_max_rinst_len()
    elif func == 7:
        get_max_numpair_frame(root_dir, dbname)  # vidvrd 110