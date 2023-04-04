"""
python extract_frame.py 
    --dataset vidor 
    --vpath /data4/datasets/vidor/video 
    --savepath /data4/datasets/vidor/images
"""
from joblib import delayed, Parallel
import os 
import glob 
from tqdm import tqdm 
import cv2
import argparse

import matplotlib.pyplot as plt
plt.switch_backend('agg')


def str2bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]


def extract_video_opencv(v_path, save_root, dataset):
    v_class, v_name = v_path.split('/')[-2:]
    v_name = v_name.split(".")[0]
    if dataset == "vidvrd":
        out_dir = os.path.join(save_root, v_name)
    else:
        out_dir = os.path.join(save_root, v_class, v_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    vidcap = cv2.VideoCapture(v_path)
    nb_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    if (width == 0) or (height == 0):
        print(v_path, 'not successfully loaded, drop ..'); return
    #new_dim = resize_dim(width, height, dim)

    success, image = vidcap.read()
    count = 1
    while success:
        cv2.imwrite(os.path.join(out_dir, '%06d.jpg' % count), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])# quality from 0-100, 95 is default, high is good
        success, image = vidcap.read()
        count += 1
      
    # Correct the amount of frames
    if (count * 30) < nb_frames:
        assert 1==0
        nb_frames = int(nb_frames * 30 / 1000)

    if nb_frames > count:
        print('/'.join(out_dir.split('/')[-2::]), 'NOT extracted successfully: %df/%df' % (count, nb_frames))

    vidcap.release()


def resize_dim(w, h, target):
    '''resize (w, h), such that the smaller side is target, keep the aspect ratio'''
    if w >= h:
        return (int(target * w / h), int(target))
    else:
        return (int(target), int(target * h / w)) 


def extract_frames(dataset, v_root, save_root):
    print('extracting %s ... '%dataset)
    print('extracting videos from %s' % v_root)
    print('frame save to %s' % save_root)

    if not os.path.exists(save_root): os.makedirs(save_root)

    if dataset == "vidvrd":
        v_paths = glob.glob(os.path.join(v_root, "*.mp4"))
        for vpath in tqdm(v_paths):
            extract_video_opencv(vpath, save_root, dataset)
    else:
        v_act_root = glob.glob(os.path.join(v_root, '*/'))

        
        for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
            v_paths = glob.glob(os.path.join(j, '*.mp4'))
            Parallel(n_jobs=32)(delayed(extract_video_opencv)(vpath, save_root, dataset) for vpath in v_paths)


if __name__ == '__main__':
    # v_root is the video source path, f_root is where to store frames
    # edit 'your_path' here:

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="vidor", type=str)
    parser.add_argument('--vpath', default="vidor", type=str)
    parser.add_argument('--savepath', default="vidor", type=str)
    args = parser.parse_args()

    extract_frames(args.dataset, v_root=args.vpath, save_root=args.savepath)

