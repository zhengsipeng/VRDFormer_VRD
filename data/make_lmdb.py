import os
import os.path as osp
import os, sys
import os.path as osp
from PIL import Image
import six
import string
import glob
import lmdb
import pickle
import msgpack
from tqdm import tqdm
import pyarrow as pa
import cv2
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length =pa.deserialize(txn.get(b'__len__'))
            self.keys= pa.deserialize(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def path_remake(path):
    return path.replace(' ', '\ ').replace('(','\(').replace(')','\)').replace('&','\&').replace(';', '\;')


def folder2lmdb(dbname, write_frequency=10, num_workers=16):
    
    lmdb_dir = "/data4/zsp/data/"
    if not os.path.exists(lmdb_dir):
        os.makedirs(lmdb_dir)
    lmdb_path = osp.join(lmdb_dir, dbname)
    isdir = os.path.isdir(lmdb_path)
    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)
    txn = db.begin(write=True)

    if dbname == "vidor":
        rootdir = "/data4/zsp/data/%s/images/**/*"%dbname
    else:
        rootdir = "/data4/zsp/data/%s/images/*"%dbname
    video_dirs = glob.glob(rootdir)
    keys = []
    idx = 0
    for video_dir in tqdm(video_dirs):
        video = video_dir.split("/")[-1]
        
        img_names = os.listdir(video_dir)
    
        for img_name in img_names:
            
            imgpath = video_dir + "/" + img_name
            with open(imgpath, "rb") as f:
                image = f.read()
   
            key = video + "-" + img_name.split(".")[0]
            keys.append(key)

            #txn.put(u'{}'.format(key).encode('ascii'), dumps_pyarrow(image))
            txn.put(key.encode(), image)
            if idx % write_frequency == 0:
                txn.commit()
                txn = db.begin(write=True)
            idx += 1
        #if idx > 100:
        #    break
            

    #keys = [u'{}'.format(k).encode('ascii') for k in keys]
    txn.commit()
    with db.begin(write=True) as txn:
        txn.put('__keys__'.encode(), dumps_pyarrow(keys))
        txn.put('__len__'.encode(), dumps_pyarrow(len(keys)))
    txn = db.begin(write=False)

    db.sync()
    db.close()


import numpy as np
import io
def read_from_lmdb(dbname):
    lmdb_dir = "/data4/zsp/data/lmdb"
    if not os.path.exists(lmdb_dir): os.makedirs(lmdb_dir)
    lmdb_path = osp.join(lmdb_dir, dbname)
    env = lmdb.open(lmdb_path, subdir=osp.isdir(lmdb_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
    txn = env.begin(write=False) 
    byteflow = txn.get("__keys__".encode())
    keys = pa.deserialize(byteflow)

    for i, key in enumerate(keys):
        print(key)
        byteflow = txn.get(key.encode())#.decode()

        imgs = Image.open(io.BytesIO(byteflow))
        print(imgs.size)
        assert 1==0
        byteflow = np.frombuffer(byteflow, dtype=np.uint8)
        print(byteflow.shape)
        img = cv2.imdecode(byteflow, cv2.IMREAD_COLOR)
        print(img.shape)
        img = Image.open(byteflow)
        assert 1==0
        img_bytes = pa.deserialize(byteflow)
        print(len(img_bytes))
        img = Image.open(img_bytes)
        #img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        print(i, key, img.shape)
        assert 1==0


if __name__ == "__main__":
    folder2lmdb("vidvrd", num_workers=16)
    #read_from_lmdb("kinetics100")
