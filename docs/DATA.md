# DATA PREPARATION

## DataSet
first download the VidVRD and VidOR dataset, put them into root_dir, and then run the following script:
```
python data/prepare.py --func prep_vidor
```

the orgnization of dataset should be like this:
```
|-- vidor/vidvrd
|---- annotations
|------ train
|-------- xxx.json
|------ val
|---- videos
|------ xxx.mp4
|----
```

## Prepare Annotation
```
python prepare.py --func get_anno --dbname vidor
python prepare.py --func get_anno --dbname vidvrd
```

## Prepare FrameIDs for training
```
python prepare.py --func get_fid --dbname vidvrd --split train --stage 1 --timestep 1 --minmax_dur 24
python prepare.py --func get_fid --dbname vidvrd --split train --stage 2 --timestep 1 --minmax_dur 24
python prepare.py --func get_fid --dbname vidvrd --split val --timestep 1 --minmax_dur 24

python prepare.py --func get_fid --dbname vidor --split train --stage 1 --timestep 8 --minmax_dur 32
python prepare.py --func get_fid --dbname vidor --split train --stage 2 --timestep 8 --minmax_dur 32
python prepare.py --func get_fid --dbname vidor --split val --timestep 8 --minmax_dur 32

python prepare.py --func get_fid --dbname vidorpart --split train --stage 1 --timestep 8 --minmax_dur 32
python prepare.py --func get_fid --dbname vidorpart --split train --stage 2 --timestep 8 --minmax_dur 32
python prepare.py --func get_fid --dbname vidorpart --split val --timestep 8 --minmax_dur 32
```