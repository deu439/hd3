#!/bin/bash

python -u inference.py \
  --task=flow \
  --data_root=data/kitti12 \
  --data_list=lists/KITTI_flow_train_2012.txt \
  --context \
  --encoder=dlaup \
  --decoder=hda \
  --batch_size=1 \
  --workers=16 \
  --flow_format=png \
  --evaluate \
  --model_path=model_zoo/hd3fc_chairs_things_kitti-bfa97911.pth \
  --save_folder=save_folder/kitti12




