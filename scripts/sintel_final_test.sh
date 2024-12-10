#!/bin/bash

python -u inference.py \
  --task=flow \
  --data_root=data/sintel \
  --data_list=lists/MPISintel_train.txt \
  --context \
  --encoder=dlaup \
  --decoder=hda \
  --batch_size=1 \
  --workers=16 \
  --flow_format=flo \
  --evaluate \
  --model_path=model_zoo/hd3fc_chairs_things_sintel-0be17c83.pth \
  --save_folder=save_folder/sintel_final
