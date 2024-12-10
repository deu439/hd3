python -u inference.py \
  --task=flow \
  --data_root=$DATA_ROOT \
  --data_list=lists/KITTI_flow_train_2012.txt \
  --encoder=dlaup \
  --context \
  --decoder=hda \
  --batch_size=1 \
  --workers=16 \
  --flow_format=flo \
  --evaluate \
  --model_path=/home/deu/Models/hd3/hd3sc_things_kitti-368975c0.pth \
  --save_folder=$SAVE_FOLDER