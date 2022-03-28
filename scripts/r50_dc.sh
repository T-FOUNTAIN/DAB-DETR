EXP_DIR=output/r50_e80

if [ ! -d "output" ]; then
    mkdir output
fi

if [ ! -d "${EXP_DIR}" ]; then
    mkdir ${EXP_DIR}
fi

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --use_env main.py \
    --batch_size 2 \
    --val_batch_size 2 \
    --dilation \
    --epochs 80 \
    --lr_drop 40 \
    --data_path /home/ttfang/dataset/split_dota_v1_800_05_pos\
    --resume /home/ttfang/code/DAB-DETR/weights/checkpoint_03_23.pth\
    --output_dir ${EXP_DIR} \
     2>&1 | tee ${EXP_DIR}/detailed_log.txt

  # --multiscale \
      #--data_path /home/ttfang/dataset/split_dota_v1_800_05_pos\