python main.py \
--batch_size 2 --val_batch_size 2 \
--inter_attn_vis \
--dilation --lr_drop 40 \
--data_path /home/ttfang/dataset/split_dota_v1_800_05_pos \
--resume /home/ttfang/code/DAB-DETR/output/r50_e80/checkpoint0079.pth \
--output_dir output/r50_e80