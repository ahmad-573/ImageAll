#!/bin/bash

DATA_DIR="../../../data/drive_2/hmdb51"

evaluation() {
  python eval.py \
    --test_dir "$DATA_DIR" \
    --data_type "$1" \
    --src_model "$2" \
    --img_size "$3" \
    --pre_trained "$4" \
    --batch_size 10 \
    --num_classes 51 \
    --num_temporal_views 3
}


evaluation "hmdb51" "vit_base_patch16_224_timeP_1" "224" "../timesformer/TimeSformer/train_output/196/hmdb/vit/1_224_joint_1p_no_timeembed/results/checkpoint_epoch_00015.pyth" 