#!/bin/bash

DATA_DIR="../../../data/drive_2/ucf101/annotations_svt"

evaluation() {
  python eval_adv_base.py \
    --test_dir "$DATA_DIR" \
    --data_type "$1" \
    --src_model "$2" \
    --tar_model "$3" \
    --image_list "$4" \
    --attack_type "$5" \
    --target_label "$6" \
    --iter "$7" \
    --eps "$8" \
    --index "$9" \
    --pre_trained "${10}" \
    --tar_pre_trained "${11}" \
    --num_temporal_views 3 \
    --num_classes 101 \
    --batch_size 10 \
    --num_frames 8 \
    --num_gpus 1 \
    --src_frames 1 \
    --eps 70 
}

#for ATTACK in 'mifgsm', 'dim'; do
#  for MODEL in "resnet152" "BIT" 'wide_resnet50_2' 'densenet201' 'T2t_vit_7' 'T2t_vit_24' 'tnt_s_patch16_224' 'vit_base_patch16_224' 'BIT'; do
#    for INDEX in  'last' 'all'; do
#      for RES in  '56' '96' '120' '224' '56_96' '56_96_120' '56_96_120_224'; do
#        evaluation_ens  'IN' "deit_base_patch16_224_resPT_1" "$MODEL"  "image_list_5k.json" "$ATTACK" -1 10 16 "$INDEX" "$RES"
#      done
#    done
#  done
#done

evaluation 'ucf101' "deit_base_patch16_224_base_prompt" "timesformer_vit_base_patch16_224"  "" "dim" -1 10 16 "all" "pretrained_video_models/baselines/prompt/ucf/deit/1_joint_1p/results/checkpoint_epoch_00015.pyth" "pretrained_video_models/timesformer/ucf/8_224_joint/results/checkpoint_epoch_00015.pyth"

#evaluation 'IN' "deit_base_patch16_224" "T2t_vit_7"  "image_list_5k.json" "pifgsm" -1 10 16 "last"
#evaluation 'IN' "deit_base_patch16_224" "T2t_vit_7"  "image_list_5k.json" "pifgsm" -1 10 16 "all"

#evaluation_ens  'IN' "deit_base_patch16_224_resPT_1" "resnet152"  "image_list_5k.json" "mifgsm" -1 10 16 "all" "56_96_120_224"