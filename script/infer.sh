# /usr/bin/bash

RUN_ROOT="/home/xxx/Downloads"
DATA_ROOT="/home/xxx/Data"

export LOG_ROOT="$RUN_ROOT/vis"
export TB_ROOT="$RUN_ROOT/tboard"
export CKPOINT_ROOT="$RUN_ROOT/ckpoint"
export CHECKPOINT_DIR="$RUN_ROOT/qwen2vl_flux"

python3 infer.py \
--restore_ckpt $RUN_ROOT/model.pth \
--depthany_model_dir $RUN_ROOT \
--model_name "RAFTStereoDepthBetaRefine" \
--lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" \
--modulation_ratio 1.0 \
--diff_num_inference_steps 12 \
--root $DATA_ROOT \
--img_path_txt $DATA_ROOT/path.txt \
--sv_root $RUN_ROOT/vis \
--test_exp_name "infer_test"

