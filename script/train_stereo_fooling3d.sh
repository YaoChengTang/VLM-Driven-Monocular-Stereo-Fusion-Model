# /usr/bin/bash


# 获取当前 shell 文件名（不包含路径和扩展名）
SCRIPT_NAME=$(basename "$0" .sh)

# 获取当前时间
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")

# 如果有参数，使用参数作为文件夹名，否则使用脚本名加时间
if [ -n "$1" ]; then
    FOLDER_NAME="${1}_${CURRENT_TIME}"
    EXP_NAME="${1}"
else
    FOLDER_NAME="${SCRIPT_NAME}_${CURRENT_TIME}"
    EXP_NAME="${SCRIPT_NAME}"
fi


# export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0  # 设置正确的网络接口
# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=29501
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=1,2

# "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/Sceneflow"
# "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/Middlebury"
# "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/KITTI2015"
# "/horizon-bucket/saturn_v_dev/01_users/chengtang.yao/ETH3D"
# export DATASET_ROOT="/data6/sceneflow/sceneflow"
export DATASET_ROOT="./datasets/Fooling3D"

export LOG_ROOT="/data5/yao/runs/log/${FOLDER_NAME}"
export TB_ROOT="/data5/yao/runs/tboard/${FOLDER_NAME}"
export CKPOINT_ROOT="/data5/yao/runs/ckpoint/${FOLDER_NAME}"

# 输出新的路径，确认设置正确
echo "LOG_ROOT is set to: $LOG_ROOT"
echo "TB_ROOT is set to: $TB_ROOT"
echo "CKPOINT_ROOT is set to: $CKPOINT_ROOT"

nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)  # Count the number of GPUs

# torchrun --nnode 1 --nproc_per_node $nproc_per_node --master_port 29501 train_stereo_fooling3d.py --batch_size 32 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 50000 --mixed_precision --model_name "RAFTStereoDepthBetaRefine" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea --restore_ckpt "/data5/yao/runs/ckpoint/RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim_20241102_014050/50000_RaftStereoDepthBetaK53DispRefineSigmoidPreMonoBatch48ConfDim.pth" --lr 0.0005 --train_refine_mono --exp_name "Fooling3D_BetaConf"

# torchrun --nnode 1 --nproc_per_node $nproc_per_node --master_port 29500 train_stereo_fooling3d.py --batch_size 32 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 50000 --mixed_precision --model_name "RAFTStereoDepthPostFusion" --depthany_model_dir "/data5/yao/pretrained" --restore_ckpt "/data5/yao/runs/ckpoint/RaftStereoDepthAny_20240908_125231/RaftStereoDepthAny.pth" --lr 0.0005 --exp_name "Fooling3D_ME_PostFusion"

# torchrun --nnode 1 --nproc_per_node $nproc_per_node --master_port 29502 train_stereo_fooling3d.py --batch_size 8 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 50000 --mixed_precision --model_name "RaftStereoDisp" --restore_ckpt "/data5/yao/runs/ckpoint/RaftStereoDisp_20240821_142314/RaftStereoDisp.pth" --exp_name "Fooling3D_RaftStereo"


# torchrun --nnode 1 --nproc_per_node $nproc_per_node --master_port 29501 train_stereo_fooling3d.py --batch_size 32 --train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 50000 --mixed_precision --model_name "RAFTStereoDepthBetaRefine" --depthany_model_dir "/data5/yao/pretrained" --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea --restore_ckpt "/data5/yao/runs/ckpoint/Fooling3D_BetaConf_20250225_104116/Fooling3D_BetaConf.pth" --lr 0.0005 --train_refine_mono --exp_name "Fooling3D_BetaConf2"

# torchrun --nnode 1 --nproc_per_node $nproc_per_node --master_port 29502 train_stereo_fooling3d.py --batch_size 8 --train_iters 5 --valid_iters 5 --train_fusion_iters 17 --valid_fusion_iters 27 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 50000 --mixed_precision --model_name "RAFTStereoDepthAdaptivePostFusion" --depthany_model_dir "/data5/yao/pretrained" --restore_ckpt "/data5/yao/pretrained" --restore_ckpt "/data5/yao/runs/ckpoint/RaftStereoDepthAny_20240908_125231/RaftStereoDepthAny.pth" --lr 0.0005 --exp_name "Fooling3D_ME_AdaptivePostFusion"

# torchrun --nnode 1 --nproc_per_node $nproc_per_node --master_port 29502 train_stereo_fooling3d.py --batch_size 8 --train_iters 22 --valid_iters 32 --train_fusion_iters 22 --valid_fusion_iters 27 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthAdaptivePostFusion" --depthany_model_dir "/data5/yao/pretrained" --restore_ckpt "/data5/yao/pretrained" --restore_ckpt "/data5/yao/runs/ckpoint/Fooling3D_ME_AdaptivePostFusion_20250228_042118/Fooling3D_ME_AdaptivePostFusion.pth" --lr 0.0005 --exp_name "Fooling3D_ME_AdaptivePostFusion2"

torchrun --nnode 1 --nproc_per_node $nproc_per_node --master_port 29503 train_stereo_fooling3d.py --batch_size 8 --train_iters 22 --valid_iters 32 --train_fusion_iters 22 --valid_fusion_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2 --num_steps 100000 --mixed_precision --model_name "RAFTStereoDepthAdaptiveSingle" --depthany_model_dir "/data5/yao/pretrained" --restore_ckpt "/data5/yao/pretrained" --restore_ckpt "/data5/yao/runs/ckpoint/RaftStereoDepthAny_20240908_125231/RaftStereoDepthAny.pth" --lr 0.0005 --exp_name "Fooling3D_ME_AdaptiveSingle"
