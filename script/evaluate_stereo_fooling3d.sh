# /usr/bin/bash

export LOG_ROOT="/data5/yao/runs/log"
export TB_ROOT="/data5/yao/runs/tboard"
export CKPOINT_ROOT="/data5/yao/runs/ckpoint"

export CUDA_VISIBLE_DEVICES=0



# python evaluate_stereo_fooling3d.py --model_name "RAFTStereoDepthPostFusionNoDepthMonoFea" --depthany_model_dir "/data5/yao/pretrained" --restore_ckpt "/data5/yao/runs/ckpoint/Fooling3D_NoME_PostFusion_20250304_094912/30000_Fooling3D_NoME_PostFusion.pth" --dataset fooling3d

# python evaluate_stereo_fooling3d.py --model_name "RAFTStereoDepthAdaptivePostFusion" --valid_iters 32 --valid_fusion_iters 27 --depthany_model_dir "/data5/yao/pretrained" --restore_ckpt "/data5/yao/runs/ckpoint/Fooling3D_ME_AdaptivePostFusion2_20250301_211301/70000_Fooling3D_ME_AdaptivePostFusion2.pth" --dataset fooling3d
