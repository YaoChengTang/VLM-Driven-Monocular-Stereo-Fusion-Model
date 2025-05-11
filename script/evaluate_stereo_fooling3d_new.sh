# /usr/bin/bash

export LOG_ROOT="/data5/yao/runs/log"
export TB_ROOT="/data5/yao/runs/tboard"
export CKPOINT_ROOT="/data5/yao/runs/ckpoint"

export CUDA_VISIBLE_DEVICES=0



python evaluate_stereo_fooling3d_new.py --model_name "RAFTStereoDepthPostFusion" --valid_iters 32 --valid_fusion_iters 27 --depthany_model_dir "/data5/yao/pretrained" --restore_ckpt "/data5/yao/runs/ckpoint/Fooling3D_ME_PostFusion_20250225_103903/Fooling3D_ME_PostFusion.pth" --mixed_precision  --dataset fooling3d --sv_name Fooling3D_ME_PostFusion

python evaluate_stereo_fooling3d_new.py --model_name "RaftStereoDisp" --valid_iters 32 --valid_fusion_iters 27 --depthany_model_dir "/data5/yao/pretrained" --restore_ckpt "/data5/yao/runs/ckpoint/Fooling3D_RaftStereo_20250225_103952/Fooling3D_RaftStereo.pth" --mixed_precision  --dataset fooling3d --sv_name Fooling3D_RaftStereo

python evaluate_stereo_fooling3d_new.py --model_name "RAFTStereoDepthBetaRefine" --valid_iters 32 --valid_fusion_iters 27 --depthany_model_dir "/data5/yao/pretrained" --restore_ckpt "/data5/yao/runs/ckpoint/Fooling3D_BetaConf2_20250228_023601/Fooling3D_BetaConf2.pth"   --mixed_precision  --lbp_neighbor_offsets "(-5,-5), (5,5), (5,-5), (-5,5), (-3,0), (3,0), (0,-3), (0,3)" --modulation_ratio 1.0 --conf_from_fea  --dataset fooling3d --sv_name Fooling3D_BetaConf2


python evaluate_stereo_fooling3d_new.py --model_name "RAFTStereoDepthAdaptivePostFusion" --valid_iters 32 --valid_fusion_iters 27 --depthany_model_dir "/data5/yao/pretrained" --restore_ckpt "/data5/yao/runs/ckpoint/Fooling3D_ME_AdaptivePostFusion_20250228_042118/Fooling3D_ME_AdaptivePostFusion.pth" --mixed_precision --dataset fooling3d --sv_name Fooling3D_ME_AdaptivePostFusion

python evaluate_stereo_fooling3d_new.py --model_name "RAFTStereoDepthAdaptivePostFusion" --valid_iters 32 --valid_fusion_iters 27 --depthany_model_dir "/data5/yao/pretrained" --restore_ckpt "/data5/yao/runs/ckpoint/Fooling3D_ME_AdaptivePostFusion2_20250301_211301/70000_Fooling3D_ME_AdaptivePostFusion2.pth" --mixed_precision --dataset fooling3d --sv_name Fooling3D_ME_AdaptivePostFusion2



python evaluate_stereo_fooling3d_new.py --model_name "RAFTStereoDepthAdaptiveSingle" --valid_iters 32 --valid_fusion_iters 27 --depthany_model_dir "/data5/yao/pretrained" --restore_ckpt "/data5/yao/runs/ckpoint/Fooling3D_ME_AdaptiveSingle_20250302_003744/Fooling3D_ME_AdaptiveSingle.pth" --mixed_precision --dataset fooling3d --sv_name Fooling3D_ME_AdaptiveSingle  --valid_fusion_iters 27 


python evaluate_stereo_fooling3d_new.py --model_name "RAFTStereoDepthAdaptiveSingle" --valid_iters 32 --valid_fusion_iters 27 --depthany_model_dir "/data5/yao/pretrained" --restore_ckpt "/data5/yao/runs/ckpoint/Fooling3D_ME_AdaptiveSingle2_20250304_010820/60000_Fooling3D_ME_AdaptiveSingle2.pth" --mixed_precision --dataset fooling3d --sv_name Fooling3D_ME_AdaptiveSingle2  --valid_fusion_iters 27 


python evaluate_stereo_fooling3d_new.py --model_name "RAFTStereoDepthPostFusionNoDepthMonoFea" --valid_iters 32 --valid_fusion_iters 27 --depthany_model_dir "/data5/yao/pretrained" --restore_ckpt "/data5/yao/runs/ckpoint/Fooling3D_NoME_PostFusion_20250304_094912/60000_Fooling3D_NoME_PostFusion.pth" --mixed_precision --dataset fooling3d --sv_name Fooling3D_NoME_PostFusion