# VLM-Driven Monocular–Stereo Fusion Model

This repository provides an official implementation of a VLM-driven monocular–stereo fusion model.  
More materials please refer to [3D Visual Illusion Depth Estimation](https://github.com/YaoChengTang/3D-Visual-Illusion-Depth-Estimation) and [Stereo Simulator](https://github.com/YaoChengTang/StereoSimulator).

---

## 🚀 Installation

Create and activate the environment:

```bash
conda env create -n illusion -f envs/environment.yaml
conda activate illusion
```

---

## 📥 Download

### 1. Install HuggingFace CLI
```bash
pip install "huggingface_hub[cli]"
huggingface-cli login
```

### 2. Download Required Checkpoints
```bash
huggingface-cli download Djrango/Qwen2vl-Flux --local-dir ~/Downloads/qwen2vl_flux
huggingface-cli download AdamYao/3D_Visual_Illusion_Depth_Estimation --local-dir ~/Downloads/3DVIllusion
cd ./3DVIllusion
cat model.pth-part-* > model.pth
```

### 3. Download Depth Anything V2
Download the DepthAnything V2 checkpoint from:
[Depth Anything V2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2)

---

## 🔍 Inference

Specify the dataset (root path and list file) and model weight paths in `script/infer.sh`

Run inference:
```bash
bash script/infer.sh
```

---

## 📄 Notes
- Ensure dataset paths and file lists in `infer.sh` are correct.
- `model.pth` must be reassembled from part files before use.
- More evaluation scripts, please refer to:  
  [3DVisualIllusion Evaluation Scripts](https://github.com/YaoChengTang/StereoSimulator/)

---

