import os
import glob
import pickle

import pandas as pd

from pathlib import Path



def extract_paths(root):
    df = pd.read_csv(os.path.join(root, 'meta_data/scale_factors.csv'), header=None, names=["right_image", "scale_factor"])
    scale_factors = df["scale_factor"].values
    file_paths = [p for p in df["right_image"]]
    file_paths_left = [path.replace("video_frame_sequence_right", "video_frame_sequence")  for path in file_paths]
    file_paths_disp = [path.replace("video_frame_sequence_right", "depth_rect")  for path in file_paths]
    file_paths_mask = []
    file_paths_mask_set = []
    for path in file_paths:
        tmp_path = path.replace("video_frame_sequence_right", "sam_mask")
        mask_dir = os.path.dirname(tmp_path)
        frame_name = os.path.splitext(os.path.basename(tmp_path))[0]
        mask_paths = glob.glob( os.path.join(mask_dir, f'{frame_name}*.jpg') )
        file_paths_mask += mask_paths

        rel_mask_paths = [str(Path(mask_path).relative_to(root)) for mask_path in mask_paths]
        file_paths_mask_set.append(rel_mask_paths)
    
    path_set = file_paths + file_paths_left + file_paths_disp + file_paths_mask
    path_dict = {"scale_factor": scale_factors, 
                 "right_image": [str(Path(path).relative_to(root)) for path in file_paths], 
                 "left_image":  [str(Path(path).relative_to(root)) for path in file_paths_left], 
                 "disp_image":  [str(Path(path).relative_to(root)) for path in file_paths_disp], 
                 "mask_image":  file_paths_mask_set,
                }

    print("Total files: ", len(path_set))
    print(len(scale_factors), len(file_paths), len(file_paths_left), len(file_paths_disp), len(file_paths_mask))

    return path_set, path_dict

# 使用示例
root = "/data2/Fooling3D"
required_files, required_files_dict = extract_paths(root)

# # 保存文件列表（自动处理路径格式）
# with open("/data5/yao/tmp/filelist.txt", 'w') as f:
#     for path in required_files:
#         f.write(f"{Path(path).relative_to(root)}\n")

# 保存字典为pkl文件
with open("/data5/yao/tmp/training_enter.pkl", 'wb') as f:
    pickle.dump(required_files_dict, f)


# rsync -avh --progress --files-from=/data5/yao/tmp/filelist.txt \
#     --relative "/data2/Fooling3D/./" \
#     yao@10.24.1.35:"/mnt/nvme2/Fooling3D"