export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_SPEED_LIMIT=5MB

export HF_HUB_RETRIES=10
export HF_HUB_RETRY_DELAY=5


python main.py --mode variation \
              --input_image /mnt/nvme2/Fooling3D/video_frame_sequence/video0/300_3d_wall_painting_kaise_karen_asian_paints_colour_kaise_banaye_/frame_0026.png \
              --prompt "A beautiful landscape" \
              --image_count 4
