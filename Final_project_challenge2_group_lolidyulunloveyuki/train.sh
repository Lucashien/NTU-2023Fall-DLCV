# Run training
python3 train_anchor.py --cfg './config/train.yaml' --clip_path $1 --anno_dir $2 --Use_orig_data 'True'  \
                        --Use_UFS_3 'False' --Use_UFS_5 'False' --Use_UFS_gt_10 'False' --Use_UFS_gt_20 'False' --Use_UFS_test 'False' 
