# Run inference 
python3 inference_predict.py --clip_path $1 --anno_fn $2 --output_fn $3 --cfg './config/eval.yaml'
python3 inference_results.py --clip_path $1 --anno_fn $2 --output_fn $3 --cfg './config/eval.yaml'