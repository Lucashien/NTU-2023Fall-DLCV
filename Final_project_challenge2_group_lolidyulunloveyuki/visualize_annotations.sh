# Convert output json to annotated json
python3 convert_pseudo_label.py --unannotated_json_fn $1 --pred_json_fn $2

# Use annotated json to generated videos
python3 visualize_annotations.py --clips-root $3 
