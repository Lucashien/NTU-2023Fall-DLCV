#!/bin/bash
# python3 P2_adapter.py
python3 P2_inference_adapter.py b

# # 定義一個含有不同預測檔案名稱的陣列
# PRED_FILES=(
#     "model_P2_0_b.json"
#     "model_P2_1_b.json"
#     "model_P2_2_b.json"
#     "model_P2_3_b.json"
#     "model_P2_4_b.json"
#     "model_P2_5_b.json"
#     "model_P2_6_b.json"
#     "model_P2_7_b.json"
#     "model_P2_8_b.json"
#     "model_P2_9_b.json"
#     "model_P2_10_b.json"
# )

# # 迴圈遍歷每個檔案名稱
# for FILE in "${PRED_FILES[@]}"
# do
#     echo "正在處理檔案: $FILE" >> output_P2.txt
#     # 執行 Python 腳本，並將當前檔案名稱作為參數，同時將輸出追加到 output.txt
#     python3 P2_evaluate.py --pred_file="$FILE" --annotation_file="../hw3_data/p2_data/val.json" --images_root="../hw3_data/p2_data/images/val" >> output_P2.txt 2>&1
# done

# echo "所有檔案處理完成" >> output_P2.txt
