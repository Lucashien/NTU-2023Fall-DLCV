import os
import pprint
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import itertools
import argparse
import json
import tqdm
from queue import Empty as QueueEmpty
import sys
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
from torch import multiprocessing as mp

def parse_args():
    parser = argparse.ArgumentParser(description='Convert output json to annotated json')

    parser.add_argument(
        "--unannotated_json_fn", type = str, default=sys.argv[1], help = "Path to the unannotated test json file")  
    
    parser.add_argument(
        "--pred_json_fn", type = str, default=sys.argv[2], help = "Path to the prediction json file")        
    
    args, rest = parser.parse_known_args()
    return args

def convert_label(orig_annotations, pred_annotation):

    orig = orig_annotations
    for clip_id, clip_set in orig.items():
        cnt = 0
        for annotation in clip_set["annotations"]:
            for qset_id, qset in annotation["query_sets"].items():
                pred_rt = pred_annotation[clip_id]["predictions"][cnt]["query_sets"][qset_id]["bboxes"]
                orig_w = qset["visual_crop"]["original_width"]
                orig_h = qset["visual_crop"]["original_height"]
                for i in pred_rt:
                    i["frame_number"] = i["fno"]
                    i["x"] = i["x1"]
                    i["width"] = i["x2"] - i["x1"]
                    i["y"] = i["y1"]
                    i["height"] = i["y2"]-i["y1"]  
                    i["original_width"] = orig_w
                    i["original_height"] = orig_h
                    del i["fno"]  
                    del i["x1"]
                    del i["x2"]
                    del i["y1"]      
                    del i["y2"]   
                qset["response_track"] = pred_rt
                qset["score"] = pred_annotation[clip_id]["predictions"][cnt]["query_sets"][qset_id]["score"]     
            cnt += 1  

    return orig


if __name__ == '__main__':

    args = parse_args()

    #orig_annotation_path = '/home/remote/mplin/DLCV/VQLoC/new_data/data/vq_test_unannotated.json'  
    orig_annotation_path = args.unannotated_json_fn  
    with open(orig_annotation_path) as fp:
        orig_annotations = json.load(fp)


    #pred_annotation_path = '/home/remote/mplin/DLCV/VQLoC/new_output/pred.json' 
    pred_annotation_path = args.pred_json_fn
    with open(pred_annotation_path) as fp:
        pred_annotation = json.load(fp)

    convert_annotations = convert_label(orig_annotations, pred_annotation)
    
    with open('./annotated_pred.json', 'w') as fp:
        out = json.dumps(convert_annotations, indent=4)
        fp.write(out)


