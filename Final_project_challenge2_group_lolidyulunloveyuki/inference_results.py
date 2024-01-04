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

from config.config import config, update_config
from utils import exp_utils
from evaluation import eval_utils
from evaluation.task_inference_results import Task
from model.corr_clip_spatial_transformer2_anchor_2heads_hnm import ClipMatcher


class WorkerWithDevice(mp.Process):
    def __init__(self, config, task_queue, results_queue, worker_id, device_id):
        self.config = config
        self.device_id = device_id
        self.worker_id = worker_id
        super().__init__(target=self.work, args=(task_queue, results_queue))

    def work(self, task_queue, results_queue):

        device = torch.device(f"cuda:{self.device_id}")

        while True:
            try:
                task = task_queue.get(timeout=1.0)
            except QueueEmpty:
                break
            key_name = task.run(self.config, device)
            results_queue.put(key_name)
            del task


def get_results(annotations, config, args):
    num_gpus = torch.cuda.device_count()
    mp.set_start_method("forkserver")

    task_queue = mp.Queue()
    for _, annots in annotations.items():
        task = Task(config, annots, args)
        task_queue.put(task)
    # Results will be stored in this queue
    results_queue = mp.Queue()

    num_processes = 30 #num_gpus

    pbar = tqdm.tqdm(
        desc=f"Get RT results",
        position=0,
        total=len(annotations),
    )

    workers = [
        WorkerWithDevice(config, task_queue, results_queue, i, i % num_gpus)
        for i in range(num_processes)
    ]
    # Start workers
    for worker in workers:
        worker.start()
    # Update progress bar
    predicted_rts = {}
    n_completed = 0
    while n_completed < len(annotations):
        pred = results_queue.get()
        predicted_rts.update(pred)
        n_completed += 1
        pbar.update()
    # Wait for workers to finish
    for worker in workers:
        worker.join()
    pbar.close()
    return predicted_rts


def format_predictions(annotations, predicted_rts):
    # Format predictions
    predictions= {}
    for clip_n, clip_ann in annotations.items():
        clip_predictions = {}
        apred = []
        for a in clip_ann["annotations"]:
            qt = {}
            qidt = {}
            auid = a["annotation_uid"]
            for qid in a["query_sets"].keys(): # qid = num

                    if (auid, qid) in predicted_rts:
                        rt_pred = predicted_rts[(auid, qid)][0].to_json()
                        qidt[qid] = rt_pred
                    else:
                        qidt[qid] = {"bboxes": [], "score": 0.0}
            qt["query_sets"] = qidt
            apred.append(qt)
        clip_predictions["predictions"] = apred
        predictions[clip_n] = clip_predictions
    return predictions


def parse_args():
    parser = argparse.ArgumentParser(description='Train hand reconstruction network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        #"--eval", dest="eval", action="store_true",help="evaluate model")
        "--eval", type=bool, default=True)
    parser.add_argument(
        "--debug", dest="debug", action="store_true",help="evaluate model")
    parser.add_argument(
        "--gt-fg", dest="gt_fg", action="store_true",help="evaluate model")
    
    parser.add_argument(
        "--clip_path", type = str, default=sys.argv[1], help = "Path to the clip directory")  
    
    parser.add_argument(
        "--anno_fn", type = str, default=sys.argv[2], help = "Path to the unannotated test json file")        
    
    parser.add_argument(
        "--output_fn", type = str, default=sys.argv[3], help = "Path to the output json file") 
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


if __name__ == '__main__':
    args = parse_args()
    logger, output_dir, tb_log_dir = exp_utils.create_logger(config, args.cfg, phase='train')
    mode = 'eval' if args.eval else 'val'
    config.inference_cache_path = os.path.join(output_dir, 'inference_cache_test')
    print(config.inference_cache_path)
    os.makedirs(config.inference_cache_path, exist_ok=True)
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # set random seeds
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    mode = 'test_unannotated' if args.eval else 'val'
    if mode == 'val':
        annotation_path = os.path.join('/home/remote/mplin/DLCV/VQLoC/new_data/data/', 'vq_{}.json'.format(mode))
    else:
        #annotation_path = '/home/remote/mplin/DLCV/VQLoC/new_data/data/vq_test_unannotated.json'
        annotation_path = args.anno_fn  

    with open(annotation_path) as fp:
        annotations = json.load(fp)
    clipwise_annotations_list = eval_utils.convert_annotations_to_clipwise_list(annotations)

    if args.debug:
        clips_list = list(clipwise_annotations_list.keys())
        clips_list = sorted([c for c in clips_list if c is not None])
        clips_list = clips_list[: 20]
        clipwise_annotations_list = {
            k: clipwise_annotations_list[k] for k in clips_list
        }
    predictions_rt = get_results(clipwise_annotations_list, config, args)
    predictions = format_predictions(annotations, predictions_rt)
    with open(args.output_fn, 'w') as fp:
        out = json.dumps(predictions, indent=4)
        fp.write(out)