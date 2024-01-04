import os
import pprint
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import itertools
import argparse
import warnings

import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast

from config.config import config, update_config

from model.corr_clip_spatial_transformer2_anchor_2heads_hnm import ClipMatcher
from utils import exp_utils, train_utils, dist_utils
from dataset import dataset_utils
from func.train_anchor import train_epoch, validate
#from train import train_epoch, validate

import transformers
import wandb
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Train hand reconstruction network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        "--eval", dest="eval", action="store_true",help="evaluate model")
    parser.add_argument(
        '--local_rank', default=-1, type=int, help='node rank for distributed training')
    
    parser.add_argument(
        "--clip_path", type = str, default=sys.argv[1], help = "Path to the clip directory")  
    
    parser.add_argument(
        "--anno_dir", type = str, default=sys.argv[2], help = "Path to the original json file (including vq_train.json, vq_val.json)")
            
    parser.add_argument(
         "--Use_orig_data", type = str , default='True', help = "Train with original training data")
    
    parser.add_argument(
         "--Use_UFS_3", type = str , default='False', help = "Train with 2 additional visual crops from Unbalanced Frame Sampling") 
    
    parser.add_argument(
         "--Use_UFS_5", type = str , default='False', help = "Train with 4 additional visual crops from Unbalanced Frame Sampling") 
    
    parser.add_argument(
         "--Use_UFS_gt_10", type = str , default='False', help = "Train with additional response track from Unbalanced Frame Sampling (window length = 10)") 
    
    parser.add_argument(
         "--Use_UFS_gt_20", type = str , default='False', help = "Train with additional response track from Unbalanced Frame Sampling (window length = 20)") 
    
    parser.add_argument(
         "--Use_UFS_test", type = str , default='False', help = "Train with pseudo response track from orginal model prediction and 2 additional visual crops from Unbalanced Frame Sampling")         
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def main():
    # Get args and config
    args = parse_args()
    logger, output_dir, tb_log_dir = exp_utils.create_logger(config, args.cfg, phase='train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # set random seeds
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # set device
    gpus = range(torch.cuda.device_count())
    distributed = torch.cuda.device_count() > 1
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if "LOCAL_RANK" in os.environ:
        dist_utils.dist_init(int(os.environ["LOCAL_RANK"]))
        local_rank = dist.get_rank()
    else:
        local_rank = -1
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        wandb_name = config.exp_name
        wandb_proj_name = config.exp_group
        wandb_run = wandb.init(project=wandb_proj_name, group=wandb_name)#, name='smooth-puddle-94', resume=True)
        wandb.config.update({
            "exp_name": config.exp_name,
            "batch_size": config.train.batch_size,
            "total_iteration": config.train.total_iteration,
            "lr": config.train.lr,
            "weight_decay": config.train.weight_decay,
            "loss_weight_bbox_giou": config.loss.weight_bbox_giou,
            "loss_prob_bce_weight": config.loss.prob_bce_weight,
            "model_num_transformer": config.model.num_transformer,
            "model_resolution_transformer": config.model.resolution_transformer,
            "model_window_transformer": config.model.window_transformer,
        })
    else:
        wandb_run = None

    # get model
    model = ClipMatcher(config).to(device)

    # get optimizer
    optimizer = train_utils.get_optimizer(config, model)
    schedular = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=config.train.schedular_warmup_iter,
                                                             num_training_steps=config.train.total_iteration)
    scaler = torch.cuda.amp.GradScaler()

    best_iou, best_prob = 0.0, 0.0
    ep_resume = None
    if config.train.resume:
        try:
            model, optimizer, schedular, scaler, ep_resume, best_iou, best_prob = train_utils.resume_training(
                                                                                model, optimizer, schedular, scaler, 
                                                                                output_dir,
                                                                                cpt_name='/home/remote/mplin/DLCV/VQLoC/orig_ckpt/cpt_best_probDDD.pth.tar')
            print('LR after resume {}'.format(optimizer.param_groups[0]['lr']))
        except:
            print('Resume failed')
    # distributed training
    ddp = False


    # get dataset and dataloader    
    train_data = dataset_utils.get_dataset(args, config, split='train')
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.train.batch_size, 
                                               shuffle=True,
                                               num_workers=int(config.workers), 
                                               pin_memory=True, 
                                               drop_last=True)#,
                                               #sampler=train_sampler)
    val_data = dataset_utils.get_dataset(args, config, split='val')
    val_loader = torch.utils.data.DataLoader(val_data,
                                               batch_size=config.test.batch_size, 
                                               shuffle=False,
                                               num_workers=int(config.workers), 
                                               pin_memory=True, 
                                               drop_last=False)    
 
    start_ep = ep_resume if ep_resume is not None else 0
    end_ep = 100000000 #int(config.train.total_iteration / len(train_loader)) + 1
    
    ckpt_fn = './cpt_best_prob.pth.tar' 
    if device is not None:
        checkpoint = torch.load(ckpt_fn, map_location=device)
    else:
        checkpoint = torch.load(ckpt_fn, map_location=torch.device('cpu'))
        
    # load model
    if "module" in list(checkpoint["state_dict"].keys())[0]:
        state_dict = {key.replace('module.',''): item for key, item in checkpoint["state_dict"].items()}
    else:
        state_dict = checkpoint["state_dict"]
    missing_states = set(model.state_dict().keys()) - set(state_dict.keys())
    if len(missing_states) > 0:
        warnings.warn("Missing keys ! : {}".format(missing_states))
    strict = True
    model.load_state_dict(state_dict, strict=strict)    

    # train    
    for epoch in range(start_ep, end_ep):
        iou, prob = validate(config,
                                loader=val_loader,
                                model=model,
                                epoch=epoch,
                                output_dir=output_dir,
                                device=device,
                                rank=local_rank,
                                ddp=ddp,
                                 wandb_run=wandb_run
                                 )
        logger.info('Rank {}, best iou: {} (current {}), best probability accuracy: {} (current {})'.format(local_rank, best_iou, iou, best_prob, prob))

        train_epoch(config,
                    loader=train_loader,
                    model=model,
                    optimizer=optimizer,
                    schedular=schedular,
                    scaler=scaler,
                    epoch=epoch,
                    output_dir=output_dir,
                    device=device,
                    rank=local_rank,
                    ddp=ddp,
                    wandb_run=wandb_run
                    )
        torch.cuda.empty_cache()

        if epoch % 1 == 0:
            print('Doing validation...')
            iou, prob = validate(config,
                                loader=val_loader,
                                model=model,
                                epoch=epoch,
                                output_dir=output_dir,
                                device=device,
                                rank=local_rank,
                                ddp=ddp,
                                 wandb_run=wandb_run
                                 )
            torch.cuda.empty_cache()
            if iou > best_iou:
                best_iou = iou

            if prob > best_prob:
                best_prob = prob

            train_utils.save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'schedular': schedular.state_dict(),
                'scaler': scaler.state_dict(),
                'best_prob': best_prob,
                'best_iou': best_iou
            }, 
            checkpoint = './train_ckpt/cpt_train_ep{}.pth.tar'.format(epoch+1))

            logger.info('Rank {}, best iou: {} (current {}), best probability accuracy: {} (current {})'.format(local_rank, best_iou, iou, best_prob, prob))
        #dist.barrier()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    main()