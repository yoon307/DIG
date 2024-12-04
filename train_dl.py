from __future__ import division
from __future__ import print_function

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

import os, time
import os.path as osp
import argparse
import glob
import random
import pdb
from tqdm import tqdm
import importlib

import numpy as np
import torch

from torch.utils.data import DataLoader

import tools.utils as utils
import tools.pyutils as pyutils
import logging
from evaluation import eval_in_script_dl



#######################################################
#CUDA_VISIBLE_DEVICES=0,1,2,4 python train_dl.py --name 1017_train_dl_exp1 --gt_path ./experiments/dl_data_final/train --vis --dict
#######################################################
   
if __name__ == '__main__':

    categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
                  'bus', 'car', 'cat', 'chair', 'cow', 
                  'diningtable', 'dog', 'horse', 'motorbike', 'person', 
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    
    categories_dl = categories.copy()
    categories_dl.insert(0,'background')

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--gt_path", required=True, type=str)
    
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--batch_size", default=8, type=int)

    # Learning rate
    parser.add_argument("--lr", default=0.0007, type=float)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--max_epoches", default=35, type=int)
    parser.add_argument("--crop", default=321, type=int)
    parser.add_argument("--seed", default=5123, type=int)

    # Experiments
    parser.add_argument("--model", default='models.model_dl', type=str)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--gpu", default=-1, type=int)

    # Model related hyper-parameters
    parser.add_argument("--sharing_position", default=0, type=int)
    parser.add_argument("--loss_weights", default=[1,1,1,1], nargs='+', type=float)
    parser.add_argument("--prob", default=0.50, type=float)
    parser.add_argument("--cl_loop", default=1, type=int)   

    # Output
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--dict", default=True)
    parser.add_argument("--crf", action='store_true')
    parser.add_argument("--print_freq", default=100, type=int)
    parser.add_argument("--out_num", default=100, type=int)
    parser.add_argument("--alphas", default=[6,10,24], nargs='+', type=int)    

    args = parser.parse_args()
   

    torch.manual_seed(args.seed)

    print('Start experiment ' + args.name + '!')
    exp_path, ckpt_path, train_path, val_path, infer_path, dict_path, crf_path, log_path = utils.make_path_with_log(args)
    
    if osp.isfile(log_path):
        os.remove(log_path)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(log_path)
    logger.addHandler(file_handler)

    logger.info(args)

    train_dataset = utils.build_dataset_dl(phase='train', path=args.train_list, gt_path=args.gt_path, crop=args.crop)

    val_dataset = utils.build_dataset_dl(phase='val', path=args.val_list, gt_path='./data/VOC2012/SegmentationClass')

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_data_loader = DataLoader(val_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # val_data_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True)
    
    logger.info('Train dataset is loaded from ' + args.train_list)
    logger.info('Validation dataset is loaded from ' + args.val_list)

    train_num_img = len(train_dataset)
    train_num_batch = len(train_data_loader)
    max_step = train_num_img // args.batch_size * args.max_epoches
    args.max_step = max_step

    model = getattr(importlib.import_module(args.model), 'model_WSSS')(args,logger)
    model.train_setup()

    logger.info('#'*111)
    logger.info(('#'*43)+' Start magnet train loop '+('#'*43))

    miou_list = []
    max_miou = 0

    for epo in range(args.max_epoches):       

        # Train
        print('#'*111)
        print('Epoch ' + str(epo+1).zfill(3) + ' train')
        model.set_phase('train')

        for iter, pack in enumerate(tqdm(train_data_loader)):

            model.unpack(pack)
            model.update(epo)

            if iter%args.print_freq==0 and iter!=0:
                model.print_log(epo+1, iter/train_num_batch)
                model.visualize_cam(epo+1, iter//args.print_freq, train_path)

        model.save_model(epo, ckpt_path)

        # # Validation      
        logger.info('#'*111)
        logger.info('Epoch ' + str(epo+1).zfill(3) + ' validation')
        model.set_phase('eval')

        model.infer_multi_init()

        # print_epo = True
        print_epo = ((epo>=10)or (epo%10==0))

        
        for iter, pack in enumerate(tqdm(val_data_loader)):       
            model.unpack(pack)

            if print_epo:
                if iter<50:
                    model.infer_single(epo+1, val_path, dict_path, crf_path, vis=True, dict=args.dict, crf=args.crf)
                else:
                    model.infer_single(epo+1, val_path, dict_path, crf_path, vis=False, dict=args.dict, crf=args.crf)


            else:
                if iter<50:
                    model.infer_single(epo+1, val_path, dict_path, crf_path, vis=True, dict=False, crf=args.crf)
                else:
                    break

        if print_epo:
            loglist = eval_in_script_dl(logger=logger, eval_list='val', name=args.name, dict_dir='./dict')
            for i in range(21):
                if i%2 != 1:
                    logger.info('%11s:%7.3f%%\t'%(categories_dl[i],loglist[categories_dl[i]]))
                else:
                    logger.info('%11s:%7.3f%%'%(categories_dl[i],loglist[categories_dl[i]]))
            logger.info('\n======================================================')
            
            miou_temp = loglist['mIoU']
            miou_temp_str = str(loglist['mIoU'])
            miou_list.append(miou_temp_str)
            logger.info('Epoch %d mIoU %s: '%(epo,miou_temp_str))

            if miou_temp>max_miou:
                max_miou = miou_temp
                logger.info('New record!')
            # os.system('python evaluation.py --name ' + args.name + ' --task dl --dict_dir dict --list val')