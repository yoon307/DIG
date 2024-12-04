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

from evaluation_coco import eval_in_script_dl

#######################################################
# CUDA_VISIBLE_DEVICES=0,1,2,4 python train_dl.py --name 1017_train_dl_exp1 --gt_path /mnt/shyoon4tb/RIB/result_coco/DIFF/sem_seg2 --vis --dict
#######################################################

if __name__ == '__main__':

    categories= ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
                'bus', 'train', 'truck', 'boat', 'traffic_light',
                'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird',
                'cat', 'dog', 'horise', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
                'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket', 'bottle',
                'wine_glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted_plant', 'bed',
                'dining_table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell_phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']
    
    categories_dl = categories.copy()
    categories_dl.insert(0,'background')

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--train_list", default="coco14/train14_ysh.txt", type=str) ####BE CAREFUL
    # parser.add_argument("--val_list", default="coco14/val14_small.txt", type=str) ####BE CAREFUL
    parser.add_argument("--val_list", default="coco14/val14_tiny.txt", type=str) ####BE CAREFUL
    parser.add_argument("--gt_path", required=True, type=str)

    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--batch_size", default=8, type=int)

    # Learning rate
    parser.add_argument("--lr", default=0.0005, type=float)
    # parser.add_argument("--lr", default=0.0001, type=float) #Restart
    parser.add_argument("--wt_dec", default=1e-5, type=float)
    parser.add_argument("--max_epoches", default=30, type=int)

    # Experiments
    parser.add_argument("--model", default='models.model_dl_coco', type=str)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--gpu", default=-1, type=int)

    # Model related hyper-parameters
    parser.add_argument("--sharing_position", default=0, type=int)
    parser.add_argument("--loss_weights", default=[1, 1, 1, 1], nargs='+', type=float)
    parser.add_argument("--prob", default=0.50, type=float)
    parser.add_argument("--cl_loop", default=1, type=int)

    # Output
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--dict", default=True)
    parser.add_argument("--crf", action='store_true')
    parser.add_argument("--print_freq", default=300, type=int)
    parser.add_argument("--out_num", default=100, type=int)
    parser.add_argument("--alphas", default=[6, 10, 24], nargs='+', type=int)

    args = parser.parse_args()
    print(args)

    torch.manual_seed(5123)
    np.random.seed(5123)
    random.seed(5123)
    

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

    
    # train_dataset = utils.build_dataset_coco_dl(phase='train', path="coco14/train14.txt", gt_path=args.gt_path)
    # val_dataset = utils.build_dataset_coco_dl(phase='val', path="coco14/val14_small.txt", gt_path='.data/COCO2014/SegmentationClass/')
    train_dataset = utils.build_dataset_coco_dl(phase='train', path=args.train_list, gt_path=args.gt_path)
    val_dataset = utils.build_dataset_coco_dl(phase='val', path=args.val_list, gt_path='./data/COCO2014/SegmentationClass/')

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,num_workers=8,
                                   shuffle=True,pin_memory=True, drop_last=True)

    val_data_loader = DataLoader(val_dataset, num_workers=8, shuffle=False,  pin_memory=True)

    logger.info('Train dataset is loaded from ' + args.train_list)
    logger.info('Validation dataset is loaded from ' + args.val_list)



    global_iter = 0

    train_num_img = len(train_dataset)
    train_num_batch = len(train_data_loader)
    max_step = train_num_img // args.batch_size * args.max_epoches
    args.max_step = max_step


    model = getattr(importlib.import_module(args.model), 'model_WSSS')(args)
    model.train_setup()

    print('#' * 111)
    print(('#' * 43) + ' Start magnet train loop ' + ('#' * 43))
    miou_list = []
    max_miou = 0


    for epo in range(args.max_epoches):

        # Train
        print('#' * 111)
        print('Epoch ' + str(epo + 1).zfill(3) + ' train')
        model.set_phase('train')

        for iter, pack in enumerate(tqdm(train_data_loader)):

            model.unpack(pack)
            model.update(epo)

            if iter % args.print_freq == 0 and iter != 0:
                model.print_log(epo + 1, iter / train_num_batch)
                # model.visualize_cam(epo + 1, iter // args.print_freq, train_path)

            global_iter += 1


            if epo>=0 and global_iter % 5000 ==0:
                print("saving model at epo %d"%epo)
                # model.save_model(epo, global_iter, ckpt_path)
                model.save_model(epo, global_iter, ckpt_path)

            if (epo>=20 and global_iter % 5000 ==0) or global_iter == 5000:
                # Validation
                print('#' * 111)
                print('Epoch ' + str(epo + 1).zfill(3) + ' validation')
                print('Iteration %06d'%global_iter)

                model.set_phase('eval')

                # # model.infer_multi_init()
                for iter_val, pack_val in enumerate(tqdm(val_data_loader)):
                    model.unpack(pack_val)
                    
                    model.infer_single(epo + 1, val_path, dict_path, crf_path, vis=args.vis, dict=args.dict, crf=args.crf)
                
                loglist = eval_in_script_dl(logger=logger, eval_list='val14_tiny', name=args.name, dict_dir='./dict')
                for i in range(81):
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
                    model.save_model(epo, global_iter, ckpt_path, best=True)
                
                # os.system('python evaluation_coco.py --name ' + args.name + ' --task png --dict_dir dict --list val14_small --gt_dir ./coco/SegmentationClass/val2014')


                model.set_phase('train')