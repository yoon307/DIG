from cProfile import label
from http.client import NON_AUTHORITATIVE_INFORMATION
import os, time
from statistics import mode
import os.path as osp
import argparse
import glob
import random
import pdb
from turtle import pos

import numpy as np
from numpy.core.fromnumeric import size
# from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools

# Image tools
import cv2
import PIL
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from torchvision import transforms
import torchvision

import voc12.data
from tools import utils, pyutils, trmutils
from tools.imutils import save_img, denorm, _crf_with_alpha, cam_on_image
# import tools.visualizer as visualizer
from networks import mctformer

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler

# import resnet38d
# from networks import resnet38d, vgg16d
# from networks import resnet101

import sys
sys.path.append("..") 
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from kmeans_pytorch import kmeans

import shutil

def set_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class RepulsionLoss(torch.nn.Module):
    def __init__(self, strength=0.1, radius=2):
        super().__init__()
        self.strength = strength
        self.radius = radius

    def forward(self, x):
        differences = x.unsqueeze(-1) - x.unsqueeze(-2) #B C C C
        distances = differences.abs().sum(dim=1) # B C C
        repulsion_weights = (distances < self.radius).float() * self.strength
        repulsion_offsets = differences * repulsion_weights.unsqueeze(-1)
        loss = repulsion_offsets.sum(dim=-2).norm(p=2, dim=-1).mean()
        return loss

##########
# CUDA_VISIBLE_DEVICES=0,1 python train_trm.py --name diff --model diffusion_ECCV24_DiG 

##########

class model_WSSS():

    def __init__(self, args, logger):

        self.args = args
        self.categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                           'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person',
                           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        # Common things
        self.phase = 'train'
        self.dev = 'cuda'
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_wosig = nn.BCELoss()
        self.bs = args.batch_size
        self.logger = logger
        self.writer = args.writer

        # Model attributes
        self.net_names = ['net_trm']
        self.base_names = ['cls','pred_diff','cam_consistency','cls_diff','rcams_consistency']
        self.loss_names = ['loss_' + bn for bn in self.base_names]
        self.acc_names = ['acc_' + bn for bn in self.base_names]

        self.nets = []
        self.opts = []
        # self.vis = visualizer.Visualizer(args.visport, self.loss_names, self.acc_names)

        # Evaluation-related
        self.running_loss = [0] * len(self.loss_names)
        self.right_count = [0] * len(self.acc_names)
        self.wrong_count = [0] * len(self.acc_names)
        self.accs = [0] * len(self.acc_names)
        self.count = 0
        self.num_count = 0
        
        #Tensorboard
        self.global_step = 0

        self.val_wrong = 0
        self.val_right = 0

        # Define networks
        self.net_trm = create_model(
            'deit_small_MCTformerV2_DiG',
            pretrained=False,
            num_classes=args.C,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None
        )

        self.unet = Unet(
                        dim = 64,
                        dim_mults = (1, 2, 4, 8),
                        flash_attn = True
                        ).cuda()

        self.diffusion = GaussianDiffusion(
            self.unet,
            image_size = 224,
            timesteps = 1000,           # number of steps
            # sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
            sampling_timesteps = 150    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        ).cuda().eval()

        unet_state_dict = torch.load('/mnt/shyoon4tb/denoising-diffusion-pytorch/results/model-140-pascal.pt',map_location="cuda")
        # unet_state_dict = torch.load('/mnt/shyoon4tb/denoising-diffusion-pytorch/results/model-100.pt',map_location="cuda")
        self.diffusion.load_state_dict(unet_state_dict['model'])
        self.diffusion.model.training = False

        if args.finetune:
           
            checkpoint = torch.load("./pretrained/deit_small_imagenet.pth", map_location='cpu')

            try:
                checkpoint_model = checkpoint['model']
            except:
                checkpoint_model = checkpoint
            state_dict = self.net_trm.state_dict()

            if 'head.bias' in state_dict.keys():
                for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]
            else:
                for k in ['head.weight', 'head_dist.weight', 'head_dist.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]

            # interpolate position embedding
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = self.net_trm.patch_embed.num_patches
            if args.finetune.startswith('https'):
                num_extra_tokens = 1
            else:
                num_extra_tokens = self.net_trm.pos_embed.shape[-2] - num_patches

            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)

            new_size = int(num_patches ** 0.5)

            if args.finetune.startswith('https') and 'MCTformer' in args.trm:
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens].repeat(1,args.C,1)
            else:
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]

            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

            if args.finetune.startswith('https') and 'MCTformer' in args.trm:
                cls_token_checkpoint = checkpoint_model['cls_token']
                new_cls_token = cls_token_checkpoint.repeat(1,args.C,1)
                
                checkpoint_model['cls_token'] = new_cls_token

            self.net_trm.load_state_dict(checkpoint_model, strict=False)

        self.L2 = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.KD = nn.KLDivLoss(reduction='batchmean')
        self.KD_log = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.KD_none = nn.KLDivLoss(reduction='none')
        
    

    # Save networks
    def save_model(self, epo, ckpt_path):
        epo_str = str(epo).zfill(3)
        torch.save(self.net_trm.module.state_dict(), ckpt_path + '/' + epo_str + 'net_trm.pth')

    # Load networks
    def load_model(self, epo, ckpt_path):
        epo_str = str(epo).zfill(3)
        self.net_trm.load_state_dict(torch.load(ckpt_path + '/' + epo_str + 'net_trm.pth'), strict=True)

        self.net_trm = torch.nn.DataParallel(self.net_trm.to(self.dev))

    # Set networks' phase (train/eval)
    def set_phase(self, phase):

        if phase == 'train':
            self.phase = 'train'
            for name in self.net_names:
                getattr(self, name).train()
            self.logger.info('Phase : train')

        else:
            self.phase = 'eval'
            for name in self.net_names:
                getattr(self, name).eval()
            self.logger.info('Phase : eval')

    # Set optimizers and upload networks on multi-gpu
    def train_setup(self):

        args = self.args


        linear_scaled_lr = args.lr * args.batch_size * trmutils.get_world_size() / 512.0
        args.lr = linear_scaled_lr

        self.opt_trm = create_optimizer(args, self.net_trm)
        self.lr_scheduler, _ = create_scheduler(args, self.opt_trm)


        self.logger.info('Poly-optimizer for trm is defined.')

        self.net_trm = torch.nn.DataParallel(self.net_trm.to(self.dev))
        self.logger.info('Networks are uploaded on multi-gpu.')

        self.nets.append(self.net_trm)

    # Unpack data pack from data_loader
    def unpack(self, pack):

        if self.phase == 'train':
            self.img = pack[0].to(self.dev)  # B x 3 x H x W
            self.img_diff = pack[1].to(self.dev)  # B x 3 x H x W
            self.label = pack[2].to(self.dev)  # B x 20
            self.name = pack[3]  # list of image names

        if self.phase == 'eval':
            self.img = pack[0]
            self.img_diff = pack[1]
            self.label = pack[2].to(self.dev)
            self.name = pack[3][0]

        self.split_label()
    

    # Do forward/backward propagation and call optimizer to update the networks
    def update(self, epo):
        # Tensor dimensions
        B = self.img.shape[0]
        H = self.img.shape[2]
        W = self.img.shape[3]
        C = 20  # Number of cls

        lfca_epoch = 9

        self.B = B
        self.C = C

        # self.img_norm = F.interpolate(self.denormforDiff(self.img),size=(128,128),mode='bicubic',align_corners=True)
        # self.img_norm = F.interpolate(self.denormforDiff(self.img),size=(192,192),mode='bicubic',align_corners=True)
        # self.img_norm = F.interpolate(self.denormforDiff(self.img),size=(224,224),mode='bicubic',align_corners=True)
        self.img_norm = self.denormforDiff(self.img)
        
        self.img_norm = self.diffusion.normalize(self.img_norm) # -1 ~ +1

        #################diffusion process###############
        SFI = 100 #Step for image

        feat_diff_list = []
        if (self.args.W[1]>0 or self.args.W[2]>0) or self.args.W[3]>0:
            with torch.no_grad():
                # for j in [200,210,220,230,240,60]:  # GPU 0/5
                # for j in [0,10,20,30,40,60]: 
                # for j in [0,10,20,30,40,60]: 
                for j in [0,1,2,3,4,SFI]: 
                # for j in [SFI]: 
                # for j in [0,10,20,60]: 
                    slice = B
                    step_a = j

                    t = torch.randint(step_a, step_a+1, (slice,), device="cuda").long()

                    x_t, v, feat_diff = self.diffusion.p_losses(self.img_norm,t)       # sample noise, Unet output(down,up) (predict noise), Unet feature(down)

                    if j == SFI :
                        img_out = self.diffusion.predict_start_from_v(x_t,t,v)
                        img_out = self.diffusion.unnormalize(img_out)
                        
                        H,W = self.img.size()[2:]
                        # img_out = F.interpolate(img_out,size=(H,W),mode='bicubic',align_corners=True)
                        img_diff = self.normforCls(img_out)

                    if j != SFI:
                        # feat_diff0 = feat_diff[0] #512/512/256/128/64
                        # feat_diff = feat_diff[1:]
                        # feat_diff0 = torch.cat(feat_diff[:2],dim=1) #1024
                        feat_diff0 = torch.cat(feat_diff,dim=1) #1024
                        # feat_diff0 = F.interpolate(feat_diff0,size= (H//16,W//16) , mode='bilinear',align_corners= True)
                        feat_diff_list.append(feat_diff0)
        else:        
            feat_diff_list = None
 
        ################################################### Update TRM ###################################################
        self.opt_trm.zero_grad()
        self.net_trm.train()
        
        # feat_diff_list = None

        # if epo<=lfca_epoch:
        #     outputs = self.net_trm(self.img, feat_diff_list, detach=True)
        # else:    
        outputs = self.net_trm(self.img, feat_diff_list, detach=True)

        self.out = outputs['cls']
        self.out_patch = outputs['pcls']
        cams = outputs['cams']
        attn_mct = outputs['mtatt']
        patch_attn = outputs['attn']
        rcams = outputs['rcams']

        self.feat = outputs['feat']
        if feat_diff_list is not None:
            # self.cattn_diff = outputs['cattn_diff']
            self.cam_diff = outputs['cam_diff']
            self.diff_cls = outputs['diff_cls']
            self.diff_pcls = outputs['diff_pcls']           # B C

        # outputs_aug = self.net_trm(self.img_diff)
        # self.out_aug = outputs_aug['cls']
        # self.out_patch_aug = outputs_aug['pcls']

        self.loss_cls = self.args.W[0]*(
            F.multilabel_soft_margin_loss(self.out,self.label)
            + F.multilabel_soft_margin_loss(self.out_patch,self.label)
            # + F.multilabel_soft_margin_loss(self.out_aug,self.label)
            # + F.multilabel_soft_margin_loss(self.out_patch_aug,self.label)
            )
        loss_trm = self.loss_cls 

        T = self.args.T
        alpha = self.args.A

        if self.args.W[1]>0:
            S_pred_patch = F.softmax(self.diff_pcls/T, dim=1)
            S_pred_ctk = F.softmax(self.diff_cls/T, dim=1)
            T_pred_patch = F.softmax(self.out_patch.detach()/T, dim=1)
            T_pred_ctk = F.softmax(self.out.detach()/T, dim=1)

            S_pred_patch_cat = torch.log(torch.cat((S_pred_patch, 1 - S_pred_patch), dim=1))
            S_pred_ctk_cat = torch.log(torch.cat((S_pred_ctk, 1 - S_pred_ctk), dim=1))
            T_pred_patch_cat = torch.log(torch.cat((T_pred_patch, 1 - T_pred_patch), dim=1))
            T_pred_ctk_cat = torch.log(torch.cat((T_pred_ctk, 1 - T_pred_ctk), dim=1))
            
            # S_pred_patch_cat = torch.log_softmax(torch.cat((S_pred_patch, 1 - S_pred_patch), dim=1),dim=1)
            # S_pred_ctk_cat = torch.log_softmax(torch.cat((S_pred_ctk, 1 - S_pred_ctk), dim=1),dim=1)
            # T_pred_patch_cat = torch.log_softmax(torch.cat((T_pred_patch, 1 - T_pred_patch), dim=1),dim=1)
            # T_pred_ctk_cat = torch.log_softmax(torch.cat((T_pred_ctk, 1 - T_pred_ctk), dim=1),dim=1)

            self.loss_pred_diff = self.args.W[1]*(
                alpha * F.multilabel_soft_margin_loss(self.diff_cls,self.label)
                + alpha * F.multilabel_soft_margin_loss(self.diff_pcls,self.label)     # original
                + (1-alpha) * (T*T) * self.KD_log(S_pred_ctk_cat, T_pred_ctk_cat) * 0.5
                + (1-alpha) * (T*T) * self.KD_log(S_pred_patch_cat, T_pred_patch_cat) * 0.5
            )
            loss_trm += self.loss_pred_diff
        else:
            self.loss_pred_diff = torch.Tensor([0])

        if self.args.W[2]>0 and epo>lfca_epoch:
            _cams = self.max_norm(F.interpolate(cams,size=self.cam_diff.size()[2:],mode='bilinear',align_corners=False))
            
            _cam_diff = self.max_norm(self.cam_diff)*self.label.view(B,C,1,1)
            # _cam_diff = (self.max_norm(self.cam_diff*self.cattn_diff))*self.label.view(B,C,1,1)

            ###############################################
            if True:
                input = denorm(self.img[0]).permute(1,2,0)
                cam_diff = F.interpolate(self.cam_diff,size=(H,W),mode='bilinear',align_corners=False)*self.label.view(B,C,1,1)
                # cattn_diff = F.interpolate(self.cattn_diff,size=(H,W),mode='bilinear',align_corners=False)*self.label.view(B,C,1,1)
                norm_cam = self.max_norm(cam_diff)
                # norm_cam = self.max_norm(cattn_diff)

                gt = self.label[0].cpu().detach().numpy()
                self.gt_cls = np.nonzero(gt)[0]

                for c in self.gt_cls:
                    plt.imshow(input.cpu().detach().numpy())
                    plt.imshow(norm_cam[0][c].cpu().detach().numpy(), cmap='jet', alpha=0.4)
                    plt.savefig("./%s/%s%d.png"%(self.args.val_path,self.categories[c],c))
                    plt.close()
            ################################################
        
            self.loss_cam_consistency = self.args.W[2]*(
                # ((_cam_diff.detach()-_cams)*self.label.view(B,C,1,1)).abs().mean()
                ((_cam_diff-_cams)*self.label.view(B,C,1,1)).abs().mean()
            )
            # if epo>lfca_epoch:
            loss_trm += self.loss_cam_consistency
        else:
            self.loss_cam_consistency = torch.Tensor([0])

        ##############################################
        if (self.args.W[3]>0 or self.args.W[4]>0):

            outputs_diff = self.net_trm(img_diff.detach())
            self.out_diff = outputs_diff['cls']
            self.out_patch_diff = outputs_diff['pcls']
            cams_diff = outputs_diff['cams']
            attn_mct_diff = outputs_diff['mtatt']
            patch_attn_diff = outputs_diff['attn']
            rcams_diff = outputs_diff['rcams']

            self.loss_cls_diff = self.args.W[3]*(
                F.multilabel_soft_margin_loss(self.out_diff,self.label)
                +F.multilabel_soft_margin_loss(self.out_patch_diff,self.label)
                )
            loss_trm += self.loss_cls_diff
        else: 
            self.loss_cls_diff = torch.Tensor([0])

        if self.args.W[4]>0 and epo>lfca_epoch:
            self.loss_rcams_consistency = self.args.W[4]*(
                ((self.max_norm(rcams).detach()-self.max_norm(rcams_diff))*self.label.view(B,C,1,1)).abs().mean()
                # (patch_attn.detach() - patch_attn_diff).abs().mean()
            )
            loss_trm += self.loss_rcams_consistency
        else: 
            self.loss_rcams_consistency = torch.Tensor([0])


        loss_trm.backward()
        self.opt_trm.step()

        ##############################################
        torch.cuda.empty_cache()
        
        ################################################### Export ###################################################


        for i in range(len(self.loss_names)):
            self.running_loss[i] += getattr(self, self.loss_names[i]).item()

        self.count += 1
        #Tensorboard
        self.global_step +=1


        # self.count_rw(self.label, self.out, 2)
        if feat_diff_list is not None:
            self.count_rw(self.label, self.diff_pcls, 1)
            self.count_rw(self.label, self.diff_cls, 3)
        self.count_rw(self.label, self.out_patch, 0)
    
       
    # Initialization for msf-infer
    def infer_init(self,epo):
        n_gpus = torch.cuda.device_count()
        self.net_trm.eval()
        # self.net_trm_replicas = torch.nn.parallel.replicate(self.net_trm.module, list(range(n_gpus)))

    # (Multi-Thread) Infer MSF-CAM and save image/cam_dict/crf_dict
    def infer_multi(self, epo, val_path, dict_path, crf_path, vis=False, dict=False, crf=False):

        if self.phase != 'eval':
            self.set_phase('eval')

        epo_str = str(epo).zfill(3)
        gt = self.label[0].cpu().detach().numpy()
        self.gt_cls = np.nonzero(gt)[0]

        
        B, _, H, W = self.img.shape
        n_gpus = torch.cuda.device_count()


        cam = self.net_trm.module.forward(self.img.cuda(),return_att=True,n_layers= 12)
        cam = F.interpolate(cam,[H,W],mode='bilinear',align_corners=False)*self.label.view(B,20,1,1)
        

        cam_flip = self.net_trm.module.forward(torch.flip(self.img,(3,)).cuda(),return_att=True,n_layers= 12)
        cam_flip = F.interpolate(cam_flip,[H,W],mode='bilinear',align_corners=False)*self.label.view(B,20,1,1)
        cam_flip = torch.flip(cam_flip,(3,))
   
        cam = cam+cam_flip
        norm_cam = self.max_norm(cam)[0].detach().cpu().numpy()
 
        self.cam_dict = {}

        for i in range(20):
            if self.label[0, i] > 1e-5:
                self.cam_dict[i] = norm_cam[i]

        if vis:
            # img_np = denorm(self.img[0]).cpu().detach().data.permute(1, 2, 0).numpy()
            # for c in self.gt_cls:
            #     save_img(osp.join(val_path, epo_str + '_' + self.name + '_cam_' + self.categories[c] + '.png'), img_np,
            #              norm_cam[c])
            
            input = denorm(self.img[0])
            for c in self.gt_cls:
                temp = cam_on_image(input.cpu().detach().numpy(), norm_cam[c])
                self.writer.add_image(self.name+'/'+self.categories[c], temp, epo)

        if dict:
            np.save(osp.join(dict_path, self.name + '.npy'), self.cam_dict)

        if crf:
            for a in self.args.alphas:
                crf_dict = _crf_with_alpha(self.cam_dict, self.name, alpha=a)
                np.save(osp.join(crf_path, str(a).zfill(2), self.name + '.npy'), crf_dict)
    
    def denormforDiff(self,img):
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        mean_tensor = torch.tensor(imagenet_mean).reshape(1, 3, 1, 1).cuda()
        std_tensor = torch.tensor(imagenet_std).reshape(1, 3, 1, 1).cuda()

        denorm_img = img * std_tensor + mean_tensor

        return denorm_img
    
    def normforCls(self,img):
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        mean_tensor = torch.tensor(imagenet_mean).reshape(1, 3, 1, 1).cuda()
        std_tensor = torch.tensor(imagenet_std).reshape(1, 3, 1, 1).cuda()

        denorm_img = (img - mean_tensor) / std_tensor

        return denorm_img


    # Print loss/accuracy (and re-initialize them)
    def print_log(self, epo, iter):

        loss_str = ''
        acc_str = ''

        for i in range(len(self.loss_names)):
            loss_str += self.loss_names[i] + ' : ' + str(round(self.running_loss[i] / self.count, 5)) + ', '

        for i in range(len(self.acc_names)):
            if self.right_count[i] != 0:
                acc = 100 * self.right_count[i] / (self.right_count[i] + self.wrong_count[i])
                acc_str += self.acc_names[i] + ' : ' + str(round(acc, 2)) + ', '
                self.accs[i] = acc

        self.logger.info(loss_str[:-2])
        self.logger.info(acc_str[:-2])

        ###Tensorboard###
        for i in range(len(self.loss_names)):
            self.writer.add_scalar("Loss/%s"%self.loss_names[i],(self.running_loss[i] / self.count),self.global_step)

        self.running_loss = [0] * len(self.loss_names)
        self.right_count = [0] * len(self.acc_names)
        self.wrong_count = [0] * len(self.acc_names)
        self.count = 0

    def count_rw(self, label, out, idx):
        for b in range(out.size(0)):  # 8
            gt = label[b].cpu().detach().numpy()
            gt_cls = np.nonzero(gt)[0]
            num = len(np.nonzero(gt)[0])
            pred = out[b].cpu().detach().numpy()
            pred_cls = pred.argsort()[-num:][::-1]

            for c in gt_cls:
                if c in pred_cls:
                    self.right_count[idx] += 1
                else:
                    self.wrong_count[idx] += 1

    @torch.no_grad()
    def mvweight(self):
        for param_main, param_sup in zip(self.net_main.parameters(), self.net_sup.parameters()):
            # param_sup.data = self.M * param_sup.data + (1 - self.M) * param_main.data
            param_sup.data =  param_main.data

 

    # Max_norm
    def max_norm(self, cam_cp):
        N, C, H, W = cam_cp.size()
        cam_cp = F.relu(cam_cp)
        max_v = torch.max(cam_cp.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        min_v = torch.min(cam_cp.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        cam_cp = F.relu(cam_cp - min_v - 1e-5) / (max_v - min_v + 1e-5)
        return cam_cp
    
    def cam_l1(self, cam1, cam2):
        return torch.mean(torch.abs(cam2.detach() - cam1))

    def split_label(self):

        bs = self.label.shape[0] if self.phase == 'train' else 1  # self.label.shape[0]
        self.label_exist = torch.zeros(bs, 20).cuda()
        # self.label_remain = self.label.clone()
        for i in range(bs):
            label_idx = torch.nonzero(self.label[i], as_tuple=False)
            rand_idx = torch.randint(0, len(label_idx), (1,))
            target = label_idx[rand_idx][0]
            # self.label_remain[i, target] = 0
            self.label_exist[i, target] = 1
        self.label_remain = self.label - self.label_exist

        # self.label_all = self.label  # [:16]
