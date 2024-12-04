import os, time
import os.path as osp
import argparse
import glob
import random
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools

# Image tools
import PIL
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from torchvision import transforms

# import voc12.data
from tools import utils, pyutils
from tools.imutils import save_img, denorm, voc_palette, _crf_with_alpha, crf_dl
import tools.visualizer as visualizer

# from sklearn.utils.extmath import softmax

# import resnet38d
from networks import resnet38d


def set_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class balCE(nn.Module):
    def __init__(self):
        super(balCE, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=255).cuda()
    def forward(self,logit,label):
        ignore_mask_bg = torch.zeros_like(label)
        ignore_mask_fg = torch.zeros_like(label)
        
        ignore_mask_bg[label == 0] = 1
        ignore_mask_fg[(label != 0) & (label != 255)] = 1
        
        loss_bg = (self.criterion(logit,label) * ignore_mask_bg).sum() / ignore_mask_bg.sum()
        loss_fg = (self.criterion(logit,label) * ignore_mask_fg).sum() / ignore_mask_fg.sum()

        return (loss_bg+loss_fg)/2


class model_WSSS():

    def __init__(self, args):

        self.args = args

        self.categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                           'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person',
                           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        # Common things
        self.phase = 'train'
        self.dev = 'cuda'
        self.ce = nn.CrossEntropyLoss(weight=None, ignore_index=255).cuda()
        self.balCE = balCE()
        self.bs = args.batch_size

        # Model attributes
        self.net_names = ['net_dl']
        self.base_names = ['dl']
        self.loss_names = ['loss_' + bn for bn in self.base_names]
        self.acc_names = ['acc_' + bn for bn in self.base_names]

        self.nets = []
        self.opts = []
        self.vis = None #visualizer.Visualizer(5137, self.loss_names, self.acc_names)

        # Evaluation-related
        self.running_loss = [0] * len(self.loss_names)
        self.right_count = [0] * len(self.acc_names)
        self.wrong_count = [0] * len(self.acc_names)
        self.accs = [0] * len(self.acc_names)
        self.count = 0
        self.IoU = [0] * 21
        self.T = [0] * 21
        self.P = [0] * 21
        self.TP = [0] * 21
        self.IoU_count = [0] * 21

        self.hist = np.zeros((21, 21), dtype=np.float32)
        self.mious = 0
        self.num_count = 0

        # Define networks
        self.net_dl = resnet38d.Net_dl(C=81)


        # Initialize networks with ImageNet pretrained weight
        # resnet38d.convert_mxnet_to_torch('./pretrained/resnet_38d.params')
        
        self.net_dl.load_state_dict(torch.load('./pretrained/cam_coco_res38d.pth'), strict=False)
        # self.net_dl.load_state_dict(torch.load('/mnt/shyoon4tb/denoising-diffusion-pytorch/wsss_trm/experiments/coco_baseline/ckpt/018net_trm.pth',map_location="cpu"), strict=False)
        # self.net_dl.load_state_dict(torch.load('./experiments/CTS_dl_coco/ckpt/020_210000net_dl.pth'), strict=True)

    # Save networks
    def save_model(self, epo, iter, ckpt_path, best=False):
        if best:
            print('best model saved')
            torch.save(self.net_dl.module.state_dict(), ckpt_path + '/best_net_dl.pth')

        epo_str = str(epo).zfill(3)
        iter_str = str(iter).zfill(6)
        torch.save(self.net_dl.module.state_dict(), ckpt_path + '/' + epo_str+'_' + iter_str + 'net_dl.pth')
        # torch.save(self.net_dl.module.state_dict(), ckpt_path + '/' + epo_str+'_' + iter_str + 'net_dl_restart20epo.pth')

    # Load networks
    def load_model(self, epo, iter,ckpt_path):
        epo_str = str(epo).zfill(3)
        iter_str = str(iter).zfill(6)
        self.net_dl.load_state_dict(torch.load(ckpt_path + '/' + epo_str+'_' + iter_str + 'net_dl.pth'), strict=True)
        
        self.net_dl = torch.nn.DataParallel(self.net_dl.to(self.dev))

    # Set networks' phase (train/eval)
    def set_phase(self, phase):

        if phase == 'train':
            self.phase = 'train'
            for name in self.net_names:
                getattr(self, name).train()
            print('Phase : train')

        else:
            self.phase = 'eval'
            for name in self.net_names:
                getattr(self, name).eval()
            print('Phase : eval')

    # Set optimizers and upload networks on multi-gpu
    def train_setup(self):

        args = self.args
        param_dl = self.net_dl.get_parameter_groups()

        self.opt_dl = utils.PolyOptimizer([
            {'params': param_dl[0], 'lr': 1 * args.lr, 'weight_decay': args.wt_dec},
            {'params': param_dl[1], 'lr': 2 * args.lr, 'weight_decay': 0},  # non-scratch bias
            {'params': param_dl[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},  # scratch weight
            {'params': param_dl[3], 'lr': 20 * args.lr, 'weight_decay': 0}  # scratch bias
        ],
            lr=args.lr, weight_decay=args.wt_dec, max_step=args.max_step)

        print('Poly-optimizers are defined.')
        print('Base learning rate : ' + str(args.lr))
        print('non-scratch layer weight lr : ' + str(args.lr))
        print('non-scratch layer bias lr : ' + str(2 * args.lr))
        print('scratch layer weight lr : ' + str(10 * args.lr))
        print('scratch layer bias lr : ' + str(20 * args.lr))
        print('Weight decaying : ' + str(args.wt_dec) + ', max step : ' + str(args.max_step))

        self.net_dl = torch.nn.DataParallel(self.net_dl.to(self.dev))

        print('Networks are uploaded on multi-gpu.')

        self.nets.append(self.net_dl)

    # Unpack data pack from data_loader
    def unpack(self, pack, test_flag=False):

        self.name = pack[0][0]

        if self.phase == 'train':
            self.img = pack[1].to(self.dev)
            self.label = pack[2][:, 0, :, :].long().to(self.dev)

            # self.label[self.label >= 81]=255 ##Check

            # print(np.unique(self.label[0].detach().cpu().numpy()))

        if self.phase == 'eval':
            self.img = pack[1]
            # To handle MSF dataset
            for i in range(10): #########
                self.img[i] = self.img[i].to(self.dev)

            if not test_flag:
                self.label = pack[2][2][:, 0, :, :].long().to(self.dev)

    # Do forward/backward propagation and call optimizer to update the networks
    def update(self, epo):
        
        
        self.opt_dl.zero_grad()

        H,W = self.img.size()[2:]
        H_75, W_75 = int(H * 0.75), int(W * 0.75)
        H_50, W_50 = int(H * 0.50), int(W * 0.50)

        logits = self.net_dl(self.img)

        self.img75 = F.interpolate(self.img, size=(H_75, W_75), mode='bilinear', align_corners=True)
        self.img50 = F.interpolate(self.img, size=(H_50, W_50), mode='bilinear', align_corners=True)

        self.label75 = F.interpolate(self.label.float().unsqueeze(1), size=(H_75, W_75), mode='nearest').squeeze(1).long()
        self.label50 = F.interpolate(self.label.float().unsqueeze(1), size=(H_50, W_50), mode='nearest').squeeze(1).long()

        logits75 = self.net_dl(self.img75)
        logits50 = self.net_dl(self.img50)

        interp = lambda l: F.interpolate(
            l, size=(H, W), mode="bilinear", align_corners=False
        )
        logits_all = [logits] + [interp(l) for l in [logits75,logits50]]
        logits_max = torch.max(torch.stack(logits_all), dim=0)[0]

        self.loss_dl = self.ce(logits, self.label)+self.ce(logits75, self.label75)+self.ce(logits50, self.label50)+self.ce(logits_max, self.label)
        # self.loss_dl = self.balCE(logits, self.label)+self.balCE(logits75, self.label75)+self.balCE(logits50, self.label50)+self.balCE(logits_max, self.label)

        loss = self.loss_dl

        loss.backward()
        self.opt_dl.step()

        
        for i in range(len(self.loss_names)):
            self.running_loss[i] += getattr(self, self.loss_names[i]).item()
        self.count += 1

        # Infer MSF-CAM and save image/cam_dict/crf_dict


    def infer_multi_init(self):
        n_gpus = torch.cuda.device_count()
        self.net_dl_replicas = torch.nn.parallel.replicate(self.net_dl.module, list(range(n_gpus)))

    # (Multi-Thread) Infer MSF-CAM and save image/cam_dict/crf_dict
    def infer_multi(self, epo, val_path, dict_path, crf_path, vis=False, dict=False, crf=False):

        if self.phase != 'eval':
            self.set_phase('eval')

        epo_str = str(epo).zfill(3)
        # label_cls = torch.unique(self.label).cpu().detach().data.numpy()

        _, _, H, W = self.img[4].shape  #########
        n_gpus = torch.cuda.device_count()

        #
        def _work(i, img):
            with torch.no_grad():
                with torch.cuda.device(i % n_gpus):
                    img75 = F.interpolate(img.cuda(),(int(H*0.75),int(W*0.75)),mode='bilinear',align_corners=True)
                    img50 = F.interpolate(img.cuda(),(int(H*0.50),int(W*0.50)),mode='bilinear',align_corners=True)

                    pred = self.net_dl_replicas[i % n_gpus](img.cuda())
                    pred75 = self.net_dl_replicas[i % n_gpus](img75.cuda())
                    pred50 = self.net_dl_replicas[i % n_gpus](img50.cuda())


                    pred_large = F.upsample(pred, (H, W), mode='bilinear', align_corners=False)  # False
                    pred_75 = F.upsample(pred75, (H, W), mode='bilinear', align_corners=False)  # False
                    pred_50 = F.upsample(pred50, (H, W), mode='bilinear', align_corners=False)  # False

                    logits_all = [pred_large,pred_75,pred_50]

                    logits_max = torch.max(torch.stack(logits_all), dim=0)[0]

                    prob =  F.softmax(logits_max, 1) # 1 x 21 x h x w
                    if i % 2 == 1:
                        prob = torch.flip(prob, dims=(3,))
                    return prob

        scale_len = 10 ############

        thread_pool = pyutils.BatchThreader(_work, list(enumerate(self.img)), batch_size=scale_len, prefetch_size=0,
                                            processes=scale_len)

        prob_list = thread_pool.pop_results()

        probs = np.zeros((1, 81, H, W))
        for i in range(scale_len):
            probs += prob_list[i].cpu().detach().data.numpy()

        probs = probs / scale_len

        # probs = torch.from_numpy(probs)
        # self.probs = probs
        # probs = F.softmax(probs,1)
        # probs = probs.numpy()

        # self.pred_dict = {}
        # for c in range(81):
        #     self.pred_dict[c] = probs[0][c]

        preds = np.argmax(probs, axis=1)
        # pdb.set_trace()
        
        if dict:
            cv2.imwrite(osp.join(dict_path,self.name+'.png'),preds[0])


        if crf:
            crf_np = crf_dl(self.name, probs[0])
            predict = np.argmax(crf_np, axis=0).astype(np.uint8)
            cv2.imwrite(osp.join(crf_path, '10', self.name + '.png'), predict)

            # crf_dict = {}
            # for c in range(81):
            #     crf_dict[c] = crf_np[c]
            # np.save(osp.join(crf_path, '10', self.name + '.npy'), crf_dict)

            

            # h, w = list(crf_dict.values())[0].shape
            # tensor = np.zeros((81, h, w), np.float32)
            # for key in crf_dict.keys():
            #     tensor[key] = crf_dict[key]
            # crf_pred = np.argmax(tensor, axis=0).astype(np.uint8)

            # seg_vis = voc_palette(crf_pred)
            # img_np = denorm(self.img[2][0]).cpu().detach().data.permute(1, 2, 0).numpy()
            # save_img(osp.join(val_path, epo_str + '_' + self.name + '_aimg.png'), img_np)
            # save_img(osp.join(val_path, epo_str + '_' + self.name + '_pred.png'), seg_vis)
    
    
    def infer_single(self, epo, val_path, dict_path, crf_path, vis=False, dict=False, crf=False):

        if self.phase!='eval':
            self.set_phase('eval')

        epo_str = str(epo).zfill(3)
        # label_cls = torch.unique(self.label).cpu().detach().data.numpy()

        _, _, H, W = self.img[4].shape ##change
        n_gpus = torch.cuda.device_count()

        _img50 = torch.vstack([self.img[0],self.img[1]]) #2 3 h w
        _img75 = torch.vstack([self.img[2],self.img[3]]) #2 3 h w
        _img100 = torch.vstack([self.img[4],self.img[5]]) #2 3 h w
        _img150 = torch.vstack([self.img[6],self.img[7]]) #2 3 h w
        _img200 = torch.vstack([self.img[8],self.img[9]]) #2 3 h w

        _img_new = [_img50, _img75, _img100, _img150, _img200]

        prob_list = [] 

        with torch.no_grad():
            for i in range(len(_img_new)):
                img = _img_new[i]
                # img75 = F.interpolate(img.cuda(),(int(H*0.75),int(W*0.75)),mode='bilinear',align_corners=True)
                # img50 = F.interpolate(img.cuda(),(int(H*0.50),int(W*0.50)),mode='bilinear',align_corners=True)

                pred = self.net_dl(img.cuda())
                # pred75 = self.net_dl(img75.cuda())
                # pred50 = self.net_dl(img50.cuda())

                pred_large = F.upsample(pred, (H, W), mode='bilinear', align_corners=False)  # False
                # pred_75 = F.upsample(pred75, (H, W), mode='bilinear', align_corners=False)  # False
                # pred_50 = F.upsample(pred50, (H, W), mode='bilinear', align_corners=False)  # False

                # logits_all = [pred_large,pred_75,pred_50]

                # logits_max = torch.max(torch.stack(logits_all), dim=0)[0] #2 C H W

                # pred_large = F.upsample(pred, (H, W), mode='bilinear', align_corners=False) #False
                # prob =  F.softmax(logits_max, 1) # 2 x 21 x h x w
                    
                prob =  F.softmax(pred_large, 1) # 2 x 21 x h x w
                
                prob_ori = prob[0].unsqueeze(0)
                prob_flip = torch.flip(prob[1].unsqueeze(0), dims=(3,))

                prob_final = (prob_ori+prob_flip)/2
                
                prob_list.append(prob_final)

        scale_len = 5 ###change!

        probs = np.zeros((1,81,H,W))
        for i in range(scale_len):
            probs += prob_list[i].cpu().detach().data.numpy()
        
        probs = probs/scale_len

           
        self.pred_dict = {}
        for c in range(81):

            self.pred_dict[c] = probs[0][c]

        preds = np.argmax(probs, axis=1)
        seg_vis = voc_palette(preds[0])
        # print(osp.join(val_path, epo_str + '_' + self.name + '_pred.png'))
        # save_img(osp.join(val_path, epo_str + '_' + self.name + '_pred.png'), seg_vis)        
        if vis:
            palette = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0), (0.0, 0.25, 0.0), (0.0, 0.25, 0.5), (0.0, 0.25, 1.0),
           (0.0, 0.5, 0.0), (0.0, 0.5, 0.25), (0.0, 0.5, 0.5), (0.0, 0.75, 0.0), (0.0, 0.75, 0.25), (0.0, 0.75, 0.5),
           (0.0, 0.75, 0.75), (0.0, 0.75, 1.0), (0.0, 1.0, 0.25), (0.25, 0.0, 0.0), (0.25, 0.0, 0.25), (0.25, 0.0, 0.5),
           (0.25, 0.0, 1.0), (0.25, 0.25, 0.0), (0.25, 0.25, 0.5), (0.25, 0.25, 1.0), (0.25, 0.5, 0.0), (0.25, 0.5, 0.25),
           (0.25, 0.5, 1.0), (0.25, 0.75, 0.0), (0.25, 0.75, 0.25), (0.25, 0.75, 0.5), (0.25, 1.0, 0.0), (0.25, 1.0, 0.75),
           (0.5, 0.0, 0.0), (0.5, 0.0, 0.25), (0.5, 0.0, 0.5), (0.5, 0.0, 0.75), (0.5, 0.25, 0.0), (0.5, 0.25, 1.0),
           (0.5, 0.5, 0.0), (0.5, 0.5, 0.25), (0.5, 0.5, 0.5), (0.5, 0.5, 0.75), (0.5, 0.75, 0.0), (0.5, 0.75, 0.5),
           (0.5, 0.75, 0.75), (0.5, 1.0, 0.0), (0.5, 1.0, 0.25), (0.5, 1.0, 0.5), (0.5, 1.0, 1.0), (0.75, 0.0, 0.0),
           (0.75, 0.0, 0.25), (0.75, 0.0, 1.0), (0.75, 0.25, 0.0), (0.75, 0.25, 1.0), (0.75, 0.5, 0.0), (0.75, 0.5, 0.25),
           (0.75, 0.5, 0.5), (0.75, 0.5, 0.75), (0.75, 0.5, 1.0), (0.75, 0.75, 0.0), (0.75, 0.75, 0.25), (0.75, 0.75, 0.5),
           (0.75, 0.75, 0.75), (0.75, 0.75, 1.0), (0.75, 1.0, 0.0), (0.75, 1.0, 0.25), (0.75, 1.0, 0.5), (0.75, 1.0, 0.75),
           (0.75, 1.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.5), (1.0, 0.25, 0.25), (1.0, 0.25, 0.5), (1.0, 0.25, 0.75),
           (1.0, 0.25, 1.0), (1.0, 0.5, 0.0), (1.0, 0.5, 0.25), (1.0, 0.5, 0.5), (1.0, 0.5, 0.75),
           (1.0, 0.5, 1.0), (1.0, 0.75, 0.25), (1.0, 0.75, 0.5), (1.0, 0.75, 0.75)]
            palette_new = []

            for p in palette:
                for pp in p:
                    palette_new.append(int(pp*255))
            palette=palette_new

            raw_image = Image.fromarray(denorm(self.img[2][0]).cpu().detach().data.permute(1, 2, 0).numpy())
            im = Image.fromarray(preds.astype(np.uint8))
            im.putpalette(palette)
            
            im_convert = im.convert('RGB')
            blend_im = Image.blend(raw_image,im_convert,0.5)
            blend_im.save(osp.join(val_path, epo_str + '_' + self.name + "_palette.png"))

            # print(osp.join(val_path, epo_str + '_' + self.name + '_aimg.png'))
            # save_img(osp.join(val_path, epo_str + '_' + self.name + '_aimg.png'), img_np)
            # save_img(osp.join(val_path, epo_str + '_' + self.name + '_pred.png'), seg_vis)

        if dict:
            # np.save(osp.join(dict_path, self.name + '.npy'), self.pred_dict)
            pred_pil = Image.fromarray(preds[0],mode='L')
            pred_pil.save(osp.join(dict_path, self.name + '.png'))


        if crf:
            for a in self.args.alphas:
                crf_np = crf_dl(self.name, probs[0], t=a)
                crf_dict = {}
                for c in range(21):
                    # if c in label_cls:
                    crf_dict[c] = crf_np[c]
                np.save(osp.join(crf_path, str(a).zfill(2), self.name + '.npy'), crf_dict)

                h, w = list(crf_dict.values())[0].shape
                tensor = np.zeros((21, h, w), np.float32)
                for key in crf_dict.keys():
                    tensor[key] = crf_dict[key]
                crf_pred = np.argmax(tensor, axis=0).astype(np.uint8)

                seg_vis = voc_palette(crf_pred)
                img_np = denorm(self.img[2][0]).cpu().detach().data.permute(1, 2, 0).numpy()
                save_img(osp.join(val_path, epo_str + '_' + self.name + '_aimg.png'), img_np)
                save_img(osp.join(val_path, epo_str + '_' + self.name + '_pred.png'), seg_vis)

    def dict2crf(self, epo, val_path, dict_path, crf_path):
        epo_str = str(epo).zfill(3)
        # label_cls = torch.unique(self.label).cpu().detach().data.numpy()

        dict = np.load(osp.join(dict_path, self.name + '.npy'), allow_pickle=True).item()
        h, w = list(dict.values())[0].shape
        probs = np.zeros((21, h, w))

        for c in range(21):
            # if c in label_cls:
            probs[c] = dict[c]

        # probs = probs +1e-5

        # print(np.sum(probs[:,100,100]))

        for a in self.args.alphas:
            crf_np = crf_dl(self.name, probs, t=a)
            crf_dict = {}
            for c in range(21):
                # if c in label_cls:
                crf_dict[c] = crf_np[c]
            np.save(osp.join(crf_path, str(a).zfill(2), self.name + '.npy'), crf_dict)

            h, w = list(crf_dict.values())[0].shape
            tensor = np.zeros((21, h, w), np.float32)
            for key in crf_dict.keys():
                tensor[key] = crf_dict[key]
            crf_pred = np.argmax(tensor, axis=0).astype(np.uint8)

            seg_vis = voc_palette(crf_pred)
            img_np = denorm(self.img[2][0]).cpu().detach().data.permute(1, 2, 0).numpy()
            # save_img(osp.join(val_path, epo_str + '_' + self.name + '_aimg.png'), img_np)
            save_img(osp.join(val_path, epo_str + '_' + self.name + '_pred_' + str(a).zfill(2) + '.png'), seg_vis)
            # cv2.imwrite(osp.join(val_path, self.name + '.png'), crf_pred)

    # Evaluate CAM with threshold (without saving dict files)
    def evaluate_cam(self, threshold=0.2):

        gt_file = './data/VOC2012/SegmentationClass/' + self.name + '.png'
        self.gt = np.array(Image.open(gt_file))

        predict_dict = self.cam_dict
        h, w = list(predict_dict.values())[0].shape
        tensor = np.zeros((21, h, w), np.float32)

        for key in predict_dict.keys():
            tensor[key + 1] = predict_dict[key]
        tensor[0, :, :] = threshold
        self.predict = np.argmax(tensor, axis=0).astype(np.uint8)

        cal = self.gt < 255  # Reject object boundary (white region in GT)
        mask = (self.predict == self.gt) * cal

        for i in range(21):
            self.P[i] += np.sum((self.predict == i) * cal)
            self.T[i] += np.sum((self.gt == i) * cal)
            self.TP[i] += np.sum((self.gt == i) * mask)

    # Print loss/accuracy (and re-initialize them)
    def print_log(self, epo, iter):

        loss_str = ''
        acc_str = ''

        for i in range(len(self.loss_names)):
            loss_str += self.base_names[i] + ' : ' + str(round(self.running_loss[i] / self.count, 5)) + ', '

        for i in range(len(self.acc_names)):
            if self.right_count[i] != 0:
                acc = 100 * self.right_count[i] / (self.right_count[i] + self.wrong_count[i])
                acc_str += self.acc_names[i] + ' : ' + str(round(acc, 2)) + ', '
                self.accs[i] = acc

        print(loss_str[:-2])
        print(acc_str[:-2])

        # Plot visdom
        # if self.phase == 'train':
        #     self.vis.plot_loss(epo + iter, [self.running_loss[i] / self.count for i in range(len(self.loss_names))])
        #     self.vis.plot_acc(epo + iter, [self.accs[i] for i in range(len(self.acc_names))])

        self.running_loss = [0] * len(self.loss_names)
        self.right_count = [0] * len(self.acc_names)
        self.wrong_count = [0] * len(self.acc_names)

    # Print IoU of each class and final mIoU
    def print_IoU(self):
        cat = ['backg',
               'aerop', 'bicyc', 'bird ', 'boat ', 'bottl',
               ' bus ', ' car ', ' cat ', 'chair', ' cow ',
               'table', ' dog ', 'horse', 'motor', 'perso',
               'plant', 'sheep', 'sofa ', 'train', 'tvmon']

        cat_str1 = ''
        cat_str2 = ''
        iou_str1 = ''
        iou_str2 = ''

        self.IoU = [0] * 21
        for i in range(21):
            self.IoU[i] = self.TP[i] / (self.T[i] + self.P[i] - self.TP[i] + 1e-10)
        self.mIoU = 100 * sum(self.IoU) / 21

        for i in range(11):
            temp = 100 * self.IoU[i]
            cat_str1 += cat[i] + '   '
            iou_str1 += str(round(temp, 2)) + '   '
        for i in range(10):
            temp = 100 * self.IoU[i + 11]
            cat_str2 += cat[i + 11] + '   '
            iou_str2 += str(round(temp, 2)) + '   '

        print(cat_str1)
        print(iou_str1)
        print(cat_str2)
        print(iou_str2)
        print("mIoU: " + str(round(self.mIoU, 3)))

    # Save original image and CAM at given path
    def visualize_cam(self, epo, iter, path):

        epo_str = str(epo).zfill(3)
        iter_str = str(iter).zfill(3)

        img = self.img

        B, _, H, W = img.shape

        # c = torch.unique(self.label).cpu().detach().data.numpy()[B-1][1]

        img_np = denorm(img[B - 1]).cpu().detach().data.permute(1, 2, 0).numpy()
        plt.imshow(img_np)
        plt.savefig(path + '/' + epo_str + '_' + iter_str + '_' + self.name + '_img.png')
        # save_img(path + '/' + epo_str + '_' + iter_str + '_' + self.name + '_img.png', img_np)
        plt.close()

        label_np = self.label[B - 1].cpu().detach().squeeze(0).data.numpy()
        plt.imshow(label_np)
        plt.savefig(path + '/' + epo_str + '_' + iter_str + '_' + self.name + '_label.png')
        plt.close()
        # save_img(path + '/' + epo_str + '_' + iter_str + '_' + self.name + '_label.png', label_np)

        # gt = self.label[B - 1].cpu().detach().numpy()
        # c = np.nonzero(gt)[0][0]

        # with torch.no_grad():
        #     cam = self.probs[B - 1]
        #     pred_large = F.interpolate(cam.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
        #     pred_large = pred_large.squeeze(0).cpu().detach().data.numpy()
        #     save_img(
        #         path + '/' + epo_str + '_' + iter_str + '_' + self.name + '_img_cam_'+ '.png',
        #         img_np, pred_large[c])

    # Visualize GT and prediction
    def visualize_pred(self, path):
        save_img(osp.join(path, self.name + '_gt.png'), voc_palette(self.gt))
        save_img(osp.join(path, self.name + '_pred.png'), voc_palette(self.predict))

    # Define stochastic mask with given probability
    def stochastic_mask(self, mask, p):

        sto_mask = torch.randint(0, 100, size=mask.size()).cuda()
        sto_mask = (sto_mask < 100 * p).type(torch.FloatTensor).cuda()
        return sto_mask

        # Count the number of right/wrong predictions for each accuracy

    def count_rw(self, label, out, idx):
        for b in range(self.bs):
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