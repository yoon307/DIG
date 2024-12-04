import torch
import torch.nn as nn
from functools import partial
from networks.vision_transformer import VisionTransformer, _cfg, VisionTransformer_ysh
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from networks.resnet38d import ResBlock, ResBlock_bot, ABF2
from networks.vision_transformer import Block, PatchEmbed
from networks.resnet101 import ResNet, ResBlock_ysh, ResBlock_ysh2
from torch.cuda.amp import autocast

import math

import pdb


__all__ = [
           'deit_small_MCTformerV2_DiG',
           ]


class convbnrelu(nn.Module):
    def __init__(self,in_chan,out_chan,ks=1,pd=0 ,relu = True):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_chan,out_chan,ks,1,pd,bias=True)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = relu
    
    def forward(self,x):
        x = self.conv1x1(x)
        x = self.bn(x)
        
        if self.relu:
            x = self.relu(x)

        return x 
    
class Block_unet(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        # if exists(scale_shift):
        #     scale, shift = scale_shift
        #     x = x * (scale + 1) + shift

        x = self.act(x)
        return x
    
class DiffBlock_ysh(nn.Module):
    def __init__(self, dim, dim_out, relu=True):
        super().__init__()

        self.conv1x1 = nn.Conv2d(dim,dim_out,1,bias=False)
        self.conv3x3 = nn.Conv2d(dim,dim_out,3,1,1)
        # self.conv5x5 = nn.Conv2d(dim,dim_out,5,1,2)

        self.relu = relu

        self.init_weight()

    def forward(self, x):

        x1 = self.conv1x1(x)
        x3 = self.conv3x3(x)
        # x5 = self.conv5x5(x)

        # if self.idx == 0:
        x = x1 + x3
        # elif self.idx == 1:
            # x = x1 + x5
        # elif self.idx == 2:
            # x = x1 + x3 + x5

        if self.relu:
            x = F.relu(x)
    
        return x
    
    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


#Backup
from timm.models.layers import DropPath

from networks.resnet101 import Bottleneck
class LFCA2(nn.Module):
    def __init__(self, D,D_f,C, **kwargs):
        super().__init__(**kwargs)

        self.D = D
        self.D_f = D_f
        self.C = C

        num_group = 1

        self.diff_init = ResBlock_ysh(self.D_f*5,self.D_f)

        self.norm_k = nn.LayerNorm(self.D,eps=1e-6)
        self.norm_v = nn.LayerNorm(self.D_f,eps=1e-6)

        self.norm_p1 = nn.GroupNorm(num_group,self.D_f,eps=1e-6)
        self.norm_c1 = nn.LayerNorm(self.D,eps=1e-6)

        self.key =  nn.Conv2d(self.D_f,self.D,1,1,0)
        self.value =  nn.Conv2d(self.D_f,self.D_f,3,1,1)

        self.mlp_c0 = nn.Linear(self.D_f,self.D)
        self.mlp_p0 = nn.Conv2d(self.D_f,self.D_f,3,1,1)
        
        self.mlp_c1 = nn.Sequential(
            nn.Linear(self.D,self.D*4),
            nn.GELU(),
            nn.Linear(self.D*4,self.D),
            )
        self.mlp_p1 = nn.Sequential(
            nn.Conv2d(self.D_f,self.D_f*4,3,1,1),
            nn.GELU(),
            # nn.Dropout(0.1),
            nn.Conv2d(self.D_f*4,self.D_f,1,1,0),
            # nn.Dropout(0.1)
            )

        # self.drop_path = DropPath(0.1)
        self.drop_path = DropPath(0.)
        self.attn_drop = nn.Dropout(0.1)
        self.drop = nn.Dropout(0.1)

        self.init_weight()
        self._reset_parameters()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def _reset_parameters(self):
        # torch.manual_seed(0)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)


    
    def forward(self, query_vit, feat_diff_cat):
        
        query_vit_input = query_vit
        B,_,h_diff,w_diff = feat_diff_cat.size()
        N = query_vit[:,self.C:,:].size(1)
        h_trm, w_trm = int(N ** 0.5), int(N ** 0.5)

        kv = self.diff_init(feat_diff_cat) #B D_f H W

        #Cross-attention
        key_diff = self.norm_k(self.key(kv).view(B,self.D,-1).permute(0,2,1)) #B D_f H' W' -> B N_f D
        value_diff = self.norm_v(self.value(kv).view(B,self.D_f,-1).permute(0,2,1)) #B D_f H' W' -> B N_f D_f
        # key_diff = self.norm_k(self.key(kv)).view(B,self.D,-1).permute(0,2,1) #B D_f H' W' -> B N_f D
        # value_diff = self.norm_v(self.value(kv)).view(B,self.D_f,-1).permute(0,2,1) #B D_f H' W' -> B N_f D_f

        query_vit = query_vit.view(query_vit.shape[0], query_vit.shape[1], 8, 384 // 8)
        key_diff = key_diff.view(key_diff.shape[0], key_diff.shape[1], 8, 384 // 8)
        value_diff = value_diff.view(value_diff.shape[0], value_diff.shape[1], 8, self.D_f // 8)

        query_vit = query_vit.permute(0, 2, 1, 3) # [batch_size, num_heads, query_len, head_dim]
        key_diff = key_diff.permute(0, 2, 1, 3)
        value_diff = value_diff.permute(0, 2, 1, 3)

        cross_attn = torch.matmul(query_vit, key_diff.transpose(-2, -1)) / ((384//8) ** 0.5)
        # cross_attn = torch.matmul(query_vit, key_diff.transpose(-2, -1)) / ((384) ** 0.5)

        cross_attn = F.softmax(cross_attn, dim=-1)
        weight = cross_attn
        # cross_attn = self.attn_drop(cross_attn)
        o = torch.matmul(cross_attn, value_diff)  # [batch_size, num_heads, query_len, head_dim]
   
        o = o.transpose(1,2).reshape(B,N+self.C,-1)

        o_c = o[:, :self.C, :]    # B C D_f
        o_p = o[:, self.C:, :].permute(0,2,1).reshape(B,self.D_f,h_trm,w_trm) # B N D_f -> B D_f H W

        o_c = self.mlp_c0(o_c)
        x_c = o_c + query_vit_input[:,:self.C,:]
        x_c = self.mlp_c1(self.norm_c1(x_c)) + x_c

        o_p = F.relu(self.mlp_p0(o_p),inplace=True)
        o_p = F.interpolate(o_p,size=(h_diff,w_diff),mode='bilinear',align_corners=False) #align False better
        # o_p = F.interpolate(o_p,size=(h_diff,w_diff),mode='bilinear',align_corners=True) 
        x_p = kv + self.drop_path(o_p)
        # x_p = self.drop_path(F.relu(self.mlp_p1(self.norm_p1(x_p)),inplace=True)) + x_p
        x_p = self.drop_path(F.relu(self.mlp_p1(x_p),inplace=True)) + x_p

        return x_c, x_p, weight.sum(1)[:,:self.C,:].reshape(B,self.C,h_diff,w_diff)
        # return x_c, x_p, weight[:,:self.C,:].reshape(B,self.C,h_diff,w_diff)
    
class LFCA(nn.Module):
    def __init__(self, D,D_f,C, **kwargs):
        super().__init__(**kwargs)

        self.D = D
        self.D_f = D_f
        self.C = C

        self.diff_init = ResBlock_ysh(self.D_f*5,self.D_f)         # ReLU(2 ConvLNReLU + 1x1 Conv)
        # self.diff_init = ResBlock_ysh(self.D_f,self.D_f)         # ReLU(2 ConvLNReLU + 1x1 Conv)
        self.lnorm = nn.GroupNorm(1,self.D_f,eps=1e-6)
        self.key =  nn.Conv2d(self.D_f,self.D,1,1,0) #nn.Conv2d(self.D_f,self.D,1,1,0,bias=False)
        self.value =  nn.Conv2d(self.D_f,self.D_f,1,1,0) #nn.Conv2d(self.D_f,self.D_f,1,1,0,bias=False) #
        self.diff_head0 = nn.Conv2d(self.D_f,self.D_f,3,1,1) # attn2-base
        self.diff_head1 = nn.Conv2d(self.D_f,self.D_f,3,1,1)   # attn3-base


        self.drop_path = DropPath(0)

        nn.init.constant_(self.lnorm.weight, 1)
        nn.init.constant_(self.lnorm.bias, 0)
        nn.init.kaiming_normal_(self.key.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.key.bias, 0)
        nn.init.kaiming_normal_(self.value.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.value.bias, 0)
        nn.init.kaiming_normal_(self.diff_head0.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.diff_head1.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, query_vit, feat_diff_cat):
        
        B,_,h_diff,w_diff = feat_diff_cat.size()
        N = query_vit[:,self.C:,:].size(1)
        h_trm, w_trm = int(N ** 0.5), int(N ** 0.5)

        feat_diff = self.diff_init(feat_diff_cat)

        feat_diff_ln = self.lnorm(feat_diff)

        key_diff = self.key(feat_diff_ln).view(B,self.D,-1).permute(0,2,1) #B D_f H' W' -> B N_f D
        value_diff = self.value(feat_diff_ln).view(B,self.D_f,-1).permute(0,2,1) #B D_f H' W' -> B N_f D_f

        cross_attn = (query_vit @ key_diff.permute(0,2,1)) * ((self.D)**-0.5) #B C+N N_f

        cross_attn = F.softmax(cross_attn,dim=-1) #B C+N N_f

        out_cross = (cross_attn @ value_diff)   # B C+N D_f

        out_cross_c = out_cross[:, :self.C, :]    # B C D_f
        out_cross_p = out_cross[:, self.C:, :].permute(0,2,1).reshape(B,self.D_f,h_trm,w_trm) # B N D_f -> B D_f H W

        out_cross_p = F.relu(self.diff_head0(out_cross_p),inplace=True)

        out_cross_p = F.interpolate(out_cross_p,size=(h_diff,w_diff),mode='bilinear',align_corners=True)
        out_cross_p = self.drop_path(out_cross_p) + feat_diff
        out_cross_p = self.drop_path(F.relu(self.diff_head1(self.lnorm(out_cross_p)),inplace=True)) + out_cross_p


        return out_cross_c, out_cross_p

class MCTformerV2_DiG(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_classes, self.embed_dim))

        if True:
    
            self.D_f = 512+256+128+64+64
            self.D = 384
            
            self.LFCA0 = LFCA2(self.D,self.D_f,self.num_classes)
            self.LFCA1 = LFCA2(self.D,self.D_f,self.num_classes)
            # self.LFCA2 = LFCA2(self.D,self.D_f,self.num_classes)


            self.diff_head_p = nn.Sequential(DiffBlock_ysh(self.D_f,self.D_f),nn.Conv2d(self.D_f,20,1,1,0,bias=False))
            # self.diff_head_p = nn.Sequential(nn.Conv2d(self.D_f*2,self.D_f*2,1,1,0,bias=False),nn.Conv2d(self.D_f*2,20,1,1,0,bias=False))
            
            # self.diff_head_c = nn.Linear(self.D,self.D)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        # nn.init.kaiming_normal_(self.diff_head_p[1].weight, mode='fan_in', nonlinearity='relu')
        print(self.training)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
        N = self.pos_embed.shape[1] - self.num_classes
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:self.num_classes]
        patch_pos_embed = self.pos_embed[:, self.num_classes:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)
    
    def max_norm(self, cam_cp):
        N, C, H, W = cam_cp.size()
        cam_cp = F.relu(cam_cp)
        max_v = torch.max(cam_cp.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        min_v = torch.min(cam_cp.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        cam_cp = F.relu(cam_cp - min_v - 1e-5) / (max_v - min_v + 1e-5)
        return cam_cp

    def forward_features(self, x, n=12):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
     
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        attn_weights = []
        x_list = []
        ctk_list = []
        q_list = []
        k_list = []

        for i, blk in enumerate(self.blocks):
            x, weights_i, qkv,_ = blk(x)
            attn_weights.append(weights_i)
            x_list.append(x)
            ctk_list.append(x[:, 0:self.num_classes])
            q_list.append(qkv[0])
            k_list.append(qkv[1])

        return x[:, 0:self.num_classes], x[:, self.num_classes:], attn_weights , x_list, q_list, k_list

    def forward(self, x, feat_diff = None, return_att=False, n_layers=12, detach=True):
        w, h = x.shape[2:]
        x_cls, x_patch, attn_weights, x_list, q_list, k_list = self.forward_features(x)
        n, p, c = x_patch.shape

        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
    
        else:
            x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])

        if feat_diff != None:
            if w != h:
                w_trm = w // self.patch_embed.patch_size[0]
                h_trm = h // self.patch_embed.patch_size[0]
            else:
                h_trm, w_trm = int(p ** 0.5), int(p ** 0.5)


            h_diff, w_diff = feat_diff[0].size()[2:]

            feat_diff_cat =  torch.stack(feat_diff,dim=1).view(n,-1,h_diff,w_diff)  
            # feat_diff_cat =  sum(feat_diff)/len(feat_diff)

            ####Cross-attn####
            q_stacked = torch.stack(q_list,dim=0) #B H C+N D//H  # layer 10,11 mean
            query0 = q_stacked[-1].permute(0,2,1,3).reshape(n,-1,384).detach() #B H N+C D//head -> B C+N D
            query1 = q_stacked[-2].permute(0,2,1,3).reshape(n,-1,384).detach() #B H N+C D//head -> B C+N D
     
            out_cross_c0, out_cross_p0 , cattn0 = self.LFCA0(query0,feat_diff_cat)
            out_cross_c1, out_cross_p1 , cattn1= self.LFCA1(query1,feat_diff_cat)
            # out_cross_c2, out_cross_p2 , cattn2 = self.LFCA2(query2,feat_diff_cat)


            out_cross_c = (out_cross_c0 + out_cross_c1)/2 #BEST
            out_cross_p = (out_cross_p0 + out_cross_p1)/2
            # out_cross_p = torch.cat([out_cross_p0,out_cross_p1],dim=1)
            cattn = (cattn0+cattn1)/2

            # out_cross_c = (out_cross_c0 + out_cross_c1 + out_cross_c2)/3 #BEST
            # out_cross_p = (out_cross_p0 + out_cross_p1 + out_cross_p2)/3
            
            # out_cross_c = torch.cat([out_cross_c0,out_cross_c1],dim=1)
            # out_cross_p = torch.cat([out_cross_p0,out_cross_p1],dim=1)
     
            # out_cross_c = self.diff_head_c(out_cross_c) #+ out_cross_c
            cam_diff = self.diff_head_p(out_cross_p)

            cam_diff_out = F.relu(cam_diff)
        
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        
        feat_imd = x_patch
        
        x_patch = self.head(x_patch)

        # x_patch_logits = self.avgpool(x_patch).squeeze(3).squeeze(2) ######ORIGINAL
        x_patch_logits = torch.topk(F.relu(x_patch).view(n,self.num_classes,-1),dim=-1,k=3)[0].mean(-1) - torch.topk(F.relu(-x_patch).view(n,self.num_classes,-1),dim=-1,k=3)[0].mean(-1)

        attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N

        feature_map = x_patch#.detach().clone()  # B * C * 14 * 14
        feature_map = F.relu(feature_map)

        n, c, h, w = feature_map.shape

        mtatt = attn_weights[-n_layers:].sum(0)[:, 0:self.num_classes, self.num_classes:].reshape([n, c, h, w])

        cams = mtatt * feature_map  # B * C * 14 * 14

        patch_attn = attn_weights[:, :, self.num_classes:, self.num_classes:]
        patch_attn = torch.sum(patch_attn, dim=0) #B 196 196

        ##########################################################################
        x_cls_logits = x_cls.mean(-1)
        ##########################################################################

        rcams = torch.matmul(patch_attn.unsqueeze(1), cams.view(cams.shape[0],cams.shape[1], -1, 1)).reshape(cams.shape[0],cams.shape[1], h, w) #(B 1 N2 N2) * (B,20,N2,1)

        ########################
        if feat_diff != None:

            diff_cls_logits = out_cross_c.mean(-1)
            diff_patch_logits = torch.topk(F.relu(cam_diff).view(n,self.num_classes,-1),dim=-1,k=3)[0].mean(-1) - torch.topk(F.relu(-cam_diff).view(n,self.num_classes,-1),dim=-1,k=3)[0].mean(-1)

        #######################

        outs = {}
        outs['cls']= x_cls_logits
        outs['pcls']= x_patch_logits
        outs['cams']= F.relu(x_patch)
        outs['fcams']= F.relu(x_patch) * mtatt
        outs['attn']= patch_attn
        outs['mtatt']= mtatt
        outs['rcams']= rcams
        # outs['rcams_v1'] = rcams_v1

        outs['feat'] = feat_imd


        if feat_diff != None:
            # outs['feat_diff'] = feat_diff_sum
            outs['cattn_diff'] = cattn
            outs['cam_diff'] = cam_diff_out
            outs['diff_cls'] = diff_cls_logits
            outs['diff_pcls'] = diff_patch_logits
            # outs['cam_diff_cs'] = cam_diff_cs
            # outs['pred_diff_cs'] = pred_diff_cs
        if return_att:
            # return cam_diff_out
            return rcams
            # return F.relu(x_patch)
        else:
            return outs
     

def ConvLnReLU(in_chan, out_chan, ks, stride, pd, dilation=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_chan, out_chan, ks, stride=stride, padding=pd, bias=False, groups=groups, dilation=dilation),
        nn.GroupNorm(groups,out_chan), nn.ReLU(inplace=True))

@register_model
def deit_small_MCTformerV2_DiG(pretrained=False, **kwargs):
    model = MCTformerV2_DiG(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['cls_token', 'pos_embed']}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model
