from functools import lru_cache
import math  # 新增导入

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.generator import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling

from .APBSN import APBSN
from .CSCBSN import CSCBSN
from .MMBSN import MMBSN

class BSN(nn.Module):

    @classmethod
    @lru_cache(maxsize=1)
    def bsn_model(cls):
        return cls()

    def __init__(self, type='MMBSN', pd_a=5, pd_b=2, pd_pad=0, R3=True, R3_T=8, R3_p=0.16,
                 in_ch=3, bsn_base_ch=128, bsn_num_module=9, DCL1_num=2, DCL2_num=7, mask_type='o_fsz',
                 extend_large=True, use_moe=True, hard_moe=True, 
                 use_semantic=True, semantic_encoder_type='DINOv2_ViT_Small'):
        '''
        Args:
            type           : BSN model type
            pd_a           : 'PD stride factor' during training
            pd_b           : 'PD stride factor' during inference
            pd_pad         : pad size between sub-images by PD process (建议设为0)
            R3             : flag of 'Random Replacing Refinement'
            R3_T           : number of masks for R3
            R3_p           : probability of R3
            bsn            : blind-spot network type
            in_ch          : number of input image channel
            bsn_base_ch    : number of bsn base channel
            bsn_num_module : number of module
            mask_type      : types of mask,eg:'o_fsz', 'o_r_c'
            use_semantic   : whether to use semantic guidance
            semantic_encoder_type : type of semantic encoder
        '''
        super().__init__()

        # network hyper-parameters
        self.pd_a = pd_a
        self.pd_b = pd_b
        self.pd_pad = pd_pad  # 建议设为0，因为我们在denoise中使用了更好的保护策略
        self.R3 = R3
        self.R3_T = R3_T
        self.R3_p = R3_p

        # define network
        if type == 'APBSN':
            self.bsn = APBSN(in_ch, in_ch, bsn_base_ch, bsn_num_module, mask_type)
        elif type == 'CSCBSN':
            self.bsn = CSCBSN(in_ch, in_ch, bsn_base_ch, bsn_num_module, mask_type)
        elif type == 'MMBSN':
            # 将新参数传递给MMBSN
            self.bsn = MMBSN(in_ch, in_ch, bsn_base_ch, DCL1_num, DCL2_num, mask_type,
                             extend_large=extend_large, use_moe=use_moe, hard_moe=hard_moe,
                             use_semantic=use_semantic, semantic_encoder_type=semantic_encoder_type)
        else:
            raise NotImplementedError('bsn type %s is not implemented' % type)

    def forward(self, img, pd=None):
        '''
        Foward function includes sequence of PD, BSN and inverse PD processes.
        Note that denoise() function is used during inference time (for differenct pd factor and R3).
        '''
        # default pd factor is training factor (a)
        if pd is None: pd = self.pd_a

        # do PD
        if pd > 1:
            pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            # 增加 mode='reflect'，防止补零产生黑边
            pd_img = F.pad(img, (p, p, p, p), mode='reflect')

        # forward blind-spot network
        pd_img_denoised = self.bsn(pd_img)

        # do inverse PD
        if pd > 1:
            img_pd_bsn = pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            img_pd_bsn = pd_img_denoised[:, :, p:-p, p:-p]

        return img_pd_bsn

    def denoise(self, x):
        '''
        [最终修复版 - 消除伪影专用]
        1. 使用 Reflection Padding (镜像填充) 消除边缘重影。
        2. 动态计算 Margin，确保其为 pd 的整数倍，消除网格伪影。
        '''
        b, c, h, w = x.shape
        pd = self.pd_b 

        # ------------------------------------------------------
        # 1. 基础对齐 (让长宽能被 pd 整除)
        # ------------------------------------------------------
        pad_h = (pd - h % pd) % pd
        pad_w = (pd - w % pd) % pd
        
        # 使用镜像填充，防止右侧和下侧出现黑边
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # ------------------------------------------------------
        # 2. 动态计算 Margin (防止网格错位)
        # ------------------------------------------------------
        # 我们希望 margin 大约是 32，但必须严格是 pd 的倍数。
        # 例如：如果 pd=5，margin=35；如果 pd=2，margin=32。
        # 这样保证了裁剪时的网格相位不发生偏移。
        base_margin = 32
        margin = int(math.ceil(base_margin / pd) * pd)

        # ------------------------------------------------------
        # 3. 核心步骤：镜像填充 -> 推理 -> 裁剪
        # ------------------------------------------------------
        # mode='reflect' 是消除声纳图边缘重影的绝对核心！
        x_padded = F.pad(x, (margin, margin, margin, margin), mode='reflect')
        
        # 推理
        img_pd_bsn_full = self.forward(img=x_padded, pd=pd)
        
        # 裁剪掉 margin (把生成的伪影切掉)
        img_pd_bsn = img_pd_bsn_full[..., margin:-margin, margin:-margin]

        # ------------------------------------------------------
        # 4. R3 循环 (同步保护)
        # ------------------------------------------------------
        if not self.R3:
            return img_pd_bsn[:, :, :h, :w]
        else:
            denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
            
            for t in range(self.R3_T):
                indice = torch.rand_like(x)
                mask = indice < self.R3_p

                tmp_input = torch.clone(img_pd_bsn).detach()
                tmp_input[mask] = x[mask]
                
                # R3 也要加同样的镜像填充，防止伪影回归
                tmp_input_padded = F.pad(tmp_input, (margin, margin, margin, margin), mode='reflect')
                out_full = self.forward(tmp_input_padded, pd=pd)
                out_cropped = out_full[..., margin:-margin, margin:-margin]
                
                denoised[..., t] = out_cropped

            return torch.mean(denoised, dim=-1)[:, :, :h, :w]