import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入掩膜类
from .masks import CentralMaskedConv2d, RowMaskedConv2d, ColMaskedConv2d, \
    SzMaskedConv2d, fSzMaskedConv2d, angle45MaskedConv2d, \
    angle135MaskedConv2d, chaMaskedConv2d, fchaMaskedConv2d, huiMaskedConv2d, \
    DeepHuiMaskedConv2d, SuperHuiMaskedConv2d, ThickAngle45MaskedConv2d

# ================= 语义注入模块 =================
class SemanticInjector(nn.Module):
    def __init__(self, in_ch, sem_dim=384):
        super().__init__()
        # 输出通道为 2 * in_ch (一半给 Scale，一半给 Shift)
        self.affine = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(sem_dim, 2 * in_ch, kernel_size=1)
        )

    def forward(self, x, f_semantic):
        # x: [B, C, H, W], f_semantic: [B, 384, 1, 1]
        if f_semantic is None:
            return x
        
        # 1. 计算仿射参数
        style = self.affine(f_semantic)
        
        # 2. 切分 Scale 和 Shift
        scale, shift = style.chunk(2, dim=1)
        
        # 3. 仿射注入公式： (1 + scale) * x + shift
        return x * (1 + scale) + shift
# =======================================================

# ================= SKNet风格软路由 =================
class SKFusionRouter(nn.Module):
    def __init__(self, in_ch, num_experts, reduction=8):
        super().__init__()
        self.num_experts = num_experts
        # 降维后的通道数，最小为 32，防止通道过少信息丢失
        d = max(in_ch // reduction, 32)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_ch, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, in_ch * num_experts)
        )
        
        self.softmax = nn.Softmax(dim=1) # 在专家维度(N)上进行 Softmax

    def forward(self, experts_stack):
        # experts_stack: [B, N, C, H, W]
        B, N, C, H, W = experts_stack.shape
        
        # 1. Fuse: 将所有专家特征相加得到 U [B, C, H, W]
        U = torch.sum(experts_stack, dim=1)
        
        # 2. Squeeze: 全局平均池化得到 s [B, C]
        s = self.avg_pool(U).view(B, C)
        
        # 3. Excitation: 生成注意力权重 z [B, N*C]
        z = self.fc(s)
        
        # Reshape 为 [B, N, C] 以便在 N 维度做 Softmax
        weights = z.view(B, N, C)
        
        # 4. Softmax: 使得每个通道上，所有专家的权重之和为 1
        weights = self.softmax(weights) # [B, N, C]
        
        # 5. Reweight & Fuse: 加权求和
        # 扩展权重维度以匹配特征图 [B, N, C, 1, 1]
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        
        # 加权 [B, N, C, H, W] * [B, N, C, 1, 1] -> Sum over N -> [B, C, H, W]
        V = torch.sum(experts_stack * weights, dim=1)
        
        return V
# =======================================================

# ================= 双分支专家模块 =================
class DualBranchExpert(nn.Module):
    """
    双分支专家模块：内部包含两个不同掩膜的 DC_branchl，
    输出融合后的特征，作为一个单一的专家提供给 Router。
    """
    def __init__(self, stride, in_ch, mask_type_a, mask_type_b, num_module, semantic_dim=384):
        super().__init__()
        
        # 分支 A
        self.branch_a = DC_branchl(stride, in_ch, mask_type_a, num_module, semantic_dim)
        # 分支 B
        self.branch_b = DC_branchl(stride, in_ch, mask_type_b, num_module, semantic_dim)
        
        # 融合层：将两个分支的输出 (2 * in_ch) 融合回 (in_ch)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, f_semantic):
        # 分别运行两个分支
        # DC_branchl 返回 (expert_feat, branch_feat)，我们取第一个 expert_feat
        feat_a, _ = self.branch_a(x, f_semantic)
        feat_b, _ = self.branch_b(x, f_semantic)
        
        # 拼接
        cat = torch.cat([feat_a, feat_b], dim=1)
        
        # 融合
        out = self.fusion(cat)
        
        return out
# =======================================================

# ================= 优化后的 MMBSN 类 =================
class MMBSN(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=128, DCL1_num=2, DCL2_num=7, 
                 mask_type='ignored', 
                 extend_large=True, use_moe=True, hard_moe=False,
                 use_semantic=True, semantic_encoder_type='DINOv2_ViT_Small'):
        super().__init__()
        
        self.base_ch = base_ch 
        self.use_moe = use_moe
        self.use_semantic = use_semantic
        self.semantic_encoder_type = semantic_encoder_type
        self.semantic_dim = 384
        self.semantic_encoder = None

        # 1. 加载 DINOv2 (保持不变)
        if self.use_semantic:
            print(f"Initializing semantic encoder: {self.semantic_encoder_type}")
            try:
                # 本地路径配置
                local_repo_path = '/root/autodl-tmp/MM-BSN/dinov2_local/dinov2-main'
                local_weight_path = '/root/autodl-tmp/MM-BSN/dinov2_local/dinov2_vits14_pretrain.pth'
                import os
                if not os.path.exists(local_repo_path): # 路径回退兼容
                     local_repo_path = './dinov2_local/dinov2-main'
                     local_weight_path = './dinov2_local/dinov2_vits14_pretrain.pth'

                # 从本地加载模型
                self.semantic_encoder = torch.hub.load(local_repo_path, 'dinov2_vits14', source='local', pretrained=False)
                state_dict = torch.load(local_weight_path, map_location='cpu')
                self.semantic_encoder.load_state_dict(state_dict)
                
                # 冻结参数
                for param in self.semantic_encoder.parameters():
                    param.requires_grad = False
                self.semantic_encoder.eval()
                print("DINOv2 loaded successfully from local repository.")
            except Exception as e:
                print(f"DINOv2 Error: {e}")
                print("Semantic guidance will be disabled.")
                self.use_semantic = False

        # 2. 初始层
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # ============ 3. 定义四个双分支专家 (核心修改) ============
        
        # --- 专家 1: 3x3 细节专家 (点+线组合) ---
        # 分支A: 中心点掩膜，分支B: 45度对角线掩膜
        self.expert_small = DualBranchExpert(
            stride=2, # 对应 3x3
            in_ch=base_ch, 
            mask_type_a='central',  # 分支 A：保持中心盲点
            mask_type_b='a45',      # 分支 B：改为 45 度对角线掩膜
            num_module=DCL1_num, 
            semantic_dim=self.semantic_dim
        )
        
        # --- 专家 2: 5x5 纹理专家 (点+粗线组合) ---
        # 分支A: 中心点掩膜，分支B: 粗斜线掩膜
        self.expert_medium = DualBranchExpert(
            stride=3, # 对应 5x5
            in_ch=base_ch, 
            mask_type_a='central',    # 分支 A：中心盲点
            mask_type_b='thick45',    # 分支 B：改为粗斜线（宽度为 3 的对角带宽）
            num_module=DCL1_num, 
            semantic_dim=self.semantic_dim
        )
        
        # --- 专家 3: 7x7 结构专家 (面+粗线组合) ---
        # 这是去声纳散斑的关键，大尺度互补掩膜
        self.expert_large = DualBranchExpert(
            stride=4, # 对应 7x7
            in_ch=base_ch, 
            mask_type_a='deephui',   # 分支 A：深层回字（挖掉中间 3x3 区域）
            mask_type_b='thick45',   # 分支 B：粗斜线（在 7x7 下会自动覆盖更大带宽）
            num_module=DCL1_num, 
            semantic_dim=self.semantic_dim
        )
        
        # --- 专家 4: 9x9 超大尺度专家 (超深回字 + 粗斜线组合) ---
        # 对应 stride=5 (2*5-1=9)，专门对付超大型声纳散斑块
        self.expert_xlarge = DualBranchExpert(
            stride=5, # 对应 9x9
            in_ch=base_ch, 
            mask_type_a='superhui', # 分支 A：超深回字（9x9中挖掉5x5）
            mask_type_b='thick45',  # 分支 B：粗斜线
            num_module=DCL1_num, 
            semantic_dim=self.semantic_dim
        )

        # 修改专家总数为 4
        self.num_experts = 4

        # ============ 4. 软路由 (SKNet Style) ============
        if self.use_moe:
            self.router = SKFusionRouter(base_ch, self.num_experts)
        
        # ============ 5. 最终输出 ============
        total_out_channels = base_ch 
        self.tail = nn.Sequential(
            nn.Conv2d(total_out_channels, base_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch//2, out_ch, kernel_size=1)
        )

    def forward(self, x):
        # 1. 提取基特征
        feat_base = self.head(x)

        # 2. 提取语义特征 (带归一化)
        f_semantic = None
        if self.use_semantic and self.semantic_encoder is not None:
            with torch.no_grad():
                # 优化显存：不clone整个张量，只创建视图
                if x.shape[1] == 1:
                    # 单通道图像复制成3通道
                    x_dino = x.expand(-1, 3, -1, -1)
                elif x.shape[1] > 3:
                    # 多通道图像取前3个通道
                    x_dino = x[:, :3, :, :]
                else:
                    # 3通道图像直接使用（不clone）
                    x_dino = x
                
                # 鲁棒性检查：如果输入是 0-255，转为 0-1
                if x_dino.max() > 1.0 + 1e-5:
                    x_dino = x_dino / 255.0
                
                # 归一化参数
                mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
                x_dino = (x_dino - mean) / std
                
                # Resize 到 14 的倍数 (DINOv2 patch size)
                h, w = x_dino.shape[2], x_dino.shape[3]
                new_h = max(14, int((h // 14) * 14))
                new_w = max(14, int((w // 14) * 14))
                
                # 避免不必要的插值
                if h != new_h or w != new_w:
                    x_dino = F.interpolate(x_dino, size=(new_h, new_w), mode='bicubic', align_corners=False)

                # 提取特征
                out = self.semantic_encoder.forward_features(x_dino)
                f_semantic = out['x_norm_clstoken'].unsqueeze(-1).unsqueeze(-1)

        # 3. 四个专家并行计算 (每个专家内部已经是双分支融合过的了)
        # 专家 1: Small (3x3) - 点+线组合
        e1 = self.expert_small(feat_base, f_semantic)
        
        # 专家 2: Medium (5x5) - 点+粗线组合
        e2 = self.expert_medium(feat_base, f_semantic)
        
        # 专家 3: Large (7x7) - 面+粗线组合
        e3 = self.expert_large(feat_base, f_semantic)
        
        # --- 新增第四个专家计算 ---
        # 专家 4: XLarge (9x9) - 超深回字+粗斜线组合
        e4 = self.expert_xlarge(feat_base, f_semantic)

        # 4. 堆叠: [B, 4, C, H, W]
        experts_stack = torch.stack([e1, e2, e3, e4], dim=1)

        # 5. 软路由融合 (SKNet)
        if self.use_moe:
            cat_final = self.router(experts_stack)
        else:
            cat_final = torch.mean(experts_stack, dim=1)

        # 6. 输出
        return self.tail(cat_final)
# =======================================================


class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch, mask_type, num_module, semantic_dim=384):
        super().__init__()

        ly0 = []
        ly1_1 = []
        ly1_2 = []
        
        # 根据掩膜类型选择对应的掩膜卷积层
        if mask_type == 'r':
            ly0 += [ RowMaskedConv2d(in_ch, in_ch, kernel_size=2*stride-1, stride=1, padding=stride-1) ]
        elif mask_type == 'c':
            ly0 += [ ColMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'sz':
            ly0 += [ SzMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'fsz':
            ly0 += [ fSzMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'a45':
            ly0 += [ angle45MaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'a135':
            ly0 += [ angle135MaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'hui':
            ly0 += [huiMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'cha':
            ly0 += [chaMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'fcha':
            ly0 += [fchaMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'deephui':  # 深层回字掩膜
            ly0 += [DeepHuiMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'superhui':  # 超深层回字掩膜
            ly0 += [SuperHuiMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'thick45':  # 粗斜线掩膜
            ly0 += [ThickAngle45MaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        else:  # 'central' - 中心盲点掩膜
            ly0 += [ CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]

        ly0 += [ nn.ReLU(inplace=True) ]
        ly0 += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly0 += [ nn.ReLU(inplace=True) ]
        ly0 += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly0 += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly0)

        ly1_1 += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly1_1 += [nn.ReLU(inplace=True)]
        self.conv1_1 = nn.Sequential(*ly1_1)
        self.conv1_2 = nn.Sequential(*ly1_1)

        # 重构 Body 部分，原来的 self.body = nn.Sequential(...) 删掉
        # 改为 ModuleList 以便在 forward 中插入注入操作
        self.dcl_layers = nn.ModuleList([
            DCl(stride, in_ch) for _ in range(num_module)
        ])
        
        # 为每一层 DCL 配备一个注入器
        self.injectors = nn.ModuleList([
            SemanticInjector(in_ch, sem_dim=semantic_dim) for _ in range(num_module)
        ])

        # 最后的 1x1 卷积
        self.body_tail = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        ly1_2 += [nn.Conv2d(in_ch*2, in_ch, kernel_size=1)]
        ly1_2 += [nn.ReLU(inplace=True)]
        self.conv1_3 = nn.Sequential(*ly1_2)

    def forward(self, x, f_semantic=None):
        y0 = self.head(x)
        conv1_1 = self.conv1_1(y0)
        
        # 核心循环注入逻辑
        feat = conv1_1
        # 同时遍历 DCL 层和 注入器
        for dcl, injector in zip(self.dcl_layers, self.injectors):
            feat = dcl(feat)               # 1. 卷积提取特征
            feat = injector(feat, f_semantic)  # 2. 注入全局语义
        
        y1 = self.body_tail(feat)

        cat0 = torch.cat([conv1_1, y1], dim=1)
        conv1_3 = self.conv1_3(cat0)
        conv1_2 = self.conv1_2(y0)

        return conv1_2, conv1_3


class DC_branchl2(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()
        ly = []
        ly += [ DCl(stride, in_ch) for _ in range(num_module) ]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return self.body(x)


class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)