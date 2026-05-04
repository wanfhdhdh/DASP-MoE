import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .masks import CentralMaskedConv2d, RowMaskedConv2d, ColMaskedConv2d, \
    SzMaskedConv2d, fSzMaskedConv2d, angle45MaskedConv2d, \
    angle135MaskedConv2d, chaMaskedConv2d, fchaMaskedConv2d, huiMaskedConv2d


# ==========================================
# 语义注入器 (FiLM机制) - 已优化
# ==========================================
class SemanticInjector(nn.Module):
    """
    标准的 SFT 注入模块：双层 MLP 结构
    流程: Input -> Conv1 -> GELU -> Conv2 -> Output
    """
    def __init__(self, in_ch, sem_dim=384):
        super().__init__()
        
        # 中间隐藏层维度
        # 通常设为 sem_dim // 2 或者 in_ch，既保证效果又节省参数
        hidden_dim = in_ch 
        
        self.mlp = nn.Sequential(
            # 第一层 1x1 Conv: 特征变换与降维
            nn.Conv2d(sem_dim, hidden_dim, kernel_size=1),
            
            # 中间激活函数: 增加非线性 (核心)
            nn.GELU(),
            
            # 第二层 1x1 Conv: 生成最终的 Gamma 和 Beta
            # 输出通道数 = 2 * in_ch (因为要切分)
            nn.Conv2d(hidden_dim, in_ch * 2, kernel_size=1)
        )
        
        # 依然需要初始化，防止黑屏
        self._init_weights()

    def _init_weights(self):
        # 初始化最后一层卷积 (Conv2)
        last_conv = self.mlp[-1]
        
        # 权重设小
        nn.init.kaiming_normal_(last_conv.weight, mode='fan_in', nonlinearity='linear')
        
        if last_conv.bias is not None:
            # 同样的一分为二初始化技巧
            out_ch = last_conv.bias.shape[0]
            half = out_ch // 2
            
            # Gamma (Scale) 设为 1.0
            nn.init.constant_(last_conv.bias[:half], 1.0)
            # Beta (Shift) 设为 0.0
            nn.init.constant_(last_conv.bias[half:], 0.0)

    def forward(self, x, f_semantic):
        if f_semantic is None:
            return x
        
        # 1. 过双层 MLP (两个卷积 + GELU)
        style = self.mlp(f_semantic)
        
        # 2. 一分为二 (Split)
        gamma, beta = style.chunk(2, dim=1)
        
        # 3. 注入
        return x * gamma + beta
    
    def reset_cache(self):
        """重置缓存，用于处理不同batch或不同尺寸的输入"""
        self.cached_semantic = None
        self.cached_size = None


# ==========================================================
# [优化版] 声呐动态决策路由器 (物理能量+物理纹理+语义三流感知)
# ==========================================================
class SonarMoERouter(nn.Module):
    def __init__(self, in_ch=1, sem_dim=384, num_experts=4):
        super().__init__()
        # 1. 物理能量路径 (均值) - 解决"去噪强度"问题
        self.energy_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, 16, 1),
            nn.GELU()
        )
        
        # ==========================================================
        # [优化版] 物理纹理路径：感知斑点噪声与目标的对比度
        # ==========================================================
        self.texture_path = nn.Sequential(
            # 1. 使用标准卷积(groups=1)，捕捉跨通道的局部方差特征
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1, groups=1), 
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True), # 使用 LeakyReLU 对微小波动更敏感
            
            # 2. 关键点：再加一层卷积增加非线性。这样能更好地区分"海床"和"目标"
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 3. 全局池化生成全局统计量
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 16, 1),
            nn.GELU()
        )
        
        # 3. 语义引导路径 (上下文) - 解决"目标保护"问题
        self.semantic_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(sem_dim, 32, 1),
            nn.GELU()
        )
        
        # 4. 决策大脑 - 三位一体融合
        self.decision_mlp = nn.Sequential(
            nn.Linear(16 + 16 + 32, 32), # 融合能量、纹理、语义
            nn.GELU(),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x, f_sem):
        b = x.shape[0]
        # 提取三路决策特征
        e_v = self.energy_path(x).view(b, -1)      # 物理能量特征
        t_v = self.texture_path(x).view(b, -1)     # 物理纹理特征
        s_v = self.semantic_path(f_sem).view(b, -1) # 语义特征
        
        # 三位一体决策：能量 + 纹理 + 语义
        weights = self.decision_mlp(torch.cat([e_v, t_v, s_v], dim=1))
        return weights


# ==========================================
# BSN 单元包装类 - 方案二的核心
# ==========================================
class BSN_Unit(nn.Module):
    def __init__(self, stride, in_ch, mask1, mask2, DCL1_num, DCL2_num, sem_dim):
        super().__init__()
        # 1. 两个独立的方向分支 (内部自带语义注入逻辑)
        self.branch1 = DC_branchl(stride, in_ch, mask1, DCL1_num, sem_dim)
        self.branch2 = DC_branchl(stride, in_ch, mask2, DCL1_num, sem_dim)
        
        # 2. 该单元专属的融合层 (只融合这2个分支)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # 3. 该单元专属的精修层 (图中绿色的 DCL 模块)
        self.refiner = DC_branchl2(stride, in_ch, DCL2_num)

    def forward(self, x, f_semantic):
        # 两个输入：特征 x 和 语义 f_semantic
        e1, br1 = self.branch1(x, f_semantic)
        e2, br2 = self.branch2(x, f_semantic)
        
        # 内部融合：将 br 拼接后通过融合层和精修层
        refined = self.refiner(self.fuse(torch.cat([br1, br2], dim=1)))
        
        # 三个输出：细节1, 细节2, 精修融合
        return e1, e2, refined


class MMBSN(nn.Module):
    '''
    Dilated Blind-Spot Network with Semantic Guidance
    四个BSN架构：8个分支（全对称、多尺度、多方向）
    BSN1: 3x3 central (o) 和 3x3 angle45 (a45)
    BSN2: 3x3 central (o) 和 3x3 angle135 (a135)
    BSN3: 5x5 central (o) 和 5x5 angle45 (a45)
    BSN4: 5x5 central (o) 和 5x5 angle135 (a135)
    '''
    def __init__(self, in_ch=3, out_ch=3, base_ch=128, DCL1_num=2, DCL2_num=7, mask_type='ignored'):
        super().__init__()

        assert base_ch % 2 == 0, "base channel should be divided with 2"

        # ============== 语义特征提取器 ==============
        self.semantic_dim = 384  # ViT-S/14 维度
        self.semantic_encoder = None
        self.load_dinov2() 
        
        # 图像标准化参数 (固定)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        # ===============================================

        # 网络头部
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # ============== [新增] 初始化路由器 ==============
        # 传入原始输入通道 in_ch 和 语义维度
        self.router = SonarMoERouter(in_ch=in_ch, sem_dim=self.semantic_dim, num_experts=4)
        # =================================================

        # ============== 定义 4 个完全独立的 BSN 单元 ==============
        # 取代原来的 8 个散乱分支 + 融合层 + 精修层
   # 取代原来的 8 个散乱分支 + 融合层 + 精修层
        self.bsn1 = BSN_Unit(2, base_ch, 'central', 'r',   DCL1_num, DCL2_num, self.semantic_dim)
        self.bsn2 = BSN_Unit(2, base_ch, 'central', 'c',  DCL1_num, DCL2_num, self.semantic_dim)
        self.bsn3 = BSN_Unit(3, base_ch, 'central', 'a45',   DCL1_num, DCL2_num, self.semantic_dim)
        self.bsn4 = BSN_Unit(3, base_ch, 'central', 'a135',  DCL1_num, DCL2_num, self.semantic_dim)


        # ============== 尾部网络 ==============
        # 输入为 4个单元 * 每个单元3输出 = 12 * base_ch
        # 原本是 8个分支 + 2个精修 = 10 * base_ch
        self.tail = nn.Sequential(
            nn.Conv2d(base_ch * 12, base_ch, kernel_size=1),  # 4个单元 * 每个单元3输出 = 12
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, out_ch, kernel_size=1)
        )

    # ============== 加载DINOv2模型 ==============
    def load_dinov2(self):
        """加载本地DINOv2模型作为语义特征提取器"""
        print("Loading local DINOv2 model...")
        try:
            # 本地路径
            local_repo_path = '/root/autodl-tmp/MM-BSN/dinov2_local/dinov2-main'
            local_weight_path = '/root/autodl-tmp/MM-BSN/dinov2_local/dinov2_vits14_pretrain.pth'
            
            # 验证路径存在性
            if not os.path.exists(local_repo_path):
                print(f"Error: DINOv2 repo not found at {local_repo_path}")
                self.semantic_encoder = None
                return
                
            if not os.path.exists(local_weight_path):
                print(f"Error: DINOv2 weights not found at {local_weight_path}")
                self.semantic_encoder = None
                return
            
            # 将repo路径添加到sys.path
            sys.path.insert(0, os.path.dirname(local_repo_path))
            
            # 加载模型结构
            self.semantic_encoder = torch.hub.load(
                local_repo_path, 
                'dinov2_vits14', 
                source='local', 
                pretrained=False
            )
            
            # 加载权重
            state_dict = torch.load(local_weight_path, map_location='cpu')
            self.semantic_encoder.load_state_dict(state_dict)
            
            # 冻结参数并设置为评估模式
            for param in self.semantic_encoder.parameters():
                param.requires_grad = False
            self.semantic_encoder.eval()
            
            print("DINOv2 loaded successfully.")
            
        except Exception as e:
            print(f"Error loading DINOv2: {e}")
            self.semantic_encoder = None

    # ============== 提取语义特征 - 最终完美版 ==============
    def get_semantic_feature(self, x):
        """
        提取DINOv2语义特征
        1. 自动适配 [0,1] 或 [0,255] 输入范围
        2. 自动适配任意图片尺寸 (动态计算 14 的倍数)
        """
        if self.semantic_encoder is None:
            return None
        
        with torch.no_grad():
            # 1. 通道适配 (确保是3通道)
            if x.shape[1] == 1: 
                x_dino = x.expand(-1, 3, -1, -1)
            elif x.shape[1] > 3: 
                x_dino = x[:, :3, :, :]
            else: 
                x_dino = x
            
            # --- 【核心修改点：自动范围判定】 ---
            # 检查输入 x 是否为 0-255 范围。如果是，则缩放到 0-1 以适配 DINOv2
            # 这里的 clone() 很重要，防止修改了 x_dino 进而影响了原始输入 x (如果是引用的话)
            if x_dino.max() > 1.1:
                x_dino = x_dino.clone() / 255.0
            # -----------------------------------

            # 2. 标准化 (ImageNet Mean/Std)
            x_dino = (x_dino - self.mean) / self.std
            
            # 3. 动态尺寸适配：计算当前尺寸最近的 14 的倍数
            # 这样无论训练(160)还是验证(1024+)，都能保证正确
            h, w = x_dino.shape[2], x_dino.shape[3]
            
            # 使用 math.ceil 向上取整，确保不丢失边缘信息
            target_h = int(math.ceil(h / 14) * 14)
            target_w = int(math.ceil(w / 14) * 14)
            
            # 如果尺寸不匹配（比如 160->168），则进行插值
            if h != target_h or w != target_w:
                x_dino = F.interpolate(x_dino, size=(target_h, target_w), 
                                      mode='bicubic', align_corners=False)

            # 4. 前向传播获取特征
            out = self.semantic_encoder.forward_features(x_dino)
            
            # 5. 获取Patch Tokens 并重排
            f_patch = out['x_norm_patchtokens']  # [B, N, 384]
            B, N, C = f_patch.shape
            
            # 计算 grid 大小
            grid_h = target_h // 14
            grid_w = target_w // 14
            
            # 重排为 [B, 384, grid_h, grid_w]
            f_semantic = f_patch.transpose(1, 2).reshape(B, C, grid_h, grid_w)
            
            # 6. 上采样回原始输入 x 的尺寸
            # 这样可以避免在每个SemanticInjector中重复插值
            f_semantic = F.interpolate(f_semantic, size=x.shape[2:], 
                                      mode='bilinear', align_corners=False)
            
        return f_semantic

    def forward(self, x):
        # 1. 提取语义特征 (全局上下文)
        f_semantic = self.get_semantic_feature(x)
        
        # 2. 战略决策 (Router): 三位一体感知，算出 4 个专家的权重
        # weights 形状为 [B, 4]
        weights = self.router(x, f_semantic)
        
        # 3. 基础特征提取 (Head)
        x_head = self.head(x)
        
        # 4. 运行 4 个独立的 BSN 专家单元 (Experts)
        # 注意：专家内部依然在执行原有的"战术级"语义注入
        o1 = self.bsn1(x_head, f_semantic)  # 返回 (e1, e2, ref)
        o2 = self.bsn2(x_head, f_semantic)
        o3 = self.bsn3(x_head, f_semantic)
        o4 = self.bsn4(x_head, f_semantic)

        # 5. 权重乘法 (Weighted Multiplication)
        # 定义一个内部辅助函数，将标量权重应用到专家的所有输出通道
        def apply_weight(expert_out, w_scalar):
            # 将权重 w_scalar [B] 扩展为 [B, 1, 1, 1] 
            w = w_scalar.view(-1, 1, 1, 1)
            return [feat * w for feat in expert_out]

        # 分别对 4 个专家的输出应用各自的权重
        o1_w = apply_weight(o1, weights[:, 0])
        o2_w = apply_weight(o2, weights[:, 1])
        o3_w = apply_weight(o3, weights[:, 2])
        o4_w = apply_weight(o4, weights[:, 3])

        # 6. 加权级联汇总 (Weighted Concat)
        # 将加权后的 12 路特征（4专家 * 3输出）拼接在一起
        out_cat = torch.cat([*o1_w, *o2_w, *o3_w, *o4_w], dim=1)
        
        # 7. 尾部重构输出最终去噪图
        return self.tail(out_cat)

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)


class DC_branchl(nn.Module):
    """带语义注入的Dilated Conv分支"""
    def __init__(self, stride, in_ch, mask_type, num_module, semantic_dim=384):
        super().__init__()

        # 掩膜卷积选择
        mask_conv_map = {
            'r': RowMaskedConv2d,
            'c': ColMaskedConv2d,
            'sz': SzMaskedConv2d,
            'fsz': fSzMaskedConv2d,
            'a45': angle45MaskedConv2d,
            'a135': angle135MaskedConv2d,
            'hui': huiMaskedConv2d,
            'cha': chaMaskedConv2d,
            'fcha': fchaMaskedConv2d,
        }
        
        ConvClass = mask_conv_map.get(mask_type, CentralMaskedConv2d)
        
        # 头部网络
        self.head = nn.Sequential(
            ConvClass(in_ch, in_ch, kernel_size=2*stride-1, stride=1, padding=stride-1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # 路径1的1x1卷积
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # 路径2的1x1卷积
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # 主体网络：交替的DCl模块和语义注入器
        self.body_layers = nn.ModuleList([
            DCl(stride, in_ch) for _ in range(num_module)
        ])
        
        self.injectors = nn.ModuleList([
            SemanticInjector(in_ch, sem_dim=semantic_dim) for _ in range(num_module)
        ])
        
        # 主体尾部
        self.body_tail = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # 特征融合
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, f_semantic=None):
        y0 = self.head(x)
        
        # 路径1：带语义注入的特征处理
        conv1_1 = self.conv1_1(y0)
        
        # 交替执行DCl卷积和语义注入
        feat = conv1_1
        for layer, injector in zip(self.body_layers, self.injectors):
            feat = layer(feat)  # DCl模块
            feat = injector(feat, f_semantic)  # 语义注入
        
        y1 = self.body_tail(feat)
        
        # 特征融合
        cat0 = torch.cat([conv1_1, y1], dim=1)
        conv1_3 = self.conv1_3(cat0)

        # 路径2：简单的1x1卷积
        conv1_2 = self.conv1_2(y0)

        return conv1_2, conv1_3


class DC_branchl2(nn.Module):
    """第二阶段的DCl模块（不带语义注入）"""
    def __init__(self, stride, in_ch, num_module):
        super().__init__()
        layers = [DCl(stride, in_ch) for _ in range(num_module)]
        layers.extend([
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.ReLU(inplace=True)
        ])
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class DCl(nn.Module):
    """Dilated Conv模块"""
    def __init__(self, stride, in_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1)
        )

    def forward(self, x):
        return x + self.body(x)