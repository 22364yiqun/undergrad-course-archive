import torch
from torch import nn
 
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
 
 
def pair(t):
    return t if isinstance(t, tuple) else (t, t)
 
 
class PreNorm(nn.Module):
    # 在执行fn之前执行一个Layer Norm
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
 
 
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # 前馈神经网络 = 2个全连接层
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
 
    def forward(self, x):
        return self.net(x)
 
 
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
 
        self.heads = heads
        self.scale = dim_head ** -0.5   # 缩放因子
 
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
 
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
 
    def forward(self, x):
        # x: [bs, 197, 1024]   197 = 1个Cls + 196个patch  1024就是每一个patch需要转为1024长度的向量
        # self.to_qkv(x)将x向量映射到长度为1024*3
        # chunk: qkv 最后是一个元祖，tuple，长度是3，每个元素形状：[1, 197, 1024]
        # 直接用x配合一个Linear生成qkv，再切分为3块
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # 再把qkv分别拆分开来
        # q: [1, 16, 197, 64]  k: [1, 16, 197, 64]  v: [1, 16, 197, 64]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # q * k转置 除以根号d_k
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # softmax得到每个token对于其他token的attention系数
        attn = self.attend(dots)
        # * v  [1, 16, 197, 64]
        out = torch.matmul(attn, v)
        # [1, 197, 1024]
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
 
 
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):  # 堆叠多个Encoder  depth个
            self.layers.append(nn.ModuleList([
                # 每个encoder = Attention(Multi-Head Attention) + FeedForward(MLP)
                # PreNorm：指在fn(Attention/FeedForward)之前执行一个Layer Norm
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
 
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    
import torch
import torch.nn as nn
from operator import itemgetter
from timm.models.layers import DropPath

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_channels,  # 每个通道独立卷积
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )

    def forward(self, x):
        x = self.depthwise(x)  # 深度卷积
        x = self.pointwise(x)  # 逐点卷积
        return x
    
class HSwish(nn.Module):   # 新型激活函数
    def forward(self, x):
        function = nn.ReLU6()
        return x * function(x + 3) / 6

class Sequential(nn.Module):  
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g ,k in self.blocks:
            x = x + f(x)
            x = x + g(x)
            x = x + k(x)
        return x

def calculate_permutations(num_dimensions):  # num_dimensions=3
    total_dimensions = num_dimensions + 1
    axial_dims = [ind for ind in range(1, total_dimensions)]

    permutations = []

    for axial_dim in axial_dims:
        last_one_dims = [axial_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_one_dims)
        permutation = [*dims_rest, *last_one_dims]   # 调换位置
        permutations.append(permutation) 
      
    return permutations



def pool_function(tmp):
    function = nn.AdaptiveAvgPool2d(16)
    tmp = function(tmp)
    return tmp

# 升维到通道数为 8
class Conv2d_BN(nn.Module):
    def __init__(self, in_features, out_features=None, kernel_size=3, stride=1, padding=0, dilation=1,
                groups=1,bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_features)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class InvertResidualBlock(nn.Module):  # 反向残差块 提取特征
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_path=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features

        # self.pwconv1_bn = Conv2d_BN(self.in_features, self.hidden_features, kernel_size=1,  stride=1, padding=0)
        self.dwconv_bn = Conv2d_BN(self.in_features, self.hidden_features, kernel_size=3,  stride=1, padding=1)
        self.pwconv2_bn = Conv2d_BN(self.hidden_features, self.out_features, kernel_size=1,  stride=1, padding=0)
        # 深度可分离卷积
        self.act = nn.ReLU(inplace=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    # @line_profile
    def forward(self, x):
        # x1 = self.pwconv1_bn(x)
        # x1 = self.act(x)
        x1 = self.dwconv_bn(x)
        x1 = self.act(x1)
        x1 = self.pwconv2_bn(x1)

        return x + x1
    
class AverageGlobslPoolingBranch(nn.Module):
    def __init__(self,in_channels,avg_pool_kernel_size=2):
        super().__init__()
        self.avg_pool_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=avg_pool_kernel_size, stride=avg_pool_kernel_size),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            # DepthwiseSeparableConv(in_channels, in_channels, kernel_size=1), # 可选：对通道进行卷积操作
            HSwish()
        )
        
        # 全局平均池化分支
        self.global_avg_pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            # DepthwiseSeparableConv(in_channels, in_channels, kernel_size=1),  # 可选：对通道进行卷积操作
            HSwish()
        )

    def forward(self, x):
        # 平均池化分支
        x_avg = self.avg_pool_branch(x)  # 形状将缩小，例如从 (B, C, H, W) -> (B, C, H/2, W/2)

        # 全局平均池化分支
        x_global_avg = self.global_avg_pool_branch(x)  # 形状将变为 (B, C, 1, 1)
        x_global_avg = x_global_avg.expand(-1, -1, x_avg.size(2), x_avg.size(3))  # 扩展形状以便与 avg 分支相加

        # 合并两个分支
        out = x_avg + x_global_avg  # 按元素相加
        return out
    
class CNNblock(nn.Module):
    def __init__(self,in_channels=8,hidden_channels=32,out_channels=8):
        super().__init__()
        self.in_channels = in_channels
        # self.conv1 = nn.Conv2d(in_channels,hidden_channels,kernel_size=1)
        # self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels,out_channels,kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self,x):
        identity = x 
        
        # out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(x)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        return out
    
class CNNBranch(nn.Module):
    def __init__(self,depth,in_channels=8,hidden_channels=32,out_channels=8):
        super().__init__()
        self.functions = nn.ModuleList()
        for _ in range(depth):
            function = CNNblock(in_channels,hidden_channels,out_channels)
            self.functions.append(function)

    def forward(self,x):
        for block in self.functions:
            x = block(x)
        return x
 
 
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)  # 32*32
        patch_height, patch_width = pair(patch_size)  # 4*4

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)  # 得到多少个token
        patch_dim = channels * patch_height * patch_width  # 1x4x4 = 16  patch展平后的维度

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),  # 把所有的patch拉平->16维
            nn.Linear(patch_dim, dim),  # 映射到encoder需要的维度16->1024
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 生成所有token和Cls的位置编码
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 生成Cls的初始化参数
        self.dropout = nn.Dropout(emb_dropout)  # embedding后面一般会接的一个Dropout

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)  # encoder
        self.pool = pool
        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(  # CLS多分类输出部分
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes))


        self.IRblock = InvertResidualBlock(1,4,8)
 
        self.norm = nn.LayerNorm(8, eps=1e-6)
        self.head = nn.Linear(8,9) 

        self.pooling = AverageGlobslPoolingBranch(in_channels=8)    
        self.CNN = CNNBranch(depth,8)

    def forward(self, img, return_feature=False):
        # img: [1, 1, 32, 32] x = [1, 64, 1024]
        # 生成每张图片的Patch Embedding
        x1 = self.to_patch_embedding(img)
        b, n, _ = x1.shape  # b = 1   n = 64

        # 为每张图片生成一个Cls符号 [1, 1, 1024]
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)

        # [1, 65, 1024] 将每张图片的Cls符号和Patch Embedding进行拼接
        x1 = torch.cat((cls_tokens, x1), dim=1)

        # 初始化位置编码 再和(Cls和Patch Embedding)对应位置相加
        x1 += self.pos_embedding[:, :(n + 1)]
        x1 = self.dropout(x1)

        # 将最终的Embedding输入Encoder  x: [1, 65, 1024] -> [1, 65, 1024]
        x1 = self.transformer(x1)

        # self.pool = 'cls' 所以取第一个输出直接进行多分类 [1, 1024]
        x1 = x1.mean(dim = 1) if self.pool == 'mean' else x1[:, 0]
        x1 = self.to_latent(x1)  # 恒等映射 [1, 1024]

        # Cls Head 多分类 [1, cls_num]

        x2 = self.IRblock(img)  #(1,8,32,32)

        x2 = self.CNN(x2)
        x2 = self.pooling(x2)

        x2 = self.norm(x2.mean([-2,-1]))
        out = torch.cat((x2,x1),dim=1)
        out = self.head(x2)
        
        feature = out.view(out.size(0),-1)

        if return_feature:
            return feature, out
        else:
            return out
        
class blocks(nn.Module):
    def __init__(self,dim=1,hidden_features=4,out_features=8,dim_heads=1,
                 depth=1,heads=8,axial_pos_emb_shape=(8,16,16),
                 reversible=False):
        super().__init__()
        permutations = calculate_permutations(3)
        self.dim = dim
        self.in_features = dim
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.IRblock = InvertResidualBlock(self.in_features,self.hidden_features,self.out_features)
 
        self.norm = nn.LayerNorm(8, eps=1e-6)
        self.head = nn.Linear(8,10) 

        self.pooling = AverageGlobslPoolingBranch(in_channels=out_features)    
        self.CNN = CNNBranch(depth,out_features)

    def forward(self,x,return_feature=False):
        x = self.IRblock(x)  #(1,8,32,32)

        x2 = self.CNN(x)
        x2 = self.pooling(x2)

        x2 = self.norm(x2.mean([-2,-1]))
        out = self.head(x2)

        feature = out.view(out.size(0),-1)
        if return_feature:
            return feature,out
        else:
            return out

     
def my_own_model(option):
    model=ViT(image_size=32,  # 输入图像的大小
        patch_size=4,  # 每个token/patch的大小16x16
        num_classes=9,  # 多分类
        dim=16,  # encoder规定的输入的维度
        depth=6,  # Encoder的个数
        heads=16,  # 多头注意力机制的head个数
        mlp_dim=64,  # mlp的维度
        dropout=0.1,  #
        emb_dropout=0.1  # embedding一半会接一个dropout
        )
    return model