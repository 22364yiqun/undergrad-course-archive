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
        self.head = nn.Linear(8,9) 

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
    model=blocks(depth=option['depth'],heads=option['heads'])
    return model