from torch import nn
from torch.nn import functional as F
import torch


# 目的：为输入特征的每个通道分配一个权重，从而突出重要通道信息。
# 实现：首先，分别对输入做自适应平均池化（AdaptiveAvgPool3d）和最大池化（AdaptiveMaxPool3d），将空间尺寸压缩为 1×1×1，从而得到两个 [batch, channel, 1, 1, 1] 的描述。
# 接着，将这两种描述都输入到同一个共享的多层感知机（实际上是两个 1×1×1 的卷积层，中间有 ReLU 激活）中，将通道数先降为 channel // ratio 再恢复到原始通道数。
# 最后将两个输出相加，再经过 Sigmoid 激活，生成一个在 [0,1] 之间的通道注意力权重。
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=8):  #64->16  8->2
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv3d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv3d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm3d(out_channel),
            nn.Dropout3d(0.3),
            nn.ReLU(),
            nn.Conv3d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm3d(out_channel),
            nn.Dropout3d(0.3),
            nn.ReLU()
        )

    def forward(self,x):
        return self.layer(x)

class DownSampple(nn.Module):
    def __init__(self,channel):
        super(DownSampple,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels=channel,out_channels=channel,kernel_size=3,stride=2,padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm3d(channel),
            nn.ReLU()
        )

    def forward(self,x):
        return self.layer(x)

class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample,self).__init__()
        self.layer = nn.Conv3d(in_channels=channel,out_channels=channel//2,kernel_size=1,stride=1)

    def forward(self,x,feature_map):
        up = F.interpolate(x,scale_factor=2,mode='nearest')
        out = self.layer(up)
        return torch.cat((out,feature_map),dim=1)

class FeatureFusion_1(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(FeatureFusion_1,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=2,padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm3d(out_channel),
            nn.Dropout3d(0.3),
            nn.ReLU(),
        )

    def forward(self,x):
        return self.layer(x)

class FeatureFusion_2(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(FeatureFusion_2,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=4,padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm3d(out_channel),
            nn.Dropout3d(0.3),
            nn.ReLU(),
        )

    def forward(self,x):
        return self.layer(x)

class FeatureFusion_3(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(FeatureFusion_3,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=8,padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm3d(out_channel),
            nn.Dropout3d(0.3),
            nn.ReLU(),
        )

    def forward(self,x):
        return self.layer(x)

class AUNet(nn.Module):
    def __init__(self):
        super(AUNet,self).__init__()
        # 特征融合分支，利用FeatureFusion对输入x进行下采样，得到多尺度融合特征，这些特征将在后续阶段与编码器的特征相加，帮助网络获得更多层次信息
        self.f1 = FeatureFusion_1(1,128)
        self.f2 = FeatureFusion_2(1,256)
        self.f3 = FeatureFusion_3(1,512)

        self.c1 = Conv_Block(1,64)
        self.d1 = DownSampple(64)
        self.c2 = Conv_Block(64,128)
        self.d2 = DownSampple(128)
        self.c3 = Conv_Block(128,256)
        self.d3 = DownSampple(256)
        self.c4 = Conv_Block(256,512)
        self.d4 = DownSampple(512)
        self.c5 = Conv_Block(512,1024)

        self.cbam1 = CBAM(channel=64)
        self.cbam2 = CBAM(channel=128)
        self.cbam3 = CBAM(channel=256)
        self.cbam4 = CBAM(channel=512)

        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024,512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512,256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256,128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128,64)
        self.out = nn.Conv3d(in_channels=64,out_channels=1,kernel_size=3,stride=1,padding=1)
        self.Th = nn.Sigmoid()

    def forward(self,x):
        F1 = self.f1(x)
        F2 = self.f2(x)
        F3 = self.f3(x)

        R1 = self.c1(x)
        R1 = self.cbam1(R1) + R1

        R2 = self.c2(self.d1(R1))
        R2 = F1 + R2
        R2 = self.cbam2(R2) + R2

        R3 = self.c3(self.d2(R2))
        R3 = F2 + R3
        R3 = self.cbam3(R3) + R3

        R4 = self.c4(self.d3(R3))
        R4 = F3 + R4
        R4 = self.cbam4(R4) + R4
        R5 = self.c5(self.d4(R4))

        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.Th(self.out(O4))


if __name__ == '__main__':
    net = AUNet()
    dummy_input = torch.randn(1,1, 64, 64, 32, device='cpu')
    out = net(dummy_input)
    print(1)

