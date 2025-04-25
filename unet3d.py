from torch import nn
from torch.nn import functional as F
import torch

class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm3d(out_channel),
            nn.Dropout3d(0.3),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm3d(out_channel),
            nn.Dropout3d(0.3),
            nn.LeakyReLU()
        )

    def forward(self,x):
        return self.layer(x)

class DownSampple(nn.Module):
    def __init__(self,channel):
        super(DownSampple,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels=channel,out_channels=channel,kernel_size=3,stride=2,padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm3d(channel),
            nn.LeakyReLU()
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

class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        self.c1 = Conv_Block(1,8)
        self.d1 = DownSampple(8)
        self.c2 = Conv_Block(8,16)
        self.d2 = DownSampple(16)
        self.c3 = Conv_Block(16,32)
        self.d3 = DownSampple(32)
        self.c4 = Conv_Block(32,64)
        self.d4 = DownSampple(64)
        self.c5 = Conv_Block(64,128)

        self.u1 = UpSample(128)
        self.c6 = Conv_Block(128,64)
        self.u2 = UpSample(64)
        self.c7 = Conv_Block(64,32)
        self.u3 = UpSample(32)
        self.c8 = Conv_Block(32,16)
        self.u4 = UpSample(16)
        self.c9 = Conv_Block(16,8)
        self.out = nn.Conv3d(in_channels=8,out_channels=1,kernel_size=3,stride=1,padding=1)
        self.Th = nn.Sigmoid()

    def forward(self,x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))

        O1 = self.c6(self.u1(R5,R4))
        O2 = self.c7(self.u2(O1,R3))
        O3 = self.c8(self.u3(O2,R2))
        O4 = self.c9(self.u4(O3,R1))

        return self.Th(self.out(O4))

if __name__ == '__main__':
    net = UNet()
    dummy_input = torch.randn(1,1, 64, 64, 32, device='cpu')
    out = net(dummy_input)
    print(1)