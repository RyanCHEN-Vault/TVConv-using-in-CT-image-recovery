import torch
from torch import nn
from torch.nn import functional as F
from .TVConv import TVConvInvertedResidual

'''卷积层'''
class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            # 卷积核 = 3X3, 步长 = 1，padding = 1, padding_model = 映射padding增加信息保存率
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),  # dropout操作防止过拟合
            nn.LeakyReLU(),

            # 用Conv2d执行pooling步骤
            # 池化下采样比较粗暴，可能将有用的信息滤除掉，而卷积下采样过程控制了步进大小，信息融合较好
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)

'''TVConv卷积层'''
class TVConv_Block(nn.Module):
    def __init__(self,in_channel,out_channel,h_w):
        super(TVConv_Block, self).__init__()
        self.layer=nn.Sequential(
            TVConvInvertedResidual(in_channel, out_channel, h_w, stride=1),
            # InvertedResidual(in_channel, out_channel, stride=1, h_w=16)

            # pooling by conv2d
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
        )
    def forward(self,x):
        return self.layer(x)

'''下采样层'''
class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            # 卷积操作代替max池化，以增加信息保存率
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)

'''上采样层'''
class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample, self).__init__()
        # channel//2输出量为输入的一半(channel/2后的整数)     1X1卷积核，步长为1   --> 降通道
        self.layer=nn.Conv2d(channel,channel//2,1,1)
    def forward(self,x,feature_map):
        # 最近邻插值法for填充扩大图像之后的空洞，scale_factor=2 --> h、w扩大2倍
        up=F.interpolate(x,scale_factor=2,mode='nearest')
        out=self.layer(up)
        # 跳跃链接：将该层输出和对应下采样层的输出（feature_map）进行拼接
        # dim=1：输出元组（n,channel,width,height），需要channel层的
        return torch.cat((out,feature_map),dim=1)


class UNet_tv(nn.Module):
    def __init__(self,num_classes):
        super(UNet_tv, self).__init__()
        '''c:卷积操作，d:下采样操作，u:上采样操作'''
        self.c1 = TVConv_Block(3,64,256)  # in_channel = 3, out_channel = 64
        self.d1 = DownSample(64)
        self.c2 = TVConv_Block(64,128,128)
        self.d2 = DownSample(128)
        self.c3 = TVConv_Block(128,256,64)
        self.d3 = DownSample(256)
        self.c4 = TVConv_Block(256,512,32)
        self.d4 = DownSample(512)

        self.c5 = TVConv_Block(512,1024,16)

        self.u1 = UpSample(1024)
        self.c6 = TVConv_Block(1024,512,32)
        self.u2 = UpSample(512)
        self.c7 = TVConv_Block(512,256,64)
        self.u3 = UpSample(256)
        self.c8 = TVConv_Block(256,128,128)
        self.u4 = UpSample(128)
        self.c9 = TVConv_Block(128,64,256)
        # num_classes = 3 要三通道的RGB图像, 卷积核3X3，步长=1，padding=1保持图像大小不变
        self.out = nn.Conv2d(64,num_classes,3,1,1)
        # self.Th = nn.Sigmoid()输出图像只需要两种颜色：黑->背景，另一种颜色->语义

    def forward(self,x):
        # U型图操作流程 --> R：U的左边，O：U的右边, O的输入是上一个O的输出cat上对应R的输出（跳跃链接）
        # print('x_shape:{:}'.format(x.shape))
        R1 = self.c1(x)
        # print('R1_shape:{:}'.format(R1.shape))
        R2 = self.c2(self.d1(R1))
        # print('R2_shape:{:}'.format(R2.shape))
        R3 = self.c3(self.d2(R2))
        # print('R3_shape:{:}'.format(R3.shape))
        R4 = self.c4(self.d3(R3))
        # print('R4_shape:{:}'.format(R4.shape))
        R5 = self.c5(self.d4(R4))
        # print('in  to   R5_shape:{:}'.format(self.d4(R4).shape))
        # print('out from R5_shape:{:}'.format(R5.shape))
        # print('in  to   O1_shape:{:}'.format(self.u1(R5,R4).shape))
        O1 = self.c6(self.u1(R5,R4))
        # print('O1_shape:{:}'.format(O1.shape))
        O2 = self.c7(self.u2(O1, R3))
        # print('O2_shape:{:}'.format(O2.shape))
        O3 = self.c8(self.u3(O2, R2))
        # print('O3_shape:{:}'.format(O3.shape))
        O4 = self.c9(self.u4(O3, R1))
        # print('O4_shape:{:}'.format(O4.shape))
        # exit(0)

        return self.out(O4)
        # return self.Th(self.out(O4)) #sigmoid后输出
