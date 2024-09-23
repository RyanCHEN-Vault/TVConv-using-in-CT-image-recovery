import torch
from torch import nn
from .TVConv import  InvertedResidual


'''VGG：连续的小卷积核代替一个大的卷积核，不改变感受野的情况下减少参数量'''
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

'''DWConv卷积层'''
class DWConv_Block(nn.Module):
    def __init__(self,in_channel,out_channel,h_w):
        super(DWConv_Block, self).__init__()
        self.layer=nn.Sequential(
            InvertedResidual(in_channel, out_channel, h_w, stride=1)
        )
    def forward(self,x):
        return self.layer(x)

class NestedUNet_DW(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        feature_hw = [256, 128, 64, 32, 16]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DWConv_Block(input_channels, nb_filter[0], feature_hw[0])
        self.conv1_0 = DWConv_Block(nb_filter[0], nb_filter[1], feature_hw[1])
        self.conv2_0 = DWConv_Block(nb_filter[1], nb_filter[2], feature_hw[2])
        self.conv3_0 = DWConv_Block(nb_filter[2], nb_filter[3], feature_hw[3])

        self.conv4_0 = DWConv_Block(nb_filter[3], nb_filter[4], feature_hw[4])

        # self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        # self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_1 = DWConv_Block(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = DWConv_Block(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = DWConv_Block(nb_filter[2]+nb_filter[3], nb_filter[2], feature_hw[2])
        self.conv3_1 = DWConv_Block(nb_filter[3]+nb_filter[4], nb_filter[3], feature_hw[3])

        # self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_2 = DWConv_Block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = DWConv_Block(nb_filter[1]*2+nb_filter[2], nb_filter[1], feature_hw[1])
        self.conv2_2 = DWConv_Block(nb_filter[2]*2+nb_filter[3], nb_filter[2], feature_hw[2])

        self.conv0_3 = DWConv_Block(nb_filter[0]*3+nb_filter[1], nb_filter[0], feature_hw[0])
        self.conv1_3 = DWConv_Block(nb_filter[1]*3+nb_filter[2], nb_filter[1], feature_hw[1])

        self.conv0_4 = DWConv_Block(nb_filter[0]*4+nb_filter[1], nb_filter[0], feature_hw[0])

        if self.deep_supervision:  # default: False （不启用）
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        # print('into x0_0:',input.shape)
        x0_0 = self.conv0_0(input)

        # print('into x1_0:',self.pool(x0_0).shape)
        x1_0 = self.conv1_0(self.pool(x0_0))

        # print('into x0_1:',torch.cat([x0_0, self.up(x1_0)], 1).shape)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        # print('into x2_0:',self.pool(x1_0).shape)
        x2_0 = self.conv2_0(self.pool(x1_0))

        # print('into x1_1:',torch.cat([x1_0, self.up(x2_0)], 1).shape)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))

        # print('into x0_2:',torch.cat([x0_0, x0_1, self.up(x1_1)], 1).shape)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        # print('into x3_0:',self.pool(x2_0).shape)
        x3_0 = self.conv3_0(self.pool(x2_0))

        # print('into x2_1:',torch.cat([x2_0, self.up(x3_0)], 1).shape)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))

        # print('into x1_2:',torch.cat([x1_0, x1_1, self.up(x2_1)], 1).shape)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))

        # print('into x0_3:',torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1).shape)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        output = self.final(x0_3)
        return output
