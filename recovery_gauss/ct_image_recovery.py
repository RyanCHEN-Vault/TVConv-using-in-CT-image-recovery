import torch
import argparse
from time import process_time
from network import *
from data_process.data_loader import data_loader
from process.train_val import train_val
from process.test import test

# weight_path = 'params/unet.pth'  # 权重地址

def parse_args():
    parser = argparse.ArgumentParser()

    # device
    parser.add_argument('--device', default='cuda:0')

    # data_set
    parser.add_argument('--patient_number', default=0, type=int, metavar='N',
                        help='serial number of data_resources: L096 by 0, L067 by 1, L109 by 2, all patients by 3')
    parser.add_argument('--st', default=3, type=int, metavar='N',
                        help='choose the depth of section thickness, including 3mm(3) and 1mm(1)')

    # train
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--lr','--learn_rate', default=1e-3, type=float)

    # model
    parser.add_argument('--model_name', default='NestedUNet_TV',
                        help='those model are available: UNet, UNet_TV, UNet_DW, NestedUNet, NestedUNet_TV, NestedUNet_DW')

    # parser.add_argument('--num_workers', default=0, type=int)

    config = parser.parse_args()
    return config

if __name__ == '__main__':
    config = vars(parse_args())
    print(config)

    # 0. device
    device = torch.device(config['device'])

    T1 = process_time()

    cuda = torch.cuda.is_available()
    torch.cuda.manual_seed(1337)

    # 1. data_set
    train_loader, val_loader, test_loader = data_loader(config)

    # 2. model
    if config['model_name'] == 'UNet':
        model = UNet(3)
    if config['model_name'] == 'UNet_TV':
        model = UNet_tv(3)
    if config['model_name'] == 'UNet_DW':
        model = UNet_dw(3)
    if config['model_name'] == 'NestedUNet':
        model = NestedUNet(3, 3, False)
    if config['model_name'] == 'NestedUNet_TV':
        model = NestedUNet_TV(3, 3, False)
    if config['model_name'] == 'NestedUNet_DW':
        model = NestedUNet_DW(3, 3, False)
    # num_classes = 3; input_channels = 3; deep_supervision = False（默认0-4层输出，0-1~0-3层不当输出层...U++图最顶上那一列只选最后一个当输出层）

    # model = nn.DataParallel(model)  # 多gpu跑程序才要用到这一句
    model.to(device)

    # 3.train_val
    train_val(config, train_loader, val_loader, model, device)
    model_best = torch.load('./model_best.pt')

    # 4.test
    model = model_best
    model.eval()
    test(test_loader, model, device)
    print("程序运行时间:{:.5f}分钟".format(T1/60))


