from PIL import Image
from torchvision import transforms
import torch
from network.unet import UpSample,DownSample,Conv_Block
device = torch.device('cuda:0')
'''加载图片和model'''
image = Image.open('/home/zhouzihao/image_var/data/L067_full_3mm_3.png')
image = image.convert('RGB')

transforms = transforms.Compose([transforms.Resize((256, 256)),
                                         transforms.ToTensor()])


# 图片需要经过一系列数据增强手段以及统计信息(符合ImageNet数据集统计信息)的调整，才能输入模型
image = transforms(image)
# print(f"Image shape before: {image.shape}")
image = (image - 0.5).true_divide(0.5)
image = image.unsqueeze(0)
# print(f"Image shape after: {image.shape}")

image = image.to(device)

model = torch.load('./model_best.pt')
model.to(device)

fmap_block = []
input_block = []

def forward_hook(module, data_input, data_output):
    fmap_block.append(data_output)
    input_block.append(data_input)

model_children = list(model.children())

model.c1.register_forward_hook(forward_hook)
model.c2.register_forward_hook(forward_hook)
model.c3.register_forward_hook(forward_hook)
model.c4.register_forward_hook(forward_hook)
model.c5.register_forward_hook(forward_hook)
model.c6.register_forward_hook(forward_hook)
model.c7.register_forward_hook(forward_hook)
model.c8.register_forward_hook(forward_hook)
model.c9.register_forward_hook(forward_hook)

with torch.no_grad():
    outputs = model(image)

    # for i in range(len(fmap_block)):
    #     print('第{:}层的特征图大小：{:}'.format(i,fmap_block[i].shape))


processed = []
for feature_map in fmap_block:
    feature_map = feature_map.squeeze(0)  # torch.Size([1, 64, 112, 112]) —> torch.Size([64, 112, 112])  去掉第0维 即batch_size维
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]  # torch.Size([64, 112, 112]) —> torch.Size([112, 112])   从彩色图片变为黑白图片  压缩64个颜色通道维度，否则feature map太多张
    processed.append(gray_scale.data.cpu().numpy())  # .data是读取Variable中的tensor  .cpu是把数据转移到cpu上  .numpy是把tensor转为numpy


'''tensor 转 numpy'''
import numpy as np
import cv2


for i in range(len(processed)):   # len(processed) = 17
    '''[-1,1]转[0,255]'''

    processed[i] = cv2.normalize(processed[i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F)
    processed[i].astype(np.uint8)
    cv2.imwrite('./result_ex/{:}.png'.format(i), processed[i])



'''FLAG'''
# processed = []
# for feature_map in fmap_block:
#     feature_map = feature_map.squeeze(0)  # torch.Size([1, 64, 112, 112]) —> torch.Size([64, 112, 112])  去掉第0维 即batch_size维
#     gray_scale = torch.sum(feature_map,0)
#     gray_scale = gray_scale / feature_map.shape[0]  # torch.Size([64, 112, 112]) —> torch.Size([112, 112])   从彩色图片变为黑白图片  压缩64个颜色通道维度，否则feature map太多张
#     processed.append(gray_scale.data.cpu().numpy())  # .data是读取Variable中的tensor  .cpu是把数据转移到cpu上  .numpy是把tensor转为numpy
#
# # for fm in processed:
# #     print(fm.shape)
#
# import cv2
# for i in range(len(processed)):   # len(processed) = 17
#     cv2.imwrite('./result/{:}.png'.format(i), processed[i])