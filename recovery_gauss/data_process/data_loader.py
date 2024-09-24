from .my_dataset import MyDataSet
import torch
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from torchvision import transforms
import numpy as np

'''完整数据地址'''
image_root = './home/zhouzihao/TVConv/gauss/data/'

def data_loader(config):
    # 1.数据导入与预处理
    if config['patient_number'] == 0:
        patient_name = ['L096']
    if config['patient_number'] == 1:
        patient_name = ['L067']
    if config['patient_number'] == 2:
        patient_name = ['L109']
    if config['patient_number'] == 3:
        patient_name = os.listdir(image_root)
    if config['patient_number'] == 4:
        patient_name = ['sm']

    if config['pic_number'] == 1:
        pic_depth = ['1mm']
    if config['pic_number'] == 3:
        pic_depth = ['3mm']

    dose_name = ['full', 'quarter']

    '''guass_noise'''
    # mean = 0
    # sigma = 20
    # gauss = np.random.normal(mean, sigma, (h,w,c))

    image_x = []
    image_y = []
    for pn in patient_name:
        for ln in pic_depth:
            x_root = image_root + dose_name[1]
            y_root = image_root + dose_name[0]
            for index in range(len(os.listdir(x_root))):
                image = Image.open(
                    x_root + '/' + pn + '_' + dose_name[1] + '_' + ln + '_{:}'.format(index + 1) + '.png')
                # image = image + gauss
                # image = np.clip(image, a_min=0, a_max=1)
                image_x.append(image.copy())
                image.close()

                image = Image.open(
                    y_root + '/' + pn + '_' + dose_name[0] + '_' + ln + '_{:}'.format(index + 1) + '.png')
                # image = image + gauss
                # image = np.clip(image, a_min=0, a_max=1)
                image_y.append(image.copy())
                image.close()
                # print(image_y(index).shape)
                # exit(0)


    # 8:1:1
    x_tv, x_test_list, y_tv, y_test_list = train_test_split(image_x, image_y, test_size=0.1, random_state=8)
    x_train_list, x_v_list, y_train_list, y_v_list = train_test_split(x_tv, y_tv, test_size=1/9, random_state=8)

    data_transform = transforms.Compose([transforms.Resize((256, 256)),
                                         transforms.ToTensor()])

    train_data_set = MyDataSet(image_x_list=x_train_list,
                               image_y_list=y_train_list,
                               transform=data_transform)
    val_data_set = MyDataSet(image_x_list=x_v_list,
                             image_y_list=y_v_list,
                             transform=data_transform)
    test_data_set = MyDataSet(image_x_list=x_test_list,
                              image_y_list=y_test_list,
                              transform=data_transform)

    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data_set,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=0)
    return train_loader, val_loader, test_loader