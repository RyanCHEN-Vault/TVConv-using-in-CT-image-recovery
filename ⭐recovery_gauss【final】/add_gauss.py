from tqdm import tqdm

import os

import numpy as np
import cv2

image_root = '/home/zhouzihao/TVConv/rename/data/'

dose_name = ['full', 'quarter']

'''guass_noise'''

image_x = []
image_y = []

pic_depth = ['3mm']
patient_name = ['sm']


mean = 0
sigma = 50
gauss = np.random.normal(mean, sigma, (512,512,1))

for pn in patient_name:
    for ln in pic_depth:
        x_root = image_root + dose_name[1]
        y_root = image_root + dose_name[0]
        for index in tqdm(range(len(os.listdir(x_root))), desc='Pic_processing', ncols=100, position=0 ,leave=True):
            image = cv2.imread(
                x_root + '/' + pn + '_' + dose_name[1] + '_' + ln + '_{:}'.format(index + 1) + '.png')
            image = image + gauss
            image = np.clip(image, a_min=0, a_max=255)
            cv2.imwrite('/home/zhouzihao/TVConv/gauss/data/quarter/sm_quarter_3mm_{:}.png'.format(index),image)


            image = cv2.imread(
                y_root + '/' + pn + '_' + dose_name[0] + '_' + ln + '_{:}'.format(index + 1) + '.png')
            image = image + gauss
            image = np.clip(image, a_min=0, a_max=255)
            cv2.imwrite('/home/zhouzihao/TVConv/gauss/data/full/sm_full_3mm_{:}.png'.format(index),image)