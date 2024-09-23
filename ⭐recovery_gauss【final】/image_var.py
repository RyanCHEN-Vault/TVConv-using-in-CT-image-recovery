from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np

data_root = '/home/zhouzihao/image_var/data/'
# 不同病人同部位的 -- 跨图像 小
image1 = Image.open(data_root + '4.png')
image2 = Image.open(data_root + '4_1.png')

# 单张图片 -- 内图像 大
cross_variance = []
image1_pix = []
intra_variance = []

h,w = image1.size

for i in tqdm(range(h), desc='i', position=0, leave=False):
    for j in range(h):
        pix1 = image1.getpixel((i,j))
        pix2 = image2.getpixel((i,j))
        dv = abs(pix1 - pix2)

        cross_variance.append(dv)
        image1_pix.append(pix1)

# for i in tqdm(range(512), desc='i', position=0, leave=False):
#     for j in range(512):
#         if i+j == 0:
#             dv = abs(image1_pix[0] - np.mean(image1_pix[1:]))
#         dv = image1_pix[i+j] - (sum(image1_pix[0:i+j-1]) + sum(image1_pix[i+j+1:]))/(len(image1_pix)-1)
#         intra_variance.append(dv)
for i in tqdm(range(len(image1_pix)), desc='i', position=0, leave=False):
    if i == 0:
        dv = abs(image1_pix[i] - sum(image1_pix[i+1:])/ (len(image1_pix) - 1))
    dv = abs(image1_pix[i] - (sum(image1_pix[0:i-1]) + sum(image1_pix[i+1:]))/(len(image1_pix)-1))
    intra_variance.append(dv)

list1 = [0,0,0,0,0,0,0,0,0,0]

for i in tqdm(range(len(intra_variance)), desc='i', position=0, leave=False):
    a = intra_variance[i]
    if a >= 25:
        if a >= 50:
            if a>= 75:
                if a>=100:
                    if a>=125:
                        if a>= 150:
                            if a>=175:
                                if a>=200:
                                    if a>=225:
                                        if a>=250:
                                            list1[9]=list1[9]+1
                                    else:
                                        list1[8]=list1[8]+1
                                else:
                                    list1[7]=list1[7]+1
                            else:
                                list1[6]=list1[6]+1
                        else:
                            list1[5] = list1[5]+1
                    else:
                        list1[4] = list1[4]+1
                else:
                    list1[3] = list1[3]+1
            else:
                list1[2] = list1[2]+1
        else:
            list1[1] = list1[1]+1
    else:
        list1[0] = list1[0]+1
print('intra_v')
print(list1)

list1 = [0,0,0,0,0,0,0,0,0,0]
for i in tqdm(range(len(cross_variance)), desc='i', position=0, leave=False):
    a = cross_variance[i]
    if a >= 25:
        if a >= 50:
            if a>= 75:
                if a>=100:
                    if a>=125:
                        if a>= 150:
                            if a>=175:
                                if a>=200:
                                    if a>=225:
                                        if a>=250:
                                            list1[9]=list1[9]+1
                                    else:
                                        list1[8]=list1[8]+1
                                else:
                                    list1[7]=list1[7]+1
                            else:
                                list1[6]=list1[6]+1
                        else:
                            list1[5] = list1[5]+1
                    else:
                        list1[4] = list1[4]+1
                else:
                    list1[3] = list1[3]+1
            else:
                list1[2] = list1[2]+1
        else:
            list1[1] = list1[1]+1
    else:
        list1[0] = list1[0]+1
print('cross_v')
print(list1)

# for k, g in pd.DataFrame.groupby(sorted(cross_variance), key=lambda x:x//10):
#     print('{}--{}: {}'.format(k*10, (k+1)*10-1, len(list(g))))

