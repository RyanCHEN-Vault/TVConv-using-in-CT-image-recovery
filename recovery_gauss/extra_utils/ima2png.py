import pydicom
import numpy as np
import os
import cv2


# 下面getCtHU()是将.IMA文件转成HU形式
def getCtHU(dicm):
    '''直接传入dicm文件/IMA文件'''
    img = np.array(dicm.pixel_array).astype('int32')
    img[img == -2000.0] = 0
    Hu = np.array(img).astype('float64')
    RescaleIntercept = dicm.RescaleIntercept
    RescaleSlope = dicm.RescaleSlope
    if RescaleSlope != 1:
        Hu = Hu * RescaleSlope
    Hu += RescaleIntercept
    return Hu


# 下面的windowsLevelTransform()是将上面的HU转为numpy形式
def windowsLevelTransform(Hu, window, level):
    img = Hu
    min = level - float(window) * 0.5;
    max = level + float(window) * 0.5;  # 这个是CT窗口设置，相关问题百度或评论。下面调用这个函数时候，我拟定拟定窗口[-160,240]
    img[img < min] = min
    img[img > max] = max
    norm_ = (img - min) / window
    norm_.astype('float32')
    return norm_


ima_root = r'/mnt/nas/data/MayoData/'
png_root = r'/mnt/nas/zhouzihao/data/ct_image_png/'
# png_root = r'./ct_image_png/'

patient_name = ['L067','L096','L109','L143','L192','L286','L291','L310','L333','L506']
type_name = ['full_1mm','full_3mm','quarter_1mm','quarter_3mm']
# 下面遍历每个切片

for p_name in patient_name:
    for t_name in type_name:
        ima_dir = ima_root + p_name + '/' + t_name
        png_dir = png_root + p_name + '/' + t_name

        file_size = len(os.listdir(ima_dir))

        for index in range(1,file_size+1):
            source_name = p_name + '_' + t_name + '_{:}'.format(index)
            img = pydicom.read_file(ima_dir + '/' + source_name+'.IMA')  # .IMA文件
            img_hu = getCtHU(img)  # 由.IMA文件转换成HU形式的
            img_np = windowsLevelTransform(Hu=img_hu, window=400, level=40)  # 再由HU形式的转成.numpy形式的
            if not os.path.exists(png_dir):
                os.makedirs(png_dir)
            cv2.imwrite(png_dir + '/' + source_name + '.png', img_np * 255)  # 注意这里img_np要乘上255即img_np*255，不然保存起来的图片看起来不爽，一片黑

