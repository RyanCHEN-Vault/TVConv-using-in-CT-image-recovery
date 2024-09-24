import os

image_root = r'/home/zhouzihao/TVConv/rename/data/'
dose_name = ['full','quarter']
patient_name = ['L067','L096','L109']

# /home/zhouzihao/TVConv/rename/data/dose/Lxxx_dose_3mm_index.png
# /home/zhouzihao/TVConv/rename/data/dose/sm_dose_3mm_index.png
x_root = image_root + dose_name[1] + '/'
y_root = image_root + dose_name[0] + '/'
for index in range(57,71): # 一个病人一个病人来重命名
    sp_index = index - 56  # 事先计算好该病人的图像序号
    src = x_root + patient_name[2] + '_' + dose_name[1] + '_3mm_' + '{:}'.format(sp_index) + '.png'
    dst = x_root + 'sm_' + dose_name[1] + '_3mm_' + '{:}'.format(index) + '.png'
    os.rename(src, dst)
    src = y_root + patient_name[2] + '_' + dose_name[0] + '_3mm_' + '{:}'.format(sp_index) + '.png'
    dst = y_root + 'sm_' + dose_name[0] + '_3mm_' + '{:}'.format(index) + '.png'
    os.rename(src, dst)
