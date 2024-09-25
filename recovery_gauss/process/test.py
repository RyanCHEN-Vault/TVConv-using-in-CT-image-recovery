import os

import torch
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from utils.utils import tensor2PNG

def test(test_loader, model, device):
    from utils.utils import test_metrics
    test_metrics = test_metrics()
    number = 0
    with torch.no_grad():  # model不进行梯度计算，加速程序
        for index, (image_x, image_y) in enumerate(test_loader, 1):
            number += 1
            '''数据提取'''
            image_x = image_x.to(device)
            image_y = image_y.to(device)

            '''数据录入并获取预测值'''
            target_test_pred = model(image_y)

            '''test指标计算'''
            test_metrics.test_metrics_counting(target_test_pred, image_x, image_y, device)

            '''图片（矩阵）处理'''
            tensor2PNG(target_test_pred, image_x, image_y, number)

    '''test指标记录'''
    test_metrics.test_metrics_record()

    return print('testing over')