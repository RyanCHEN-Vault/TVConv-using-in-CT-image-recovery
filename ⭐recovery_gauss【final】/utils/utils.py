import argparse
import os

import cv2
import numpy as np
import torch
import torchmetrics
import statistics
import pandas as pd

def tensor2PNG(target_test_pred, image_x, image_y, number):
    '''tensor 转 numpy'''
    target_test_pred = target_test_pred.cpu().numpy()
    image_x = image_x.cpu().numpy()
    image_y = image_y.cpu().numpy()
    # target_test_pred = (target_test_pred[0] + 1) / 2.0*255.0
    target_test_pred = np.squeeze(target_test_pred, axis=0)
    image_x = np.squeeze(image_x, axis=0)
    image_y = np.squeeze(image_y, axis=0)

    image_pred = target_test_pred.transpose(1, 2, 0)
    image_x = image_x.transpose(1, 2, 0)
    image_y = image_y.transpose(1, 2, 0)

    '''[-1,1]转[0,255]'''
    image_pred = cv2.normalize(image_pred, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F)
    image_pred.astype(np.uint8)
    image_x = cv2.normalize(image_x, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_32F)
    image_x.astype(np.uint8)
    image_y = cv2.normalize(image_y, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_32F)
    image_y.astype(np.uint8)

    '''opencv'''
    previous_path = os.getcwd()
    image_pred = cv2.cvtColor(image_pred, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('{:}/result/pred_{:}.png'.format(previous_path, number/10), image_pred)
    image_x = cv2.cvtColor(image_x, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('{:}/result/ori_{:}.png'.format(previous_path,number/10), image_x)
    image_y = cv2.cvtColor(image_y, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('{:}/result/target_{:}.png'.format(previous_path,number/10), image_y)


class test_metrics(object):
    def __init__(self):
        self.loss_record_test = []
        self.loss_record_test_ori = []
        self.loss_record_test_avg = []
        self.loss_record_test_ori_avg = []

        self.RMSE_record_test = []
        self.RMSE_record_test_ori = []
        self.RMSE_record_test_avg = []
        self.RMSE_record_test_ori_avg = []

        self.SSIM_record_test = []
        self.SSIM_record_test_ori = []
        self.SSIM_record_test_avg = []
        self.SSIM_record_test_ori_avg = []

        self.PSNR_record_test = []
        self.PSNR_record_test_ori = []
        self.PSNR_record_test_avg = []
        self.PSNR_record_test_ori_avg = []

    def test_metrics_counting(self, target_test_pred, image_x, image_y, device):
        mseloss = torch.nn.MSELoss()
        ssim = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)  # data_range[0,1]
        psnr = torchmetrics.image.PeakSignalNoiseRatio().to(device)

        '''计算loss MSE'''
        loss = mseloss(target_test_pred, image_y)
        loss_test_ori = mseloss(image_x, image_y)

        loss_test = loss.item()
        loss_test_ori = loss_test_ori.item()

        self.loss_record_test.append(loss_test)
        self.loss_record_test_ori.append(loss_test_ori)

        '''计算均方根误差 RMSE'''
        loss = loss.clone().detach()  # 免得出Waring
        RMSE_test = torch.sqrt(loss)
        RMSE_test_ori = torch.sqrt(torch.tensor(loss_test_ori))

        RMSE_test = RMSE_test.item()
        RMSE_test_ori = RMSE_test_ori.item()

        self.RMSE_record_test.append(RMSE_test)
        self.RMSE_record_test_ori.append(RMSE_test_ori)

        '''计算结构相似性 SSIM'''
        SSIM_test = ssim(target_test_pred, image_y)
        SSIM_test_ori = ssim(image_x, image_y)

        SSIM_test = SSIM_test.item()
        SSIM_test_ori = SSIM_test_ori.item()

        self.SSIM_record_test.append(SSIM_test)
        self.SSIM_record_test_ori.append(SSIM_test_ori)

        '''计算峰值信噪比 PSNR'''
        PSNR_test = psnr(target_test_pred, image_y)
        PSNR_test_ori = psnr(image_x, image_y)

        PSNR_test = PSNR_test.item()
        PSNR_test_ori = PSNR_test_ori.item()

        self.PSNR_record_test.append(PSNR_test)
        self.PSNR_record_test_ori.append(PSNR_test_ori)

    def test_metrics_record(self):
        '''整合 求均值'''
        self.loss_record_test_avg.append(statistics.mean(self.loss_record_test))
        self.loss_record_test_ori_avg.append(statistics.mean(self.loss_record_test_ori))

        self.RMSE_record_test_avg.append(statistics.mean(self.RMSE_record_test))
        self.RMSE_record_test_ori_avg.append(statistics.mean(self.RMSE_record_test_ori))

        self.SSIM_record_test_avg.append(statistics.mean(self.SSIM_record_test))
        self.SSIM_record_test_ori_avg.append(statistics.mean(self.SSIM_record_test_ori))

        self.PSNR_record_test_avg.append(statistics.mean(self.PSNR_record_test))
        self.PSNR_record_test_ori_avg.append(statistics.mean(self.PSNR_record_test_ori))

        loss_data_test = pd.DataFrame(self.loss_record_test_avg)
        loss_data_test_ori = pd.DataFrame(self.loss_record_test_ori_avg)

        RMSE_data_test = pd.DataFrame(self.RMSE_record_test_avg)
        RMSE_data_test_ori = pd.DataFrame(self.RMSE_record_test_ori_avg)

        SSIM_data_test = pd.DataFrame(self.SSIM_record_test_avg)
        SSIM_data_test_ori = pd.DataFrame(self.SSIM_record_test_ori_avg)

        PSNR_data_test = pd.DataFrame(self.PSNR_record_test_avg)
        PSNR_data_test_ori = pd.DataFrame(self.PSNR_record_test_ori_avg)

        '''excel写入'''
        writer = pd.ExcelWriter('./result/loss_record.xlsx', mode='a')
        loss_data_test.to_excel(writer, 'test_loss', float_format='%.5f')
        loss_data_test_ori.to_excel(writer, 'test_loss_ori', float_format='%.5f')

        RMSE_data_test.to_excel(writer, 'test_RMSE', float_format='%.5f')
        RMSE_data_test_ori.to_excel(writer, 'test_RMSE_ori', float_format='%.5f')

        SSIM_data_test.to_excel(writer, 'test_SSIM', float_format='%.5f')
        SSIM_data_test_ori.to_excel(writer, 'test_SSIM_ori', float_format='%.5f')

        PSNR_data_test.to_excel(writer, 'test_PSNR', float_format='%.5f')
        PSNR_data_test_ori.to_excel(writer, 'test_PSNR_ori', float_format='%.5f')

        writer.save()
        writer.close()
