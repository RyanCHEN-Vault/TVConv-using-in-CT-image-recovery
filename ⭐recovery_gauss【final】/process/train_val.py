import os.path

import torch
from tqdm import tqdm
import pandas as pd
import statistics
import os

def train_val(config, train_loader, val_loader, model, device):
    # train and val
    previous_path = os.getcwd()
    # 3. optimizer and loss
    optim = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        betas=(0.9, 0.99)
    )
    # optim = torch.optim.Adam(model.parameters())  # 源代码的优化函数设置
    mseloss = torch.nn.MSELoss()  # 源代码用的是nn.CrossEntropyLoss()

    # =============================================================================================
    # 训练模型
    # =============================================================================================
    '''训练集测试模型'''
    loss_record_val = []
    loss_record_val_ori = []

    loss_record_val_avg = []
    loss_record_val_ori_avg = []

    loss_best_val = 100
    print('training begun')
    n_Epoch = config['epochs']
    for e in tqdm(range(0, n_Epoch), desc='Training', ncols=100, position=0 ,leave=True):
        for index, (image_x, image_y) in enumerate(train_loader):  # 每次1 batch, 转完整个Epoch
            model.train()
            '''数据提取'''
            image_x = image_x.to(device)
            image_y = image_y.to(device)

            '''数据录入并获取预测值'''
            target_train_pred = model(image_x)

            '''计算loss'''
            loss = mseloss(target_train_pred, image_y)

            '''权重更新'''
            loss.backward()
            optim.step()

            '''梯度清零'''
            optim.zero_grad()
            # '''提醒'''
            # print('{:} epoch | {:} batch'.format(e,batch))
        '''loss item化'''
        # loss_train = loss.item()
        '''记录该epoch的loss_train'''
        # loss_record_train.append(loss_train)

        # =============================================================================================
        # 验证模型
        # =============================================================================================
        model.eval()
        # print('{:} batch | velrify begun'.format(e))
        with torch.no_grad():  # model不进行梯度计算，加速程序
            for index, (image_x, image_y) in enumerate(val_loader):  # 一整个包的数据都用上 ... 只跑一个次
                '''数据提取'''
                image_x = image_x.to(device)
                image_y = image_y.to(device)
                '''数据录入并获取预测值'''
                target_val_pred = model(image_x)

                '''计算loss'''
                loss = mseloss(target_val_pred, image_y)
                loss_ori = mseloss(image_x, image_y)

                '''loss item化并记录'''
                loss_val = loss.item()
                loss_val_ori = loss_ori.item()

                loss_record_val.append(loss_val)
                loss_record_val_ori.append(loss_val_ori)

            loss_avg = statistics.mean(loss_record_val)
            loss_ori_avg = statistics.mean(loss_record_val_ori)

            loss_record_val = []
            loss_record_val_ori = []

            loss_record_val_avg.append(loss_avg)
            loss_record_val_ori_avg.append(loss_ori_avg)

            '''更新loss_best_val'''
            if loss_best_val > loss_avg:
                loss_best_val = loss_avg
                torch.save(model, '{:}/model_best.pt'.format(previous_path))
                # print('updating model from epoch = {:}'.format(e))

    loss_data_val = pd.DataFrame(loss_record_val_avg)
    loss_data_val_ori = pd.DataFrame(loss_record_val_ori_avg)

    # excel记录
    writer = pd.ExcelWriter('./result/loss_record.xlsx')
    loss_data_val.to_excel(writer, 'val_loss', float_format='%.5f')
    loss_data_val_ori.to_excel(writer, 'val_loss_ori', float_format='%.5f')
    writer.save()
    writer.close()

    return print('training completed')