from torch.utils.data import Dataset

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, image_x_list, image_y_list, transform=None): # 为什么是none？ 不是在设定好了再传进来吗？
        self.image_x_list = image_x_list
        self.image_y_list = image_y_list
        self.transform = transform

    def __len__(self):
        return len(self.image_x_list)

    def __getitem__(self, item):
        '''转三通道的RGB'''
        image_x = self.image_x_list[item].convert('RGB')
        image_y = self.image_y_list[item].convert('RGB')

        '''根据transform规则转成（256,256）的tensor'''
        image_x = self.transform(image_x)
        image_y = self.transform(image_y)

        # [0,1]转[-1,1]
        image_x = (image_x - 0.5).true_divide(0.5)
        image_y = (image_y - 0.5).true_divide(0.5)

        return image_x,image_y




