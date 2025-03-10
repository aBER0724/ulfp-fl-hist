from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(1, 16, 8, 2, padding=2), # 第一层 卷积层(输入通道数,输出通道数,卷积核大小,步长,填充)
                                nn.ReLU(), # 第一层 ReLU激活函数 大于0的数据不变，小于0的数据变成0
                                nn.MaxPool2d(2, 1), # 第一层 池化层 (窗口大小,步长) 取最大值
                                nn.Conv2d(16, 32, 4, 2), # 第二层 卷积层
                                nn.ReLU(), # 第二层 ReLU激活函数
                                nn.MaxPool2d(2, 1), # 第二层 池化层
                                nn.Flatten(), # 展开为一维向量
                                nn.Linear(32 * 4 * 4, 32), # 全连接层
                                nn.ReLU(), # ReLU激活函数
                                nn.Linear(32, 10)) # 全连接层

    def forward(self,x):
        x=self.conv(x)
        return x

# ULDP
class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
