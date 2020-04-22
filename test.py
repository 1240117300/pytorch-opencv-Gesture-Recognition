# 程序中的datatrain.txt和datatest.txt中，要去掉rensor，否则读取文件时会报错
import torch
from cv2.cv2 import imread
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import torch.nn as nn
import torch.utils.data as Data
import torchvision

root = "./image"
LR = 0.001
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 64)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 48)


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        fh = open(txt, 'r')  # 以只读的方式打开txt文件
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            imgs.append((words[0], int(words[1])))  # #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # 根据两个txt的内容，words[0]是图片位置信息，words[1]是图片标签信息
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):  # 按照索引方式读取
        fn, label = self.imgs[index]  # fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = self.loader(fn)  # 将图片转为RGB格式
        if self.transform is not None:
            img = self.transform(img)  # 转换图片格式，由第41行代码的transform可知将图片转为Tensor格式
        return img, label

    def __len__(self):
        return len(self.imgs)


train_data = MyDataset(txt="test.txt", transform=transforms.ToTensor())
# 程序中直接用opencv调用测试图像，因此不需要再单独加载测试数据
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 5, 1, 2),  # 最后一个参数要满足=（第二个参数-1）/2，这样默认图片的长宽不变，只是改变通道数
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(32 * 30 * 40, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 6),
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        res = conv2_out.view(conv2_out.size(0), -1)
        out = self.dense(res)
        return out


model = Net()
print(model)

# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# loss_func = torch.nn.CrossEntropyLoss()
#
# for epoch in range(20):
#     print('epoch {}'.format(epoch + 1))
#     # training-----------------------------
#     # train_loss = 0.
#     # train_acc = 0.
#     for batch_x, batch_y in train_loader:
#         batch_x, batch_y = Variable(batch_x), Variable(batch_y)
#         out = model(batch_x)
#         loss = loss_func(out, batch_y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
# torch.save(model.state_dict(), 'model.pkl')  # 保存模型

model.load_state_dict(torch.load('model.pkl'))  # 加载模型
model.eval()
while 1:
    ret, frame = cap.read()
    gray = frame
    img_to_tensor = transforms.Compose([
        transforms.ToTensor(),
    ]
    )  # 将opencv读取的图像转为tensor格式
    tensor = img_to_tensor(gray)
    # print(tensor.shape)  # 数据大小为【3，28，28】
    inputs = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)  # 改变数据维度
    # print(inputs.shape)  # 数据大小为【1，3，28，28】
    test_output = model(inputs)
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    print(pred_y, 'prediction number')
    cv2.imshow("frame", gray)
    key_pressed = cv2.waitKey(5)
