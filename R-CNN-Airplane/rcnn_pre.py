import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision.models import vgg16
vgg = vgg16(pretrained=True)
# 图片文件夹和csv的文件夹
path = "Images"
annot = "Annotations"
class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.vgg = vgg
        # 下面都是线性分类层
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 10)
        self.fc5 = nn.Linear(10, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = vgg(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.softmax(self.fc5(x))
        return x
net = torch.load("rcnn_method1.th")
z = 0
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
for e1, i in enumerate(os.listdir(path)):
    # .  z==1为了早点结束
    if (z == 1):
        break
    if i.startswith("428483"):
        z += 1
        img = cv2.imread(os.path.join(path, i))
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()
        imout = img.copy()
        for e, result in enumerate(ssresults):
            # .  同样e==50为了早点结束
            if (e == 50):
                break
            if e < 2000:
                x, y, w, h = result
                timage = imout[y:y + h, x:x + w]
                resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                img = np.expand_dims(resized, axis=0)
                img = torch.from_numpy(img)
                img = img.transpose(3, 1)
                print(e, img.shape)
                out = net(img.to(torch.float32))
                print(f"out的形状{out}")
                if out[0][0] > 0.65:
                    cv2.rectangle(imout, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
        plt.figure()
        plt.imshow(imout)
        plt.show()