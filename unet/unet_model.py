# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        # down4
        self.maxd2f = nn.MaxPool2d(2,2)
        self.final1 = nn.Sequential(nn.Conv2d(256,512,3,1,1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True))
        self.avg = nn.AvgPool2d(7)
        self.final2 = nn.Sequential(nn.Conv2d(512,512,3,1,1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True))
        # self.down4 = down(256, 256)
        self.up1 = up(512, 256, bilinear=False)
        self.up2 = up(256, 128, bilinear=False)
        self.up3 = up(128, 64, bilinear=False)
        self.up4 = up(64, 32, bilinear=False)
        self.outc = outconv(32, n_classes)
        self.classfier = nn.Sequential(nn.Linear(5*5*512,2048),
                                       nn.Linear(2048,2))

    def forward(self, x): # x 640 * 640 * 3
        x1 = self.inc(x) # x1 640 * 640 * 32
        x2 = self.down1(x1) # x2 320 * 320 * 64
        x3 = self.down2(x2) # x3 160 * 160 * 128
        x4 = self.down3(x3) # x4 80 * 80 * 256
        temp = self.final1(self.maxd2f(x4))
        x5 = self.final2(temp)
        # x5 = self.down4(x4) # x5 40 * 40 * 512
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        temp = self.avg(temp).view(x.size(0), -1)
        temp = self.classfier(temp)
        # return temp,x
        # temp = nn.Softmax(dim=1)(temp)
        return temp,nn.Sigmoid()(x)

