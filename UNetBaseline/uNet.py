from UNetBaseline.doubleconv import DoubleConv
from torch import nn
import torch

class UNet (nn.Module):
    def __init__(self,ch_in,ch_out):
        super(UNet,self).__init__()
        # left side of UNet. Extracting low level feature
        self.conv1 = DoubleConv(ch_in, 64)
        self.pool1 = nn.MaxPool2d(2) # downsampling with factor=2 after each DoubleConv
        self.conv2 = DoubleConv(64,128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128,256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256,512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512,1024)

        # right side of UNet. Upsampling by transpose convolution.
        self.upsp6 = nn.ConvTranspose2d(1024,512,2,stride=2)
        self.conv6 = DoubleConv(1024,512)
        self.upsp7 = nn.ConvTranspose2d(512,256,2,stride=2)
        self.conv7 = DoubleConv(512,256)
        self.upsp8 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.conv8 = DoubleConv(256,128)
        self.upsp9 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.conv9 = DoubleConv(128,64)

        self.conv10 = nn.Conv2d(64,ch_out,1)

        # The energy function is computed by a pixel-wise soft-max over the final feature
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self,x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)
        upsp6 = self.upsp6(conv5)
        merg6 = torch.cat([upsp6,conv4],dim=1)
        conv6 = self.conv6(merg6)
        upsp7 = self.upsp7(conv6)
        merg7 = torch.cat([upsp7,conv3],dim=1)
        conv7 = self.conv7(merg7)
        upsp8 = self.upsp8(conv7)
        merg8 = torch.cat([upsp8,conv2],dim=1)
        conv8 = self.conv8(merg8)
        upsp9 = self.upsp9(conv8)
        merg9 = torch.cat([upsp9,conv1],dim=1)
        conv9 = self.conv9(merg9)
        conv10 = self.conv10(conv9)

        out = self.logsoftmax(conv10)
        return out