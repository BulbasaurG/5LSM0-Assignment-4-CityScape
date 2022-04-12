import torch.nn.functional as F
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DoubleConv,self).__init__()
        self.double_conv1 = nn.Conv2d(ch_in,ch_out,3,padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.double_conv2 = nn.Conv2d(ch_out,ch_out,3,padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

    def forward (self, x):
        double_conv1 = self.double_conv1(x)
        bn1 = self.bn1(double_conv1)
        double_conv2 = self.double_conv2(F.relu(bn1))
        bn2 = self.bn2(double_conv2)
        return F.relu(bn2)