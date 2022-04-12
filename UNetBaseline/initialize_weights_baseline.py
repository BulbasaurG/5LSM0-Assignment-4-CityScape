import numpy as np
from torch import nn

def initialize_weights_baseline(m):
  if isinstance(m, nn.Conv2d):
      nn.init.normal_(m.weight.data,0,np.sqrt(2/(m.in_channels*9)))
      if m.bias is not None:
          nn.init.normal_(m.bias.data,0,np.sqrt(2/(m.in_channels*9)))
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.normal_(m.weight.data,0,np.sqrt(2/(m.num_features*9)))
      nn.init.normal_(m.bias.data,0,np.sqrt(2/(m.num_features*9)))
  elif isinstance(m, nn.ConvTranspose2d):
      nn.init.normal_(m.weight.data,0,np.sqrt(2/(m.in_channels*9)))
      if m.bias is not None:
          nn.init.normal_(m.bias.data,0,np.sqrt(2/(m.in_channels*9)))