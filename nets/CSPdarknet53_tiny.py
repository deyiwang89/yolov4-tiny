import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from collections import OrderedDict

import numpy as np
from torch.nn import init

class rank(nn.Module):
    def __init__(self, channels = 512):
        self.k = channels
        super().__init__()
    def forward(self, y):
        y_shape = list(y.size())#[k,bs,channel,1,1]
        y_sub = y
        top_k = self.k//2

        filter_value = float(0.75)
        magnify_value = float(1.25)

        pred = y_sub        
        pred = pred.view(-1,y_shape[1])# bs channel

        indices_to_remove = pred < torch.topk(pred, top_k,dim = 1)[0][..., -1, None]
        pred[indices_to_remove] = filter_value*pred[indices_to_remove]
        pred[(indices_to_remove)==False] = magnify_value*pred[(indices_to_remove)==False]
        return torch.tensor(pred, dtype=None, device=None, requires_grad=False)

class External_attention(nn.Module):
    '''
    Arguments:
        c (int): The input and output channel number. 官方的代码中设为512
    '''
    def __init__(self, c):
        super(External_attention, self).__init__()
        self.conv1 = nn.Conv2d(c, c, 1)
        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)       

        self.rank = rank(channels=c) 
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.ReLU())        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
 
    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        n = h*w
        x = x.view(b, c, h*w)   # b * c * n 

        attn = self.linear_0(x) # b, k, n
        #linear_0是第一个memory unit
        attn = F.softmax(attn, dim=-1) # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True)) #  # b, k, n
        
        x = self.linear_1(attn) # b, c, n
        x = self.rank(x)
        #linear_1是第二个memory unit
        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x)
        return x
#-------------------------------------------------#
#   卷积块
#   Conv2d + BatchNorm2d + LeakyReLU
#-------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

         
'''
                    input
                      |
                  BasicConv
                      -----------------------
                      |                     |
                 route_group              route
                      |                     |
                  BasicConv                 |
                      |                     |
    -------------------                     |
    |                 |                     |
 route_1          BasicConv                 |
    |                 |                     |
    -----------------cat                    |
                      |                     |
        ----      BasicConv                 |
        |             |                     |
      feat           cat---------------------
                      |
                 MaxPooling2D
'''
#---------------------------------------------------#
#   CSPdarknet53-tiny的结构块
#   存在一个大残差边
#   这个大残差边绕过了很多的残差结构
#---------------------------------------------------#
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body, self).__init__()
        self.out_channels = out_channels

        self.conv1 = BasicConv(in_channels, out_channels, 3)

        self.conv2 = BasicConv(out_channels//2, out_channels//2, 3)
        self.conv3 = BasicConv(out_channels//2, out_channels//2, 3)

        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2,2],[2,2])


    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1(x)
        # 引出一个大的残差边route
        route = x
        
        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c//2, dim = 1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x,route1], dim = 1) 

        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)

        feat = x
        x = torch.cat([route, x], dim = 1)
        
        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x,feat

class CSPDarkNet(nn.Module):
    def __init__(self):
        super(CSPDarkNet, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 =  Resblock_body(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 =  Resblock_body(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 =  Resblock_body(256, 256)
        # 13,13,512 -> 13,13,512
        self.conv3 = BasicConv(512, 512, kernel_size=3)
        
        self.exa = External_attention(c = 512)

        self.num_features = 1
        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _    = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _    = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1    = self.resblock_body3(x)

        # 13,13,512 -> 13,13,512
        x = self.conv3(x)

        # bs,ch,w,h = x.shape
        # x = x.view(bs,-1,ch)
        x = self.exa(x)
        # x = x.view(bs,ch,w,h)

        feat2 = x
        return feat1,feat2

def darknet53_tiny(pretrained, **kwargs):
    model = CSPDarkNet()
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model
