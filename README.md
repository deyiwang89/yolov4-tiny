## YOLOV4-Tiny：You Only Look Once-Tiny目标检测模型在Pytorch当中的实现
---



## 目录
[TOC]

## 性能情况

参见results文件夹中的结果，均可复现

## 所需环境
torch>=1.2.0

## 注意事项
代码中的yolov4_tiny_weights_coco.pth和yolov4_tiny_weights_voc.pth是基于416x416的图片训练的。

## 小技巧的设置
在train.py文件下：   
1、mosaic参数可用于控制是否实现Mosaic数据增强。--实际不好用， 高精度结果是不使用moasic生成的   
2、Cosine_scheduler可用于控制是否使用学习率余弦退火衰减。--可以自己实验看看   
3、label_smoothing可用于控制是否Label Smoothing平滑。--一共两种类别， 不需要检测

## 文件下载
训练所需的yolov4_tiny_weights_voc.pth在model_data中

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载yolov4_tiny_voc.pth，放入model_data，运行predict.py，输入  
```python
img/street.jpg
```
2. 利用video.py可进行摄像头检测。  
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    "model_path": '已经调整好',
    "anchors_path": 'model_data/yolo_anchors.txt',#--需要使用kmeans生成
    "classes_path": 'model_data/voc_classes.txt,
    "score" : 0.5,
    "iou" : 0.3,
    # 显存比较小可以使用416x416
    # 显存比较大可以使用608x608
    "model_image_size" : (416, 416)
}

```
3. 运行predict.py，输入  
```python
img/street.jpg
```
4. 利用video.py可进行摄像头检测。  

## 训练步骤
1. 本文使用VOC格式进行训练。  
2. 训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3. 训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4. 在训练前利用voc2yolo4.py文件生成对应的txt。  
5. 再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。**注意不要使用中文标签，文件夹中不要有空格！**   
```python
#需要根据数据集进行修改
classes = ["unmasked", "masked"]
```
6. 此时会生成对应的2007_train.txt，每一行对应其**图片位置**及其**真实框的位置**。  
7. **在训练前需要务必在model_data下新建一个txt文档，文档中输入需要分的类，在train.py中将classes_path指向该文件**，示例如下：   
```python
classes_path = 'model_data/new_classes.txt'    
```
model_data/new_classes.txt文件内容为：   
```python
masked 

unmasked
...
```
8. 运行train.py即可开始训练。

## 创新点介绍：
基于论文《Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks》改进，在external attention基础上，增加了扩增与压制机制，对external attention的权重进行二次增强，实现对注意力机制的优化。具体代码体现在CSPdarknet53_tiny.py的代码中：
```python
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
```

## mAP目标检测精度计算更新
更新了get_gt_txt.py、get_dr_txt.py和get_map.py文件。  
get_map文件克隆自https://github.com/Cartucho/mAP  
具体mAP计算过程可参考：https://www.bilibili.com/video/BV1zE411u7Vw

## Reference
https://github.com/qqwweee/keras-yolo3/  
https://github.com/Cartucho/mAP  
https://github.com/Ma-Dan/keras-yolo4  
