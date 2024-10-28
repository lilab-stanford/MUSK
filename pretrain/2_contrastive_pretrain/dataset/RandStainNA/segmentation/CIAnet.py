import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from torchvision.io import read_image
from torch.utils.data import Dataset,DataLoader,TensorDataset
from os import listdir
import os
from PIL import Image
from torchvision import transforms

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class info_agg(nn.Module):
    def __init__(self, in_size, out_size=256):
        # information aggregation module
        # 1 is mask, 2 is contour
        super(info_agg,self).__init__()
        self.conv01 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=False)
        self.conv02 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=False)
        self.convshare = nn.Conv2d(2*out_size, out_size, kernel_size=3, padding=1, bias=False)
        self.conv11 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=False)
        self.conv12 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=False)

        # initialise the blocks
        # for m in self.children():
        #     if m.__class__.__name__.find('unetConv2') != -1: continue
        #     init_weights(m, init_type='kaiming')

    def forward(self, input1, input2):
        input1 = self.conv01(input1)
        input2 = self.conv02(input2)
        fshare = self.convshare(torch.cat([input1, input2], 1))
        return self.conv11(fshare), self.conv12(fshare)


class Lateral_Connection(nn.Module):
    def __init__(self, left_size, down_size, is_deconv=False, n_concat=2):
        super(Lateral_Connection, self).__init__()
        self.conv = nn.Conv2d(left_size, 256, kernel_size=3, padding=1, bias=False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(down_size, down_size//2, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        # for m in self.children():
        #     if m.__class__.__name__.find('unetConv2') != -1: continue
        #     init_weights(m, init_type='kaiming')

    def forward(self, down_input, left_input):
        outputs0 = self.up(down_input)
        outputs0 = torch.cat([outputs0, self.conv(left_input)], 1)
        return outputs0

class CIAnet(nn.Module):
    def __init__(self, growthRate, nDenseBlocks, reduction, bottleneck):
        super(CIAnet, self).__init__()

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks[0], bottleneck)
        nChannels1 = nChannels + nDenseBlocks[0]*growthRate

        nOutChannels = int(math.floor(nChannels1*reduction))
        self.trans1 = Transition(nChannels1, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks[1], bottleneck)
        nChannels2 = nChannels + nDenseBlocks[1]*growthRate
        nOutChannels = int(math.floor(nChannels2*reduction))
        self.trans2 = Transition(nChannels2, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks[2], bottleneck)
        nChannels3 = nChannels + nDenseBlocks[2]*growthRate
        nOutChannels = int(math.floor(nChannels3*reduction))
        self.trans3 = Transition(nChannels3, nOutChannels)

        nChannels = nOutChannels
        self.dense4 = self._make_dense(nChannels, growthRate, nDenseBlocks[3], bottleneck)
        nChannels4 = nChannels + nDenseBlocks[3]*growthRate

        # decoder
        # 0 denotes semantic mask, 1 denotes boundary mask
        self.latconnect04 = Lateral_Connection(nChannels3,nChannels4)
        self.latconnect14 = Lateral_Connection(nChannels3,nChannels4)
        self.iam4 = info_agg(nChannels4+256)

        self.latconnect03 = Lateral_Connection(nChannels2,256)
        self.latconnect13 = Lateral_Connection(nChannels2,256)
        self.iam3 = info_agg(512)

        self.latconnect02 = Lateral_Connection(nChannels1,256)
        self.latconnect12 = Lateral_Connection(nChannels1,256)
        self.iam2 = info_agg(512)

        self.final_0 = nn.Conv2d(256, 1, 1)
        self.final_1 = nn.Conv2d(256, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        # encoder
        out = self.conv1(x)
        x0 = self.dense1(out)
        out = self.trans1(x0)
        x1 = self.dense2(out)
        out = self.trans2(x1)
        x2 = self.dense3(out)
        out = self.trans3(x2)
        x3 = self.dense4(out)

        # decoder
        # 0 denotes mask, 1 denotes boundary
        y02 = self.latconnect04(x3,x2)
        y12 = self.latconnect14(x3,x2)
        y02,y12 = self.iam4(y02,y12)

        y01 = self.latconnect03(y02,x1)
        y11 = self.latconnect13(y12,x1)
        y01,y11 = self.iam3(y01,y11)

        y00 = self.latconnect02(y01,x0)
        y10 = self.latconnect12(y11,x0)
        y00,y10 = self.iam2(y00,y10)

        # final layer
        y00 = self.final_0(y00)
        y10 = self.final_1(y10)
        return (torch.sigmoid(y00), torch.sigmoid(y10))


# loss function

# CIA loss
def _cia_loss(pred, target):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        classes = target[i] > 0
        Iand1 = -torch.sum(classes*torch.log(pred[i][0]+1e-6)/(torch.sum(classes)+1) + ~classes*torch.log(1-pred[i][0]+1e-6)/(torch.sum(~classes)+1))
        # print('class{}: {}'.format(j,Iand1))
        IoU = IoU + Iand1

    return IoU/b

def _st_loss(pred, target, thresh):
    # Smooth Truncated Loss
    b = pred.shape[0]
    ST = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        w = target[i] > 0
        pt = w*pred[i][0]
        certain = pt > thresh
        Iand1 = -(torch.sum( certain*torch.log(pt+1e-6) + ~certain*(np.log(thresh) - (1-(pt/thresh)**2)/2) ))
        ST = ST + Iand1/512/512

    return ST/b

class CIA(torch.nn.Module):
    def __init__(self, size_average = True):
        super(CIA, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target, thresh, lw):
        # print(_cia_loss(pred, target), _st_loss(pred, target, thresh))
        return _cia_loss(pred, target) + lw * _st_loss(pred, target, thresh)

def cia_loss(pred, label, thr=0.2, lamb=0.5):
    Cia_loss = CIA(size_average=True)
    cia_out = Cia_loss(pred, label, thr, lamb)
    return cia_out


# IOU loss
def _iou(pred, target, size_average = True):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        w = target[i] == 0
        Iand1 = torch.sum(target[i]*pred[i])
        Ior1 = torch.sum(target[i]) + torch.sum(pred[i])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)

def my_loss(pred,label):
    iou_loss = IOU(size_average=True)
    iou_out = iou_loss(pred, label)
    # print("iou_loss:", iou_out.data.cpu().numpy())
    return iou_out


# accuracy
def my_acc(pred,target):
    # IOU
    temp = pred > 0.5
    temp = temp.long()
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i]*temp[i])
        Ior1 = torch.sum(target[i]) + torch.sum(temp[i])-Iand1
        IoU1 = Iand1.float()/Ior1.float()
        IoU = IoU + IoU1
    
    IoU = IoU/b
    return IoU.detach().cpu().numpy()

def bd_acc(pred,target):
    # boundary accuracy
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        temp = pred[i] > 0.5
        label = torch.zeros(2,512,512)
        label[0] = target[i] > 0
        label[1] = target[i] > 1
        # temp = temp.long()
        # compute the IoU of the foreground\
        label = label.cuda()
        Iand1 = torch.sum(label*temp)
        Ior1 = torch.sum(label) + torch.sum(temp)-Iand1
        IoU1 = Iand1.float()/Ior1.float()
        IoU = IoU + IoU1
    
    IoU = IoU/b
    return IoU.detach().cpu().numpy()

def dice_acc(pred,target):
    # dice coefficient
    temp = pred > 0.5
    temp = temp.long()
    IoU = 0.0
    Iand1 = 2*torch.sum(target[0]*temp)
    Ior1 = torch.sum(target[0]) + torch.sum(temp)
    if Ior1 == 0:
        IoU = 0
        return IoU
    else:
        IoU1 = Iand1.float()/Ior1.float()
        IoU = IoU + IoU1
        return IoU.detach().cpu().numpy()  

def pixel_acc(pred,target):
    # pixel accuracy
    temp = pred > 0.5
    temp = temp.long()
    b = pred.shape[0]
    precision,recall,F1 = 0.0,0.0,0.0
    for i in range(0,b):
        TP = torch.sum(target[i]*temp[i])
        FP = torch.sum((1-target[i])*temp[i])
        FN = torch.sum(target[i]*(1-temp[i]))
        precision = precision + TP.float() / (TP.float()+FP.float())
        recall = recall + TP.float() / (FN.float()+TP.float())
        F1 = F1 + 2*(precision*recall)/(recall+precision)
    
    return precision.detach().cpu().numpy(), recall.detach().cpu().numpy(), F1.detach().cpu().numpy()

#2.22添加数据集
class XuDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.data_names = listdir(os.path.join(self.img_dir, 'data'))
        self.label_names = listdir(os.path.join(self.img_dir, 'label'))
        self.bound_names = listdir(os.path.join(self.img_dir, 'bound'))

    def __len__(self):
        return len(listdir(os.path.join(self.img_dir, 'data')))

    def __getitem__(self, idx):
        img_name = self.data_names[idx]
        img_path = os.path.join(self.img_dir, 'data', img_name)
        # print(img_name)
        # image = read_image(img_path)
        # image = image.float()
        # image = image / image.max()
        trans = transforms.Compose([transforms.ToTensor()])
        image = Image.open(img_path).convert('RGB')
        image = trans(image)
        # print(image.shape)
        # print(torch.max(image))

        mask_name = self.label_names[idx]
        mask_path = os.path.join(self.img_dir, 'label', mask_name)
        # print(mask_name)
        # label = read_image(mask_path)
        # label = label.float()
        # label = label / 255.0
        label = Image.open(mask_path).convert('1')
        label = trans(label)
        # print(label.shape)
        # print(torch.max(label))

        bound_name = self.bound_names[idx]
        bound_path = os.path.join(self.img_dir, 'bound', bound_name)
        # print(bound_name)
        # bound = read_image(bound_path)
        # bound = bound.float()
        # bound = bound / 255.0 # 学习label的处理方式先试一下
        bound = Image.open(bound_path).convert('1')
        bound = trans(bound)
        # print(bound.shape)
        # print(torch.max(bound))
        
        # return image[0].unsqueeze(0), label[0].unsqueeze(0), bound[0].unsqueeze(0)
        # return image.unsqueeze(0), label.unsqueeze(0), bound.unsqueeze(0)
        return image, label, bound
    
class ciaData(Dataset): #继承Dataset
    def __init__(self, root_dir, transform=None): #__init__是初始化该类的一些基础参数
        self.root_dir = root_dir   #文件目录
        self.transform = transform #变换
        self.images = os.listdir(os.path.join(self.root_dir, 'data'))#目录里的所有文件
    
    def __len__(self):#返回整个数据集的大小
        return len(self.images)
    
    def __getitem__(self,index):#根据索引index返回dataset[index]
        image_index = self.images[index]#根据索引index获取该图片
        img_path = os.path.join(self.root_dir, 'data', image_index)#获取索引为index的图片的路径名
        lab_path = os.path.join(self.root_dir, 'label', image_index)
        bou_path = os.path.join(self.root_dir, 'bound', image_index)
        img = Image.open(img_path).convert('RGB')# 读取该图片
        label = Image.open(lab_path).convert('1')
        bound = Image.open(bou_path).convert('1')

        
        # if (self.transform == None):
        #     T=transforms.Compose([
        #         transforms.Resize((512, 512)),
        #         transforms.ToTensor()
        #     ])
        # else:
        T1 = transforms.Compose([
                # transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])
        T = self.transform
            
        # 如果没有设置，就有个T的默认设置，否则就用给定的
        img = T(img)#对样本进行变换
        label = T1(label)
        bound = T1(bound)   

        return img,label,bound #返回该样本