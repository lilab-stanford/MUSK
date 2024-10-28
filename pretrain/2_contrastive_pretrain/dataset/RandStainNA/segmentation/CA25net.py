import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torchvision import models
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
from os import listdir
from torchvision.io import read_image


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x



class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=False, n_concat=2):
        super(unetUp_origin, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)


class info_agg(nn.Module):
    def __init__(self, in_size, out_size=256):
        # information aggregation module
        # 1 is mask, 2 is contour
        super(info_agg,self).__init__()
        self.convshare = unetConv2(in_size*3, out_size, True)
        self.conv11 = unetConv2(out_size, out_size, False)
        self.conv12 = unetConv2(out_size, out_size, False)
        self.conv13 = unetConv2(out_size, out_size, False)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, input1, input2, input3):
        fshare = self.convshare(torch.cat([input1, input2, input3], 1))
        return self.conv11(fshare), self.conv12(fshare), self.conv13(fshare)


from torch.nn import init
def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class Cia(nn.Module):

    def __init__(self, in_channels=1, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True,is_ds=False):
        super(Cia, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        self.feature_scale = feature_scale

        filters = [32, 64, 128, 256, 512]
        info_channel = 256
        #filters = [64, 128, 256, 512, 1024]
        # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)


        # upsampling
        # mask path
        self.up_concat01 = unetUp_origin(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp_origin(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp_origin(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp_origin(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp_origin(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp_origin(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp_origin(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp_origin(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp_origin(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp_origin(filters[1], filters[0], self.is_deconv, 5)

        # boundary path
        self.upb_concat01 = unetUp_origin(filters[1], filters[0], self.is_deconv)
        self.upb_concat11 = unetUp_origin(filters[2], filters[1], self.is_deconv)
        self.upb_concat21 = unetUp_origin(filters[3], filters[2], self.is_deconv)
        self.upb_concat31 = unetUp_origin(filters[4], filters[3], self.is_deconv)

        self.upb_concat02 = unetUp_origin(filters[1], filters[0], self.is_deconv, 3)
        self.upb_concat12 = unetUp_origin(filters[2], filters[1], self.is_deconv, 3)
        self.upb_concat22 = unetUp_origin(filters[3], filters[2], self.is_deconv, 3)

        self.upb_concat03 = unetUp_origin(filters[1], filters[0], self.is_deconv, 4)
        self.upb_concat13 = unetUp_origin(filters[2], filters[1], self.is_deconv, 4)

        self.upb_concat04 = unetUp_origin(filters[1], filters[0], self.is_deconv, 5)

        # three class path
        self.upt_concat31 = unetUp_origin(filters[4], filters[3], self.is_deconv)
        self.upt_concat22 = unetUp_origin(filters[3], filters[2], self.is_deconv, 3)
        self.upt_concat13 = unetUp_origin(filters[2], filters[1], self.is_deconv, 4)
        self.upt_concat04 = unetUp_origin(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)
        self.finalb_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.finalb_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.finalb_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.finalb_4 = nn.Conv2d(filters[0], n_classes, 1)
        self.finalt_4 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
          if isinstance(m, nn.Conv2d):
              init_weights(m)
          elif isinstance(m, nn.BatchNorm2d):
              init_weights(m)

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)
        maxpool0 = self.maxpool0(X_00)
        X_10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool1(X_10)
        X_20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool2(X_20)
        X_30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool3(X_30)
        X_40 = self.conv40(maxpool3)

        # Semantic segmentation path
        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)

        Y_01 = self.upb_concat01(X_10, X_00)
        Y_11 = self.upb_concat11(X_20, X_10)
        Y_21 = self.upb_concat21(X_30, X_20)
        Y_31 = self.upb_concat31(X_40, X_30)
        
        Z_31 = self.upt_concat31(X_40, X_30)

        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)

        Y_02 = self.upb_concat02(Y_11, X_00, Y_01)
        Y_12 = self.upb_concat12(Y_21, X_10, Y_11)
        Y_22 = self.upb_concat22(Y_31, X_20, Y_21)
        
        Z_22 = self.upt_concat22(Z_31, X_20, Y_21)

        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)

        Y_03 = self.upb_concat03(Y_12, X_00, Y_01, Y_02)
        Y_13 = self.upb_concat13(Y_22, X_10, Y_11, Y_12)
        
        Z_13 = self.upt_concat13(Z_22, X_10, Y_11, Y_12)

        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)
        Y_04 = self.upb_concat04(Y_13, X_00, Y_01, Y_02, Y_03)
        Z_04 = self.upt_concat04(Z_13, X_00, Y_01, Y_02, Y_03)

        # final layer
        final_m = self.final_4(X_04)
        final_b1 = self.finalb_1(Y_01)
        final_b2 = self.finalb_2(Y_02)
        final_b3 = self.finalb_3(Y_03)
        final_b4 = self.finalb_4(Y_04)
        final_t = self.finalt_4(Z_04)

        if self.is_ds:
            return torch.sigmoid(final_m), torch.sigmoid(torch.cat([final_b4,final_t],1))
        else:
            return torch.sigmoid(final_m), torch.sigmoid(torch.cat([(0.5*final_b1+0.75*final_b2+1.25*final_b3+1.5*final_b4)/4,final_t],1))


# loss function
# boundary loss
def _bd_loss(pred, target):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = -(torch.sum(target[i]*torch.log(pred[i]+1e-6) + (1-target[i])*torch.log(1-pred[i]+1e-6)))
        IoU = IoU + Iand1/512/512

    return IoU/b

class BD(torch.nn.Module):
    def __init__(self, size_average = True):
        super(BD, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return _bd_loss(pred, target)

def bd_loss(pred,label):
    loss = BD(size_average=True)
    bd_out = loss(pred, label)
    return bd_out

# CIA loss
def _cia_loss(pred, target, w):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        classes = target[i] > 0
        Iand1 = -torch.sum(classes*torch.log(pred[i][0]+1e-6)/(torch.sum(classes)+1) + ~classes*torch.log(1-pred[i][0]+1e-6)/(torch.sum(~classes)+1))
        # print('class{}: {}'.format(j,Iand1))
        IoU = IoU + (1-w)*Iand1
        
        classes = target[i] > 1
        Iand1 = -torch.sum(classes*torch.log(pred[i][1]+1e-6)/(torch.sum(classes)+1) + ~classes*torch.log(1-pred[i][1]+1e-6)/(torch.sum(~classes)+1))
        # print('class2: {}'.format(Iand1))
        IoU = IoU + w*Iand1            

    return IoU/b

def _st_loss(pred, target, thresh):
    # Smooth Truncated Loss
    b = pred.shape[0]
    ST = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        w = target[i] > 1
        pt = w * pred[i][1]
        w = target[i] > 0
        pt = pt + w*pred[i][0]
        certain = pt > thresh
        Iand1 = -(torch.sum( certain*torch.log(pt+1e-6) + ~certain*(np.log(thresh) - (1-(pt/thresh)**2)/2) ))
        ST = ST + Iand1/512/512

    return ST/b

class CIA(torch.nn.Module):
    def __init__(self, size_average = True):
        super(CIA, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target, w, thresh, lw):
        # print(_cia_loss(pred, target), _st_loss(pred, target, thresh))
        return _cia_loss(pred, target, w) + lw * _st_loss(pred, target, thresh)

def cia_loss(pred, label, w, thr=0.2, lamb=0.5):
    Cia_loss = CIA(size_average=True)
    cia_out = Cia_loss(pred, label, w, thr, lamb)
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


# Dataset
# class XuDataset(Dataset):
#     def __init__(self, img_dir):
#         self.img_dir = img_dir
#
#     def __len__(self):
#         return len(listdir(os.path.join(self.img_dir, 'data')))
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, 'data', 'i'+str(idx+1)+'.png')
#         image = read_image(img_path)
#         image = image.float()
#         image = image / image.max()
#
#         mask_path = os.path.join(self.img_dir, 'label', 'm'+str(idx+1)+'.png')
#         label = read_image(mask_path)
#
#         bound_path = os.path.join(self.img_dir, 'bound', 'b'+str(idx+1)+'.png')
#         bound = read_image(bound_path)
#         return image[0].unsqueeze(0), label[0].unsqueeze(0), bound[0].unsqueeze(0)

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
        image = read_image(img_path)
        image = image.float()
        image = image / image.max()

        mask_name = self.label_names[idx]
        mask_path = os.path.join(self.img_dir, 'label', mask_name)
        label = read_image(mask_path)
        label = label.float()
        label = label / 255.0

        bound_name = self.bound_names[idx]
        bound_path = os.path.join(self.img_dir, 'bound', bound_name)
        bound = read_image(bound_path)
        return image[0].unsqueeze(0), label[0].unsqueeze(0), bound[0].unsqueeze(0)



class TestDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.data_names = listdir(os.path.join(self.img_dir))
        

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        img_name = self.data_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = read_image(img_path)
        image = image.float()
        image = image / image.max()
    
        return image[0].unsqueeze(0),img_name
