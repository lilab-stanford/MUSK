import torch
import torchvision.transforms.functional as F
from torchvision import transforms #2.13加入
try:
    from torchvision.transforms.functional import InterpolationMode
    has_interpolation_mode = True
except ImportError:
    has_interpolation_mode = False
from PIL import Image, ImageFilter #1.20加入
import warnings
import math
import random
import numpy as np
import cv2 #12.20加入
from skimage import color #12.26加入
import os #12.20加入

#12.17 norm&jitter引入方法
class color_norm_jitter(object):
    '''
    参数：
    1.lab的三个channel的mean和std（这个一般是在外面算完传入进来的，在里面算分布）
    2.Reinhard_cn方法
    3.概率p
    '''

    def __init__(self, mean, std, std_hyper=0, probability=0, color_space=None, distribution=None):
        self.mean = mean  # [l,a,b] 是l_mean的正态分布的均值和方差，是一个字典
        self.std = std  # [l,a,b]
        self.std_adjust = std_hyper #=0时按照统计规则
        self.p = probability  # 一半概率选一个
        self.color_space = color_space
        self.distribution = distribution #1.30添加，手工指定分布

    def getavgstd(self, image):
        avg = []
        std = []
        image_avg_l = np.mean(image[:, :, 0])
        image_std_l = np.std(image[:, :, 0])
        image_avg_a = np.mean(image[:, :, 1])
        image_std_a = np.std(image[:, :, 1])
        image_avg_b = np.mean(image[:, :, 2])
        image_std_b = np.std(image[:, :, 2])
        avg.append(image_avg_l)
        avg.append(image_avg_a)
        avg.append(image_avg_b)
        std.append(image_std_l)
        std.append(image_std_a)
        std.append(image_std_b)
        return (avg, std)

    def quick_loop(self, image1, image_avg, image_std, temp_avg, temp_std):
        if self.color_space != 'HED': #LAB和HSV
            image_std = np.clip(np.array(image_std), 0.001, 255)
            image1 = (image1 - np.array(image_avg)) * (np.array(temp_std) / np.array(image_std)) + np.array(temp_avg)
            image1 = np.clip(image1, 0, 255).astype(np.uint8)
        else: #HED
            image_std = np.clip(np.array(image_std), 0.0001, 255) #经常容易除0，保护一下
            image1 = (image1 - np.array(image_avg)) * (np.array(temp_std) / np.array(image_std)) + np.array(temp_avg)

        return image1

    def __call__(self, img):
        # 这边应该考虑单张图就好了
        if np.random.rand(1) < self.p:
            image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # 注意颜色空间转换
            
            if self.color_space == 'LAB':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # 注意颜色空间转换
            elif self.color_space == 'HSV':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 注意颜色空间转换
            elif self.color_space == 'HED': #1.30将HED空间扰动也加入
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #hed的变化是在rgb上变化
                image = color.rgb2hed(img) #rgb, [0,1]
                
            image_avg, image_std = self.getavgstd(image)

            l_mean, a_mean, b_mean = self.mean[0], self.mean[1], self.mean[2]
            l_std, a_std, b_std = self.std[0], self.std[1], self.std[2]
            std_adjust = self.std_adjust
            
            #1.30修改，l_mean已经有'mean','std','distribution'3个参数
            if self.distribution != None: #1.30添加，如果有手工指定分布，则全部按照分布的来，否则按照统计的来
                if self.distribution == 'uniform':
                    np_distribution = np.random.uniform #均匀分布时，按照3Σ原则来确定采样范围
                    template_avg_l = np_distribution(low=l_mean['mean']-3*l_mean['std'], high=l_mean['mean']+3*l_mean['std'])
                    template_std_l = np_distribution(low=l_std['mean']-3*l_std['std'], high=l_std['mean']+3*l_std['std'])
                    
                    template_avg_a = np_distribution(low=a_mean['mean']-3*a_mean['std'], high=a_mean['mean']+3*a_mean['std'])
                    template_std_a = np_distribution(low=a_std['mean']-3*a_std['std'], high=a_std['mean']+3*a_std['std'])
                    
                    template_avg_b = np_distribution(low=b_mean['mean']-3*b_mean['std'], high=b_mean['mean']+3*b_mean['std'])
                    template_std_b = np_distribution(low=b_std['mean']-3*b_std['std'], high=b_std['mean']+3*b_std['std'])
                    
                else: #不是均匀分布时，考虑的是均值和方差
                    if self.distribution == 'normal':
                        np_distribution = np.random.normal
                    elif self.distribution == 'laplace':
                        np_distribution = np.random.laplace
                    
                    # 2.05添加，1+std调整为全部的
                    template_avg_l = np_distribution(loc=l_mean['mean'], scale=l_mean['std']*(1+std_adjust))
                    template_std_l = np_distribution(loc=l_std['mean'], scale=l_std['std']*(1+std_adjust))
                    
                    template_avg_a = np_distribution(loc=a_mean['mean'], scale=a_mean['std']*(1+std_adjust))
                    template_std_a = np_distribution(loc=a_std['mean'], scale=a_std['std']*(1+std_adjust))
                    
                    template_avg_b = np_distribution(loc=b_mean['mean'], scale=b_mean['std']*(1+std_adjust))
                    template_std_b = np_distribution(loc=b_std['mean'], scale=b_std['std']*(1+std_adjust))
            
            else: #如果没有指定分布，则需要根据nj参数来确定各分布
                np_d_true_list = [l_mean['distribution'], l_std['distribution'], a_mean['distribution'], a_std['distribution'], b_mean['distribution'], b_std['distribution']]
                # print(np_d_true_list)
                np_d_sample_list = []
                for np_d_true in np_d_true_list:
                    if np_d_true == 'norm':
                        np_d_sample_list.append(np.random.normal)
                    elif np_d_true == 'laplace':
                        np_d_sample_list.append(np.random.laplace)
                # print(np_d_sample_list)
                # 2.5修改，1+std改为全部
                template_avg_l = np_d_sample_list[0](loc=l_mean['mean'], scale=l_mean['std']*(1+std_adjust))
                template_std_l = np_d_sample_list[1](loc=l_std['mean'], scale=l_std['std']*(1+std_adjust))

                template_avg_a = np_d_sample_list[2](loc=a_mean['mean'], scale=a_mean['std']*(1+std_adjust))
                template_std_a = np_d_sample_list[3](loc=a_std['mean'], scale=a_std['std']*(1+std_adjust))

                template_avg_b = np_d_sample_list[4](loc=b_mean['mean'], scale=b_mean['std']*(1+std_adjust))
                template_std_b = np_d_sample_list[5](loc=b_std['mean'], scale=b_std['std']*(1+std_adjust))

                
            template_avg = [float(template_avg_l), float(template_avg_a), float(template_avg_b)]
            template_std = [float(template_std_l), float(template_std_a), float(template_std_b)]

            image = self.quick_loop(image, image_avg, image_std, template_avg, template_std)
            
            if self.color_space == 'LAB':
                image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
                return Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            
            elif self.color_space == 'HSV':
                image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
                return Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)) #这个算是调整好了
            
            elif self.color_space == 'HED':
                nimg = color.hed2rgb(image)
                imin = nimg.min()
                imax = nimg.max()
                rsimg = (255 * (nimg - imin) / (imax - imin)).astype('uint8')  # rescale to [0,255]
                return Image.fromarray(rsimg)
        else:
            return img
    
    # 1.21引入，print内容添加
    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += f"methods=Reinhard"
        format_string += f", colorspace={self.color_space}"
        format_string += f", mean={self.mean}"
        format_string += f", std={self.std}"
        format_string += f", std_adjust={self.std_adjust}"
        format_string += f", distribution={self.distribution}" #1.30添加，print期望分布
        format_string += f", p={self.p})"
        return format_string
    
#12.25引入HEDJitter方法
class HEDJitter(object):
    """Randomly perturbe the HED color space value an RGB image.
    First, it disentangled the hematoxylin and eosin color channels by color deconvolution method using a fixed matrix.
    Second, it perturbed the hematoxylin, eosin and DAB stains independently.
    Third, it transformed the resulting stains into regular RGB color space.
    Args:
        theta (float): How much to jitter HED color space,
         alpha is chosen from a uniform distribution [1-theta, 1+theta]
         betti is chosen from a uniform distribution [-theta, theta]
         the jitter formula is **s' = \alpha * s + \betti**
    """
    def __init__(self, theta=0., p=1.0): # HED_light: theta=0.05; HED_strong: theta=0.2
#         assert isinstance(theta, numbers.Number), "theta should be a single number."
        self.theta = theta
        # 12.26这边的随机采样不应该是这样的，应该是每次都随机
        self.alpha = 0 # np.random.uniform(1-theta, 1+theta, (1, 3))
        self.betti = 0 # np.random.uniform(-theta, theta, (1, 3))
        self.p = p #2.13加入

    @staticmethod
    def adjust_HED(img, alpha, betti):
        img = np.array(img)

        s = np.reshape(color.rgb2hed(img), (-1, 3))
        ns = alpha * s + betti  # perturbations on HED color space
        nimg = color.hed2rgb(np.reshape(ns, img.shape))

        imin = nimg.min()
        imax = nimg.max()
        rsimg = (255 * (nimg - imin) / (imax - imin)).astype('uint8')  # rescale to [0,255]
        # transfer to PIL image
        return Image.fromarray(rsimg)

    def __call__(self, img):
        # 每张图片都重新来弄，既可以记录，又可以更新
        if np.random.rand(1) < self.p: #2.13加入概率
            self.alpha = np.random.uniform(1-self.theta, 1+self.theta, (1, 3))
            self.betti = np.random.uniform(-self.theta, self.theta, (1, 3))
            return self.adjust_HED(img, self.alpha, self.betti)
        else:
            return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'theta={0}'.format(self.theta)
        format_string += ', alpha={0}'.format(self.alpha)
        format_string += ', betti={0}'.format(self.betti)
        format_string += ', p={0})'.format(self.p)
        return format_string
    
#12.25引入LABJitter方法
class LABJitter(object):
    """Randomly perturbe the LAB color space value an RGB image.
    First, it disentangled the hematoxylin and eosin color channels by color deconvolution method using a fixed matrix.
    Second, it perturbed the hematoxylin, eosin and DAB stains independently.
    Third, it transformed the resulting stains into regular RGB color space.
    Args:
        theta (float): How much to jitter HED color space,
         alpha is chosen from a uniform distribution [1-theta, 1+theta]
         betti is chosen from a uniform distribution [-theta, theta]
         the jitter formula is **s' = \alpha * s + \betti**
    """
    def __init__(self, theta=0., p=1.0): # LAB_light: theta=0.05; LAB_strong: theta=0.2
#         assert isinstance(theta, numbers.Number), "theta should be a single number."
        self.theta = theta
        # 12.26这边的随机采样不应该是这样的，应该是每次都随机
        self.alpha = 0 # np.random.uniform(1-theta, 1+theta, (1, 3))
        self.betti = 0 # np.random.uniform(-theta, theta, (1, 3))
        self.p = p #2.13加入概率

    @staticmethod
    def adjust_LAB(img, alpha, betti):
        img = np.array(img)

        s = np.reshape(color.rgb2lab(img), (-1, 3)) #1.21修改，rgb2hed改为rgb2lab
        ns = alpha * s + betti  # perturbations on LAB color space
        nimg = color.lab2rgb(np.reshape(ns, img.shape)) #1.21修改，hed2rgb改为lab2rgb

        imin = nimg.min()
        imax = nimg.max()
        rsimg = (255 * (nimg - imin) / (imax - imin)).astype('uint8')  # rescale to [0,255]
        # transfer to PIL image
        return Image.fromarray(rsimg)

    def __call__(self, img):
        # 每张图片都重新来弄，既可以记录，又可以更新
        if np.random.rand(1) < self.p: #2.13加入概率
            self.alpha = np.random.uniform(1-self.theta, 1+self.theta, (1, 3))
            self.betti = np.random.uniform(-self.theta, self.theta, (1, 3))
            return self.adjust_LAB(img, self.alpha, self.betti)
        else:
            return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'theta={0}'.format(self.theta)
        format_string += ', alpha={0}'.format(self.alpha)
        format_string += ', betti={0}'.format(self.betti)
        format_string += ', p={0})'.format(self.p)
        return format_string

# 2.6 加入labjitter-hsv策略
# 借鉴pytorch调整hue来修改各通道参数
class LABJitter_hsv(object):
    def __init__(self, l_factor, a_factor, b_factor, p=1.0):
        self.l_factor = l_factor
        self.a_factor = a_factor
        self.b_factor = b_factor
        self.p = p #2.13加入概率
        
    def adjust_channel(self, channel, factor) -> Image.Image:
        if not (0.5 <= factor <= 1.5):
            raise ValueError(f"factor ({factor}) is not in [-0.5, 0.5].")
            
        # h, s, v = img.convert("HSV").split()
        channel = np.array(channel, dtype=np.uint8) #确保整型
        
        # uint8 addition take cares of rotation across boundaries
        with np.errstate(over="ignore"):
            channel += np.uint8(factor * 255)
        
        channel = np.array(channel, dtype=np.uint8) #将超出范围的限制在0-255
        
        return channel

    def __call__(self, img):
        # 每张图片都重新来弄，既可以记录，又可以更新
        if np.random.rand(1) < self.p: #2.13加入概率
            l_factor = np.random.uniform(1-self.l_factor, 1+self.l_factor)
            a_factor = np.random.uniform(1-self.a_factor, 1+self.a_factor)
            b_factor = np.random.uniform(1-self.b_factor, 1+self.b_factor)

            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            np_l, np_a, np_b = cv2.split(img_lab)

            np_l = self.adjust_channel(np_l, l_factor)
            np_a = self.adjust_channel(np_a, a_factor)
            np_b = self.adjust_channel(np_b, b_factor)

            LAB = cv2.merge([np_l, np_a, np_b])
            image = cv2.cvtColor(LAB, cv2.COLOR_LAB2BGR)
        
            return Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        else:
            return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'l_factor={0}'.format(self.l_factor)
        format_string += ', a_factor={0}'.format(self.a_factor)
        format_string += ', b_factor={0}'.format(self.b_factor)
        format_string += ', p={0})'.format(self.p)
        return format_string

# 2.9 动态调整p的类
class Dynamic_P_class(object):
    
    def __init__(self, epochs=0, batch_per_epoch=0, dynamic_factor=1.0, function='sin_pi'):
        self.batches = epochs * batch_per_epoch #总batches数
        # 通过正弦*某值来调控到第几个epoch时是全力，*5时，3个epoch训练完后达到1，最后3个epoch恢复
        if function=='sin_pi': #先增后减型
            self.p_list = [math.sin(math.pi*idx / self.batches) *dynamic_factor for idx in range(self.batches)] #构造时就得到所有p的取值，sin的0-π
        elif function=='sin_pi_2': #递增型
            self.p_list = [math.sin(math.pi*idx / (2*self.batches)) *dynamic_factor for idx in range(self.batches)]
        self.i = -1
        
    def step(self): #每个batch递进都会取得下一个
        self.i += 1 #初始是0
        return self.p_list[self.i]
    
# 2.17添加
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)