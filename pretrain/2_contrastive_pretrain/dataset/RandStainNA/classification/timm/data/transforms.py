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

# 2.27引入，t分布
def single_t_rvs(loc, scale):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom #自由度
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    loc = np.array(loc)
    df = 2000 #给一个差不多的值即可
    x = (np.random.chisquare(df, 1)/df)[0] # 卡方分布，所以t分布是自己实现的
                                               # https://baike.baidu.com/item/t%E5%88%86%E5%B8%83/299142
    # z = np.random.multivariate_normal(np.zeros(d),S,(n,)) #目前是多元正态，搞一元即可
    z = np.random.normal(loc=0, scale=scale)
    return loc + z/np.sqrt(x)   # same output format as random.multivariate_normal

# m = 140
# S = 40
# x = single_t_rvs(loc, scale) ，和np.randon.distribution统一

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
                    elif self.distribution == 't':
                        np_distribution = single_t_rvs #2.27添加，t分布

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
        
# 12.26 引入hed_nrom_jitter
class hed_norm_jitter(object):
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

    def __init__(self, mean, std, std_hyper=0, probability=0):  # mean是正态分布的均值，std是正态分布的方差？标准差？
        self.mean = mean  # [l,a,b] 是l_mean的正态分布的均值和方差，是一个字典
        self.std = std  # [l,a,b]
        self.std_adjust = std_hyper #=0时按照统计规则
        self.p = probability  # 一半概率选一个

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

        image1 = (image1 - np.array(image_avg)) * (np.array(temp_std) / np.array(image_std)) + np.array(temp_avg)
#         image1 = np.clip(image1, 0, 255).astype(np.uint8) #hed的范围很小

        return image1

    def __call__(self, img):

        img = np.array(img) #rgb，1.10发现bug，这边是直接rgb转的hed，但是统计的是bgr2hed，所以对应不上，结果不够理想

        img_hed = color.rgb2hed(img) #rgb, [0,1]
        
        image_avg, image_std = self.getavgstd(img_hed)

        h_mean, e_mean, d_mean = self.mean[0], self.mean[1], self.mean[2]
        h_std, e_std, d_std = self.std[0], self.std[1], self.std[2]
        std_adjust = self.std_adjust
        template_avg_h, template_std_h = np.random.normal(loc=h_mean['mean'], scale=h_mean['std']*(1+std_adjust),
                                                          size=1), np.random.laplace(loc=h_std['mean'],
                                                                                    scale=h_std['std'], size=1)
        template_avg_e, template_std_e = np.random.laplace(loc=e_mean['mean'], scale=e_mean['std']*(1+std_adjust),
                                                          size=1), np.random.laplace(loc=e_std['mean'],
                                                                                    scale=e_std['std'], size=1)
        template_avg_d, template_std_d = np.random.laplace(loc=d_mean['mean'], scale=d_mean['std']*(1+std_adjust),
                                                           size=1), np.random.laplace(loc=d_std['mean'],
                                                                                      scale=d_std['std'], size=1)
        template_avg = [float(template_avg_h), float(template_avg_e), float(template_avg_d)]
        template_std = [float(template_std_h), float(template_std_e), float(template_std_d)]
        image = self.quick_loop(img_hed, image_avg, image_std, template_avg, template_std) #返回也是[0,1]

        nimg = color.hed2rgb(image)

        imin = nimg.min()
        imax = nimg.max()

        rsimg = (255 * (nimg - imin) / (imax - imin)).astype('uint8')  # rescale to [0,255]

        return Image.fromarray(rsimg)
    
    # 1.21引入，print内容添加
    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += f"methods=Reinhard"
        format_string += f", colorspace=HED"
        format_string += f", mean={self.mean}"
        format_string += f", std={self.std}"
        format_string += f", std_adjust={self.std_adjust}"
        format_string += f", p={self.p})"
        return format_string

# 2.13加入hsvjitter
# 对transform.colorJitter封装
# 其实可以random.apply
class HSVJitter(object):
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, p=1.0):
        self.brightness=brightness
        self.contrast=contrast
        self.saturation=saturation
        self.hue=hue
        self.p=p
        self.colorJitter=transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def __call__(self, img):
        if np.random.rand(1) < self.p: #2.13加入概率
            img_process = self.colorJitter(img)
            return img_process
        else:
            return img
    
    def __repr__(self):
        format_string = "("
        format_string += self.colorJitter.__repr__()
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
    

#12.25引入GaussBlur
class RandomGaussBlur(object):
    """Random GaussBlurring on image by radius parameter.
    Args:
        radius (list, tuple): radius range for selecting from; you'd better set it < 2
    """
    def __init__(self, radius=None):
        self.radius = radius

    def __call__(self, img):

        radius = random.uniform(self.radius[0], self.radius[1]) #随机模糊

        return img.filter(ImageFilter.GaussianBlur(radius=radius)) #只需要高斯核的标准差

    def __repr__(self):
        return self.__class__.__name__ + '(Gaussian Blur radius=[{0},{1}])'.format(self.radius[0],self.radius[1])

# 12.25 引入高斯噪声
class RandomGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=np.random.uniform(0, self.variance), size=(h, w, 1)) #弄成随机方差
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255                       # 避免有值超过255而反转
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img
    
    # 1.21引入，print内容添加
    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += f"mean={self.mean}"
        format_string += f", variance=uniform[0,{self.variance}]"
        format_string += f", amplitude={self.amplitude})"
 
        return format_string
    
class ToNumpy:

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img


class ToTensor:

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return torch.from_numpy(np_img).to(dtype=self.dtype)


_pil_interpolation_to_str = {
    Image.NEAREST: 'nearest',
    Image.BILINEAR: 'bilinear',
    Image.BICUBIC: 'bicubic',
    Image.BOX: 'box',
    Image.HAMMING: 'hamming',
    Image.LANCZOS: 'lanczos',
}
_str_to_pil_interpolation = {b: a for a, b in _pil_interpolation_to_str.items()}


if has_interpolation_mode:
    _torch_interpolation_to_str = {
        InterpolationMode.NEAREST: 'nearest',
        InterpolationMode.BILINEAR: 'bilinear',
        InterpolationMode.BICUBIC: 'bicubic',
        InterpolationMode.BOX: 'box',
        InterpolationMode.HAMMING: 'hamming',
        InterpolationMode.LANCZOS: 'lanczos',
    }
    _str_to_torch_interpolation = {b: a for a, b in _torch_interpolation_to_str.items()}
else:
    _pil_interpolation_to_torch = {}
    _torch_interpolation_to_str = {}


def str_to_pil_interp(mode_str):
    return _str_to_pil_interpolation[mode_str]


def str_to_interp_mode(mode_str):
    if has_interpolation_mode:
        return _str_to_torch_interpolation[mode_str]
    else:
        return _str_to_pil_interpolation[mode_str]


def interp_mode_to_str(mode):
    if has_interpolation_mode:
        return _torch_interpolation_to_str[mode]
    else:
        return _pil_interpolation_to_str[mode]


_RANDOM_INTERPOLATION = (str_to_interp_mode('bilinear'), str_to_interp_mode('bicubic'))


class RandomResizedCropAndInterpolation:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear'):
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = str_to_interp_mode(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return F.resized_crop(img, i, j, h, w, self.size, interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = ' '.join([interp_mode_to_str(x) for x in self.interpolation])
        else:
            interpolate_str = interp_mode_to_str(self.interpolation)
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


