import cv2
import numpy as np
import time

import copy

from skimage import color, io
from PIL import Image

# 到时候可以尝试numpy优化
def quick_loop(image, image_avg, image_std, temp_avg, temp_std, isHed=False):

    # for k in range(3):
    #     image_new[:,:,k] = (image[:,:,k] - image_avg[k]) * (temp_std[k] / (image_std[k]) )+ temp_avg[k]
    # print(type(image),image.shape)
    # print(image_avg)
    image = (image - np.array(image_avg))*(np.array(temp_std)/np.array(image_std))+np.array(temp_avg)
    if isHed : #1.10添加，针对hed进行特殊操作
        pass
    else:
        image = np.clip(image, 0, 255).astype(np.uint8) #这边的问题
    return image

# @numba.jit(nopython=True)
# 原始版本，3重for循环
def for_loop(image, height, width, channel, image_avg, image_std, temp_avg, temp_std):
    for i in range(0, height):
        for j in range(0, width):
            for k in range(0, channel):
                t = image[i, j, k]
                if abs(image_std[k]) < 0.0001:
                    image_std[k] = 0.0001  # 下面有255保护
                t = (t - image_avg[k]) * (temp_std[k] / image_std[k]) + temp_avg[k]
                t = 0 if t < 0 else t
                t = 255 if t > 255 else t
                image[i, j, k] = t
    # cv2.imwrite('test1.png', image)
    return image.astype(np.uint8)

# @torch.no_grad()
def getavgstd(image):
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

# 初次实验模板数值
# target_avg:  [171.03409996811226, 151.29910714285714, 109.92771444515306]
# target_std:  [37.22305651345217, 9.072264487990362, 8.478056840434128]
def reinhard_cn(image_path, temp_path, save_path, isDebug=False, color_space=None):
    isHed = False 
    image = cv2.imread(image_path)
    if isDebug:
        cv2.imwrite('source.png',image)
    
    template = cv2.imread(temp_path)  ### template images
    if isDebug:
        cv2.imwrite('template.png',template)
        
    if color_space == 'LAB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2LAB)
#         cv2.imwrite('lab.png',image)
    elif color_space == 'HED':
        isHed = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #color.rgb2hed需要rgb的ndarray作为输入
        template = cv2.cvtColor(template,cv2.COLOR_BGR2RGB)
#         image = np.array(Image.open(image_path))
#         template = np.array(Image.open(temp_path))
        
        image = color.rgb2hed(image) #归一化
        template = color.rgb2hed(template) #归一化，所以下边要注意
    elif color_space == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    elif color_space == 'GRAY': #如果是灰色，下面都不用处理了
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(save_path,image)
        return 
    
    image_avg, image_std = getavgstd(image)
    template_avg, template_std = getavgstd(template)
    # template_avg, template_std = [171.03409996811226, 151.29910714285714, 109.92771444515306], [37.22305651345217, 9.072264487990362, 8.478056840434128]
    if isDebug: #正常
        print("isDebug!!!")
        print('source_avg: ', image_avg)
        print('source_std: ', image_std)
        print('target_avg: ', template_avg)
        print('target_std: ', template_std)

    # 注意，python函数传矩阵和list一样也会是内存空间相同，所以后面可能需要注意一下
    # quick_loop快速颜色归一化操作
    image = quick_loop(image, image_avg, image_std, template_avg, template_std, isHed=isHed)
    # origin版颜色归一化操作
    # height, width, channel = image.shape
    # image_origin = for_loop(image, height, width, channel, image_avg, image_std, template_avg, template_std)

    if color_space == 'LAB':
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        cv2.imwrite(save_path,image)
    elif color_space == 'HED':
        image = color.hed2rgb(image) # 转成0-1了，所以需要恢复一下
        imin = image.min()
        imax = image.max()
        image = (255 * (image - imin) / (imax - imin)).astype('uint8')
        image = Image.fromarray(image)
        image.save(save_path)
        
    elif color_space == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        
        cv2.imwrite(save_path,image)
        
    if isDebug:
        cv2.imwrite('results.png', image)

# 人工指定模板的归一化
def reinhard_cn_temp(image_path, temp_path, save_path, isDebug=False):

    image = cv2.imread(image_path)
    if isDebug:
        cv2.imwrite('source.png',image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    cv2.imwrite('lab.png',image)

    # template = cv2.imread(temp_path)  ### template images
    # if isDebug:
    #     cv2.imwrite('template.png',template)
    # template = cv2.cvtColor(template, cv2.COLOR_BGR2LAB)

    image_avg, image_std = getavgstd(image)
    # template_avg, template_std = getavgstd(template)
    template_avg, template_std = [159.685, 150.534, 116.994], [36.815, 8.078, 6.072] #random 3000的结果
    if isDebug:
        print("isDebug!!!")
        print('source_avg: ', image_avg)
        print('source_std: ', image_std)
        print('target_avg: ', template_avg)
        print('target_std: ', template_std)

    # 注意，python函数传矩阵和list一样也会是内存空间相同，所以后面可能需要注意一下
    # quick_loop快速颜色归一化操作
    image = quick_loop(image, image_avg, image_std, template_avg, template_std)
    # origin版颜色归一化操作
    # height, width, channel = image.shape
    # image_origin = for_loop(image, height, width, channel, image_avg, image_std, template_avg, template_std)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    cv2.imwrite(save_path,image)

    if isDebug:
        cv2.imwrite('results.png', image)

if __name__ == '__main__':
    # demo
    image_path = r'/mnt/pycharm_project_colorNorm/output/colorNorm_effect/TUM-AIQIMVKD_source.png'
    # image_path = r'/mnt/nine_class/train_use_2w/TUM/TUM-ADEMNHMK.png'
    # image_path = './demo/other/TUM-TCGA-CVATFAAT.png'
    # temp_path = r'/mnt/pycharm_project_colorNorm/output/colorNorm_effect/TUM-CEQTLTKV_target_1.png'
    # temp_path = r'/mnt/pycharm_project_colorNorm/output/colorNorm_effect/TUM-CNPQPHGS_target2.png'
    temp_path = './demo/other/TUM-AIQIMVKD_template.png'
    # save_path = r'/mnt/pycharm_project_colorNorm/output/colorNorm_effect/source_target2_effect.png'
    save_path = './save/other/norm_TUM-TCGA-CVATFAAT.png'
    t1 = time.time()
    reinhard_cn(image_path, temp_path, save_path, isDebug=True)
    t2 = time.time()
    print(t2-t1)
    print('Color Norm finished!!!!')