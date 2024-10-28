""" Transforms Factory
Factory methods for building image transforms for use with TIMM (PyTorch Image Models)

Hacked together by / Copyright 2019, Ross Wightman
"""
import math

import torch
from torchvision import transforms
from torchvision.transforms import RandomErasing as RandomErasing_torch #1.23添加
from torchvision.transforms import RandomAffine
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from timm.data.transforms import str_to_interp_mode, str_to_pil_interp, RandomResizedCropAndInterpolation, ToNumpy
from timm.data.random_erasing import RandomErasing
from timm.data.transforms import color_norm_jitter, hed_norm_jitter, HEDJitter, LABJitter, LABJitter_hsv, HSVJitter, RandomGaussBlur, RandomGaussianNoise#, Normalizer_transform #12.30修改 #1.21添加LABJitter, #2.6添加LABJitter_hsv #2.13加入HSVJitter，对cj封装

def transforms_noaug_train(
        img_size=224,
        interpolation='bilinear',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
):
    if interpolation == 'random':
        # random interpolation not supported with no-aug
        interpolation = 'bilinear'
    tfl = [
        transforms.Resize(img_size, interpolation=str_to_interp_mode(interpolation)),
        transforms.CenterCrop(img_size)
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
    return transforms.Compose(tfl)


def transforms_imagenet_train(
        img_size=224, #注意，这边其实没有用到，因为并没有resize，输入图片是多大这边就是多大，torchvision的model可以兼容不同分辨率 1.30
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        morphology=None, #12.26添加，是否增加形态学处理
        color_jitter=0.4,
        norm_jitter=None, #12.20加入，norm&jitter数据增强方法
        hed_jitter=None, #12.26添加，一个theta参数
        lab_jitter=None, #1.21添加，一个theta参数
        random_jitter=None, #1.30添加，jitter上是否需要randomaug
        cj_p=1.0, #2.13添加，jitter的概率
        auto_augment=None,
        interpolation='random',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        separate=False,
        logger=None
):
    """
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3./4., 4./3.))  # default imagenet ratio range
    
    # 12.20,修改为pytorch自带的transforms
#     primary_tfl = [transforms.RandomResizedCrop(img_size)]
    primary_tfl = [] #12.24修改，发现可能是RandomResizedCrop会损害baseline性能（先剔除，后面再说）
    # [RandomResizedCropAndInterpolation(img_size, scale=scale, ratio=ratio, interpolation=interpolation)]
    
    if hflip > 0.:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]
    
    special_tfl = [] #1.10修改，增加一个special_tfl，因为这个部分可能需要混合
    
    # 12.20修改，增加norm&jitter数据增强方法
    # 如果norm_jitter有值
    # 12.20修改，增加HED方法
    if norm_jitter is not None:
        # LAB / HED
        # 根据config文件的空间来决定是哪种
        # 12.26好像有问题，之前不知道顺序，很可怕
        if norm_jitter['methods'] == 'Reinhard':
            # 1.10修改，nj方法的lab和hsv进行统一
            # 1.30修改，lab，hsv和hed方法统一，根据color_space确定
            if norm_jitter['color_space'] == 'LAB' or norm_jitter['color_space'] == 'HSV' or norm_jitter['color_space'] == 'HED':
                color_space = norm_jitter['color_space'] #获取颜色空间名称
                
                #1.30修改，avg和std已经自带分布，在transform里面修改即可
                mean_dataset = [norm_jitter[color_space[0]]['avg'],norm_jitter[color_space[1]]['avg'],norm_jitter[color_space[2]]['avg']]
                std_dataset = [norm_jitter[color_space[0]]['std'],norm_jitter[color_space[1]]['std'],norm_jitter[color_space[2]]['std']]
                std_hyper = norm_jitter['std_hyper']
                distribution = norm_jitter['distribution'] #1.30添加，手工指定分布
                p = norm_jitter['p'] #2.9添加，采用增强的概率，默认是1
                
                special_tfl += [color_norm_jitter(mean=mean_dataset,std=std_dataset,std_hyper=std_hyper,probability=p,color_space=color_space, distribution=distribution)]

#             elif norm_jitter['color_space'] == 'HED':
                
#                 mean_dataset = [norm_jitter['H']['avg'],norm_jitter['E']['avg'],norm_jitter['D']['avg']]
#                 std_dataset = [norm_jitter['H']['std'],norm_jitter['E']['std'],norm_jitter['D']['std']]
#                 std_hyper = norm_jitter['std_hyper']
#                 special_tfl += [hed_norm_jitter(mean=mean_dataset,std=std_dataset,std_hyper=std_hyper,probability=1)]

            elif norm_jitter['color_space'] == 'Random': #1.10增加，混合多种方法，等概率随机进行选取
                distribution = norm_jitter['distribution'] #1.30添加，手工指定分布
                if 'L' in list(norm_jitter.keys()): #2.8修改，测试HED，lab，hsv三者的排列组合
                    mean_dataset = [norm_jitter['L']['avg'],norm_jitter['A']['avg'],norm_jitter['B']['avg']]
                    std_dataset = [norm_jitter['L']['std'],norm_jitter['A']['std'],norm_jitter['B']['std']]
                    std_hyper = norm_jitter['std_hyper']
                    p = norm_jitter['p'] #2.9添加，采用增强的概率，默认是1
                    special_tfl += [color_norm_jitter(mean=mean_dataset,std=std_dataset,std_hyper=std_hyper,probability=p,color_space='LAB',distribution=distribution)]
                
                if 'E' in list(norm_jitter.keys()): #2.8修改，测试HED，lab，hsv三者的排列组合
                    mean_dataset = [norm_jitter['H']['avg'],norm_jitter['E']['avg'],norm_jitter['D']['avg']]
                    std_dataset = [norm_jitter['H']['std'],norm_jitter['E']['std'],norm_jitter['D']['std']]
                    std_hyper = norm_jitter['std_hyper']
                    p = norm_jitter['p'] #2.9添加，采用增强的概率，默认是1
                    # special_tfl += [hed_norm_jitter(mean=mean_dataset,std=std_dataset,std_hyper=std_hyper,probability=1)]
                    # 1.30修改，nj方法统一lab和hed，所以统一用一个即可
                    special_tfl += [color_norm_jitter(mean=mean_dataset,std=std_dataset,std_hyper=std_hyper,probability=p,color_space='HED',distribution=distribution)]
                
                # 2.6修改，增加hsv来random
                if 'h' in list(norm_jitter.keys()): #2.8修改，测试HED，lab，hsv三者的排列组合
                    mean_dataset = [norm_jitter['h']['avg'],norm_jitter['S']['avg'],norm_jitter['V']['avg']]
                    std_dataset = [norm_jitter['h']['std'],norm_jitter['S']['std'],norm_jitter['V']['std']]
                    std_hyper = norm_jitter['std_hyper']
                    p = norm_jitter['p'] #2.9添加，采用增强的概率，默认是1
                    special_tfl += [color_norm_jitter(mean=mean_dataset,std=std_dataset,std_hyper=std_hyper,probability=p,color_space='HSV',distribution=distribution)]
            
            
        if logger is not None:
            logger.info('norm_jitter:', norm_jitter)
        else:
            print('norm_jitter:', norm_jitter)

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = str_to_pil_interp(interpolation)
        if auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            secondary_tfl += [augment_and_mix_transform(auto_augment, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]
        
    # 12.26 顺序很重要
    
    # 12.26修改，增加HED_jitter方法
    if hed_jitter > 0.001:
        secondary_tfl += [HEDJitter(hed_jitter, p=cj_p)]
        print('hed_jitter:', hed_jitter)
    # 1.21修改，增加LAB_jitter方法
    # 2.6修改，增加LAB_jitter_hsv方法，lab_jitter改为list，如果是1个，则是hed方法，3个是hsv方法
    if lab_jitter is not None: 
        if len(lab_jitter) == 1:
            secondary_tfl += [LABJitter(lab_jitter[0], p=cj_p)] #2.13只有一个时，就只能取第一个
        elif len(lab_jitter) == 3 :
            l_factor = lab_jitter[0]
            a_factor = lab_jitter[1]
            b_factor = lab_jitter[2]
            secondary_tfl += [LABJitter_hsv(l_factor,a_factor,b_factor, p=cj_p)]
        print('lab_jitter:', lab_jitter)
        
    # 12.20修改，修改cj的引入规则，固定亮度，对比度，饱和度=0.4，cj参数可变
    # 12.20修改，没有nj时才启用cj
    # 12.25修改，这样HSV过于强势，所以改成Quantify论文中的配置，BC+HSV_light_strong
    if color_jitter is not None : #默认是None
        brightness = color_jitter[0]
        contrast = color_jitter[1]
        saturation = color_jitter[2]
        hue = color_jitter[3]
        if brightness > 0.001 or contrast > 0.001 or saturation > 0.001 or hue > 0.001:
            # 2.13修改，对cj方法封装，增加概率p
            # secondary_tfl += [transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)]
            secondary_tfl += [HSVJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=cj_p)]
            if logger is not None:
                logger.info('color_jitter:',secondary_tfl)
            else:
                print('color_jitter:',secondary_tfl)
            
    final_tfl = []
    # 12.26添加
    if morphology:
        final_tfl += [
            RandomAffine(degrees=0,scale=(0.8,1.2)), #和quantify匹配 #不加弹性变化了
            RandomGaussBlur(radius=[0,0.1]),  #GaussBlur的标准差变化范围[0,0.1]，固定死了
            RandomGaussianNoise(mean=0.0, variance=0.1, amplitude=1.0), #GaussNoise均值方差和幅度，固定mean，std[0,0.1]
        ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [ToNumpy()]
    else:
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
        # 12.26修改，re_mode增加功能，如果是torch str，则用torchvision的
        if re_prob > 0. and re_mode == 'torch':
            final_tfl.append(
                RandomErasing_torch(p=re_prob, value='random')) #随机填值
            print('RandomErasing_torch')
        elif re_prob > 0.:
            final_tfl.append(
                RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits, device='cpu'))
            print('RandomErasing_timm')
            
    #########12.20 debug#######
    # 12.20
    #表明现在创建的是什么的transform
    if logger is not None:
        logger.info('train_transform:\n')
    else:
        print('train_transform:\n')
        
    ##########################     
    if norm_jitter is not None: #1.14修改，判定是否有用nj
        if norm_jitter['color_space'] == 'Random': #1.10修改，增加随机性，随机选择一组执行，各组之间差异在于随机增强的空间不同
            transforms_list = []
            for i in range(len(special_tfl)):
                transforms_list.append(transforms.Compose(primary_tfl + [special_tfl[i]] + secondary_tfl + final_tfl))
            transforms_ = transforms.RandomChoice(transforms_list)
        else:
            transforms_ = transforms.Compose(primary_tfl + special_tfl + secondary_tfl + final_tfl)
    elif random_jitter == True : # 1.30添加，HEDJitter和HSVJitter进行randomaug，注意此时如果
        transforms_list = []
        for i in range(len(secondary_tfl)):
            transforms_list.append(transforms.Compose(primary_tfl + special_tfl + [secondary_tfl[i]] + final_tfl))
        transforms_ = transforms.RandomChoice(transforms_list)
    else:
        transforms_ = transforms.Compose(primary_tfl + special_tfl + secondary_tfl + final_tfl)
        
    if separate: #1.10修改，增加一个randomchoice
        return transforms.Compose(primary_tfl), transforms.Compose(special_tfl), transforms.Compose(secondary_tfl), transforms.Compose(final_tfl)
    else:
        return transforms_  #transforms.Compose(primary_tfl + special_tfl + secondary_tfl + final_tfl)


def transforms_imagenet_eval(
        img_size=224,
        crop_pct=None,
        interpolation='bilinear',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        logger=None):
    crop_pct = crop_pct or DEFAULT_CROP_PCT
    
    # 12.20 测试的时候不需要centor crop
#     if isinstance(img_size, (tuple, list)):
#         assert len(img_size) == 2
#         if img_size[-1] == img_size[-2]:
#             # fall-back to older behaviour so Resize scales to shortest edge if target is square
#             scale_size = int(math.floor(img_size[0] / crop_pct))
#         else:
#             scale_size = tuple([int(x / crop_pct) for x in img_size])
#     else:
#         scale_size = int(math.floor(img_size / crop_pct))

#     tfl = [
#         transforms.Resize(scale_size, interpolation=str_to_interp_mode(interpolation)),
#         transforms.CenterCrop(img_size),
#     ]
    tfl = []
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                     mean=torch.tensor(mean),
                     std=torch.tensor(std))
        ]
    ########### 12.16 debug #######3
    # 12.20 
    if logger is not None:
        logger.info('\nval_transform:\n')
    else :
        print('\nval_transform:\n')
    #######################

    return transforms.Compose(tfl)


def create_transform(
        input_size,
        is_training=False,
        use_prefetcher=False,
        no_aug=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        morphology=None, #12.26添加，是否增加形态学处理
        color_jitter=0.4,
        norm_jitter=None, #12.20加入 nj方法
        hed_jitter=None, #12.26添加，一个theta参数
        lab_jitter=None, #1.21添加，一个theta参数
        random_jitter=None, #1.30添加，jitter是否需要randomaug
        cj_p=1.0, #2.13加入，jitter概率
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        crop_pct=None,
        tf_preprocessing=False,
        separate=False,
        logger=None):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if tf_preprocessing and use_prefetcher:
        assert not separate, "Separate transforms not supported for TF preprocessing"
        from timm.data.tf_preprocessing import TfPreprocessTransform
        transform = TfPreprocessTransform(
            is_training=is_training, size=img_size, interpolation=interpolation)
    else:
        if is_training and no_aug:
            assert not separate, "Cannot perform split augmentation with no_aug"
            transform = transforms_noaug_train(
                img_size,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std)
        elif is_training:
            transform = transforms_imagenet_train(
                img_size,
                scale=scale,
                ratio=ratio,
                hflip=hflip,
                vflip=vflip,
                morphology=morphology, #12.26添加，是否增加形态学处理
                color_jitter=color_jitter,
                norm_jitter=norm_jitter, #12.20加入nj方法，norm_jitter是一个包含所有信息的字典
                hed_jitter=hed_jitter, #12.26添加，一个theta参数
                lab_jitter=lab_jitter, #1.21添加，一个theta参数
                random_jitter=random_jitter, #1.30添加，jitter上是否需要randomaug
                cj_p=cj_p, #2.13加入，jitter概率
                auto_augment=auto_augment,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std,
                re_prob=re_prob,
                re_mode=re_mode,
                re_count=re_count,
                re_num_splits=re_num_splits,
                separate=separate,
                logger=None)
        else:
            assert not separate, "Separate transforms not supported for validation preprocessing"
            transform = transforms_imagenet_eval(
                img_size,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std,
                crop_pct=crop_pct,
                logger=None)

    return transform
