model_name='resnet18'
dataset_path='../../1-Datasets/Classification/CRC_8/standard'
b=32
workers=8
lr=0.1
num_classes=8
train_path='train.py'
# # ####### baseline ###############

# python $train_path $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# # ############ baseline+cj ##################

# BC
# Brightness [0.65, 1.35]
# Contrast [0.5, 1.5]
python $train_path $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment baseline_M_B.35-C.5_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --native-amp

#python train.py '/mnt/e/PythonProject/3-RandStainNA/1-Datasets/Classification/CRC_8/standard' --model resnet18 --num-classes 8 --dataset 'torch/image_folder' --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --epochs 50 --batch-size 32 --validation-batch-size 32 --warmup-epochs 3 --warmup-lr 0.001 --opt sgd --weight-decay 1e-4 --momentum 0.9 --sched cosine --lr 0.1 --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 --experiment baseline_M_B.35-C.5_resnet18 -j 15 --no-prefetcher --pin-mem --seed 97 --native-amp

#python train.py E:\PythonProject\1-Datasets\Classification\CRC_8\standard --model resnet18 --num-classes 8 --dataset 'torch/image_folder' --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --epochs 50 --batch-size 32 --validation-batch-size 32 --warmup-epochs 3 --warmup-lr 0.001 --opt sgd --weight-decay 1e-4 --momentum 0.9 --sched cosine --lr 0.1 --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 --experiment baseline_M_B.35-C.5_resnet18 -j 0 --no-prefetcher --pin-mem --seed 97 --native-amp

######### 注意，HSV的扰动是在基本的BC基础上的，所以都得考虑进去#######
# # HSV_light
# Brightness [0.65, 1.35]
# Contrast [0.5, 1.5]
# # hue [-0.1, 0.1]
# # 饱和度 [0.9, 1.1]
#python $train_path $dataset_path \
#--model $model_name \
#--num-classes $num_classes --dataset 'torch/image_folder' \
#--color-jitter 0.35 0.5 0.1 0.1 --hflip 0.5 --vflip 0.5 --morphology \
#--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
#--opt sgd --weight-decay 1e-4 --momentum 0.9 \
#--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
#--experiment baseline_M_BC_HSVlight0.1_$model_name \
#-j $workers --no-prefetcher --pin-mem \
#--seed 97 --native-amp
#
## HSV_strong
## Brightness [0.65, 1.35]
## Contrast [0.5, 1.5]
## hue [-0.5, 0.5]
## 饱和度 [0.5, 1.5]
#python $train_path $dataset_path \
#--model $model_name \
#--num-classes $num_classes --dataset 'torch/image_folder' \
#--color-jitter 0.35 0.5 0.5 0.5 --hflip 0.5 --vflip 0.5 --morphology \
#--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
#--opt sgd --weight-decay 1e-4 --momentum 0.9 \
#--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
#--experiment baseline_M_BC_HSVstrong0.5_$model_name \
#-j $workers --no-prefetcher --pin-mem \
#--seed 97 --native-amp
#
## # ############ baseline+hed_jitter ##################
#
## hed_light
## Brightness [0.65, 1.35]
## Contrast [0.5, 1.5]
#python $train_path $dataset_path \
#--model $model_name \
#--num-classes $num_classes --dataset 'torch/image_folder' \
#--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --hed-jitter 0.05 \
#--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
#--opt sgd --weight-decay 1e-4 --momentum 0.9 \
#--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
#--experiment baseline_M_BC_hedlight0.05_$model_name \
#-j $workers --no-prefetcher --pin-mem \
#--seed 97 --native-amp
#
## hed_strong
## Brightness [0.65, 1.35]
## Contrast [0.5, 1.5]
#python $train_path $dataset_path \
#--model $model_name \
#--num-classes $num_classes --dataset 'torch/image_folder' \
#--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --hed-jitter 0.2 \
#--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
#--opt sgd --weight-decay 1e-4 --momentum 0.9 \
#--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
#--experiment baseline_M_BC_hedstrong0.2_$model_name \
#-j $workers --no-prefetcher --pin-mem \
#--seed 97 --native-amp
#
## # ############ baseline+lab_jitter ##################
#
## lab_light
## Brightness [0.65, 1.35]
## Contrast [0.5, 1.5]
#python $train_path $dataset_path \
#--model $model_name \
#--num-classes $num_classes --dataset 'torch/image_folder' \
#--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --lab-jitter 0.05 \
#--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
#--opt sgd --weight-decay 1e-4 --momentum 0.9 \
#--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
#--experiment baseline_M_BC_lablight0.05_$model_name \
#-j $workers --no-prefetcher --pin-mem \
#--seed 97 --native-amp
#
## lab_strong
## Brightness [0.65, 1.35]
## Contrast [0.5, 1.5]
#python $train_path $dataset_path \
#--model $model_name \
#--num-classes $num_classes --dataset 'torch/image_folder' \
#--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --lab-jitter 0.2 \
#--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
#--opt sgd --weight-decay 1e-4 --momentum 0.9 \
#--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
#--experiment baseline_M_BC_labstrong0.2_$model_name \
#-j $workers --no-prefetcher --pin-mem \
#--seed 97 --native-amp
#
## #################### baseline+lab-hsvjitter ######################
#
## lab_hsv_light0.1
## Brightness [0.65, 1.35]
## Contrast [0.5, 1.5]
#python $train_path $dataset_path \
#--model $model_name \
#--num-classes $num_classes --dataset 'torch/image_folder' \
#--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --lab-jitter 0.1 0.1 0.1 \
#--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
#--opt sgd --weight-decay 1e-4 --momentum 0.9 \
#--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
#--experiment baseline_M_BC_labhsvlight0.1_$model_name \
#-j $workers --no-prefetcher --pin-mem \
#--seed 97 --native-amp
#
## lab_hsv_strong0.5
## Brightness [0.65, 1.35]
## Contrast [0.5, 1.5]
#python $train_path $dataset_path \
#--model $model_name \
#--num-classes $num_classes --dataset 'torch/image_folder' \
#--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --lab-jitter 0.5 0.5 0.5 \
#--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
#--opt sgd --weight-decay 1e-4 --momentum 0.9 \
#--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
#--experiment baseline_M_BC_labhsvstrong0.5_$model_name \
#-j $workers --no-prefetcher --pin-mem \
#--seed 97 --native-amp
