model_name='mobilenetv3s'
dataset_path='/root/autodl-tmp/BACH/standard'
dataset_cn_rn_lab_path='/root/autodl-tmp/BACH/colornorm'
dataset_cn_rn_hed_path='/root/autodl-tmp/BACH/colornorm_hed'
dataset_cn_rn_hsv_path='/root/autodl-tmp/BACH/colornorm_hsv'
b=128
workers=15
lr=0.1
num_classes=4
train_path='/root/autodl-tmp/3-RandStainNA/pytorch-image-models/train.py'

# # ############ cn baseline #############

# Reinhard方法
python $train_path $dataset_cn_rn_lab_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment cn-rn-lab_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --native-amp

# Reinhard+HED空间归一化
python $train_path $dataset_cn_rn_hed_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment cn-rn-hed_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --native-amp

# Reinhard+HSV空间归一化
python $train_path $dataset_cn_rn_hsv_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment cn-rn-hsv_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --native-amp