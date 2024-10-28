model_name='vit_tiny_patch16_224'
dataset_path='/root/autodl-tmp/nine_class/standard'
dataset_cn_rn_lab_path='/root/autodl-tmp/nine_class/colornorm'
dataset_cn_rn_hed_path='/root/autodl-tmp/nine_class/colornorm_hed'
dataset_cn_rn_hsv_path='/root/autodl-tmp/nine_class/colornorm_hsv'
b=512
workers=15
lr=0.001
num_classes=8

# # ####### baseline ###############

# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 1e-5 \
# --opt adamw --weight-decay 0.05 --clip-grad 1 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# # # ############ baseline+BC ##################

# # BC+Morphology
# # Brightness [0.65, 1.35]
# # Contrast [0.5, 1.5]
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 1e-5 \
# --opt adamw --weight-decay 0.05 --clip-grad 1 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_M_B.35-C.5_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# # ############ cn baseline #############

# Reinhard方法
python train.py $dataset_cn_rn_lab_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 1e-5 \
--opt adamw --weight-decay 0.05 --clip-grad 1 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment cn-rn-lab_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --native-amp

# Reinhard方法-HED空间
python train.py $dataset_cn_rn_hed_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 1e-5 \
--opt adamw --weight-decay 0.05 --clip-grad 1 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment cn-rn-hed_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --native-amp

# Macenko方法
# 注意，这个只是归一化，似乎没法指定模板
# Macenko是相同模板
# python train_clear.py /mnt/nine_class/colornorm_Macenko \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 1e-5 \
# --opt adamw --weight-decay 0.05 --clip-grad 1 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment cn-macenko_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

################## cn+augmentation ############

# Reinhard方法
# HSVmid对比
# python train.py $dataset_cn_rn_lab_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0.3 0.3 --hflip 0.5 --vflip 0.5 --morphology \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 1e-5 \
# --opt adamw --weight-decay 0.05 --clip-grad 1 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment cn-reinhard_M_BC_HSVmid0.3_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# Reinhard 方法
# HEDlight对比
# python train.py $dataset_cn_rn_lab_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --hed-jitter 0.05 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 1e-5 \
# --opt adamw --weight-decay 0.05 --clip-grad 1 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment cn-reinhard_M_BC_HEDlight0.05_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp
