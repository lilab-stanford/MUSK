model_name='resnet50'
dataset_path='/root/autodl-tmp/nine_class/standard'
dataset_cn_rn_hed_path='/root/autodl-tmp/nine_class/colornorm_hed'
dataset_cn_rn_hsv_path='/root/autodl-tmp/nine_class/colornorm_hsv'
b=128
workers=15
lr=0.1
num_classes=8




# #################### baseline+lab-hsvjitter ######################

# # lab_hsv_light0.1
# # Brightness [0.65, 1.35]
# # Contrast [0.5, 1.5]
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --lab-jitter 0.1 0.1 0.1 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_M_BC_lablight0.1_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# # lab_hsv_mid0.3
# # Brightness [0.65, 1.35]
# # Contrast [0.5, 1.5]
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --lab-jitter 0.3 0.3 0.3 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_M_BC_labhsvmid0.3_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# # lab_hsv_strong0.5
# # Brightness [0.65, 1.35]
# # Contrast [0.5, 1.5]
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --lab-jitter 0.5 0.5 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_M_BC_labhsvstrong0.5_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# ##################### cn-hsv ##############################

# # Reinhard+HSV空间归一化
# python train.py $dataset_cn_rn_hsv_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment cn-rn-hsv_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# #################### nj+hsv ############################

# nj-hsv-std0
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_HSV_randomTrue_n0.yaml' --nj-stdhyper 0 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-hsv-std0_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # nj-normal-std0
# # hsv
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_HSV_randomTrue_n0.yaml' --nj-stdhyper 0 --nj-distribution normal \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-hsv-std0-normal_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # nj-laplace-std0
# # HSV
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_HSV_randomTrue_n0.yaml' --nj-stdhyper 0 --nj-distribution laplace \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-hsv-std0-laplace_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp


# #################### nj+random ############################

# ############## 2-2-HED+LAB ########
# # nj-random(LAB+HED)-std0
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB)_n0.yaml' --nj-stdhyper 0 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random-lab+hed-std0_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # # nj-random(LAB+HED)-std0
# # normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB)_n0.yaml' --nj-stdhyper 0 --nj-distribution normal \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random-lab+hed-normal-std0_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # # nj-random(LAB+HED)-std0
# # laplace
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB)_n0.yaml' --nj-stdhyper 0 --nj-distribution laplace \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random-lab+hed-laplace-std0_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# ############## 2-2-HSV+LAB ########

# # nj-random(HSV+LAB)-std0
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HSV+LAB)_n0.yaml' --nj-stdhyper 0 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random-hsv+lab-std0_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # nj-random(HSV+LAB)-std0
# normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HSV+LAB)_n0.yaml' --nj-stdhyper 0 --nj-distribution normal \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random-hsv+lab-normal-std0_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # # nj-random(HSV+LAB)-std0
# # laplace
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HSV+LAB)_n0.yaml' --nj-stdhyper 0 --nj-distribution laplace \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random-hsv+lab-laplace-std0_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# ############## 2-2-HED+HSV ########

# # nj-random(HED+HSV)-std0
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+HSV)_n0.yaml' --nj-stdhyper 0 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random-hed+hsv-std0_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # nj-random(HED+HSV)-std0
# normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+HSV)_n0.yaml' --nj-stdhyper 0 --nj-distribution normal \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random-hed+hsv-normal-std0_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # # nj-random(HED+HSV)-std0
# # laplace
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+HSV)_n0.yaml' --nj-stdhyper 0 --nj-distribution laplace \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random-hed+hsv-laplace-std0_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# ############### 3-3 ############
# # nj-random(LAB+HED+HSV)-std0
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB+HSV)_n0.yaml' --nj-stdhyper 0 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random-lab+hed+hsv-std0_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # nj-random(LAB+HED+HSV)-std0
# normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB+HSV)_n0.yaml' --nj-stdhyper 0 --nj-distribution normal \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random-lab+hed+hsv-normal-std0_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # # nj-random(LAB+HED+HSV)-std0
# # laplace
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB+HSV)_n0.yaml' --nj-stdhyper 0 --nj-distribution laplace \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random-lab+hed+hsv-laplace-std0_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# ############### 2-12补充实验 ############

# nj-normal-std0-p1
# hed
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_HED_randomTrue_n0.yaml' --nj-stdhyper 0 --nj-distribution normal \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-hed-std0-normal_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# nj-normal-std0-p0.5
# lab
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper 0 --nj-distribution normal --nj-p 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-lab-std0-p0.5-normal_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # nj-normal-std0-p0.5
# # hed
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_HED_randomTrue_n0.yaml' --nj-stdhyper 0 --nj-distribution normal --nj-p 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-hed-std0-p0.5-normal_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # nj-normal-std0-p0.5
# # hsv
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_HSV_randomTrue_n0.yaml' --nj-stdhyper 0 --nj-distribution normal --nj-p 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-hsv-std0-p0.5-normal_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # # nj-random(LAB+HED)-std0-p0.5-normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB)_n0.yaml' --nj-stdhyper 0 --nj-distribution normal --nj-p 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random-lab+hed-normal-std0-p0.5_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # # nj-random(LAB+HED+HSV)-std0-p0.5-normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB+HSV)_n0.yaml' --nj-stdhyper 0 --nj-distribution normal --nj-p 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random-lab+hed+hsv-normal-std0-p0.5_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp


############# 2.18补充实验 #########################

# nj-normal-lab
# dynamic
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper 0 --nj-distribution normal --nj-dynamic --dynamic-factor 1 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-lab-dynamic1-std0-normal_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # nj-normal-hed
# # dynamic
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_HED_randomTrue_n0.yaml' --nj-stdhyper 0 --nj-distribution normal --nj-dynamic --dynamic-factor 1 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-hed-dynamic1-std0-normal_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # nj-normal-hsv
# # dynamic
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_HSV_randomTrue_n0.yaml' --nj-stdhyper 0 --nj-distribution normal --nj-dynamic --dynamic-factor 1 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-hsv-dynamic1-std0-normal_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # # nj-random(LAB+HED)-std0-dynamic1-normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB)_n0.yaml' --nj-stdhyper 0 --nj-distribution normal --nj-dynamic --dynamic-factor 1 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random-lab+hed-normal-dynamic1-std0_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # # nj-random(HED+HSV)-std0-dynamic1-normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+HSV)_n0.yaml' --nj-stdhyper 0 --nj-distribution normal --nj-dynamic --dynamic-factor 1 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random-lab+hsv-normal-dynamic1-std0_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # # nj-random(HSV+LAB)-std0-dynamic1-normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HSV+LAB)_n0.yaml' --nj-stdhyper 0 --nj-distribution normal --nj-dynamic --dynamic-factor 1 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random-lab+hsv-normal-dynamic1-std0_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # nj-random(LAB+HED+HSV)-std0-dynamic1-normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB+HSV)_n0.yaml' --nj-stdhyper 0 --nj-distribution normal --nj-dynamic --dynamic-factor 1 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random-lab+hed+hsv-dynamic1-normal-std0_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

############# 2.25补充实验 #########################
# lab_hsv_light0.1
# Brightness [0.65, 1.35]
# Contrast [0.5, 1.5]
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --lab-jitter 0.1 0.1 0.1 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment add_baseline_M_BC_lablight0.1_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# # lab_hsv_light0.1
# # Brightness [0.65, 1.35]
# # Contrast [0.5, 1.5]
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --lab-jitter 0.1 0.1 0.1 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment add2_baseline_M_BC_lablight0.1_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# # # nj-random(LAB+HED)-std0-p0.5-normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB)_n0.yaml' --nj-stdhyper 0 --nj-distribution normal --nj-p 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment add_nj-random-lab+hed-normal-std0-p0.5_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # # nj-random(LAB+HED)-std0-p0.5-normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB)_n0.yaml' --nj-stdhyper 0.05 --nj-distribution normal --nj-p 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment add_nj-random-lab+hed-normal-std0.05-p0.5_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # # nj-random(LAB+HED)-std0-p0.5-normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB)_n0.yaml' --nj-stdhyper 0.1 --nj-distribution normal --nj-p 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment add_nj-random-lab+hed-normal-std0.1-p0.5_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # # nj-random(LAB+HED)-std0-p0.5-normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB)_n0.yaml' --nj-stdhyper -0.1 --nj-distribution normal --nj-p 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment add_nj-random-lab+hed-normal-std-0.1-p0.5_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # # nj-random(LAB+HED)-std0-p0.5-normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB)_n0.yaml' --nj-stdhyper -0.3 --nj-distribution normal --nj-p 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment add_nj-random-lab+hed-normal-std-0.3-p0.5_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # nj-random(LAB+HED)-std0-p0.5-normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB)_n0.yaml' --nj-stdhyper -0.5 --nj-distribution normal --nj-p 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment add_nj-random-lab+hed-normal-std-0.5-p0.5_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # # nj-random(LAB+HED)-std0-p0.5-normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB)_n0.yaml' --nj-stdhyper -0.6 --nj-distribution normal --nj-p 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment add_nj-random-lab+hed-normal-std-0.6-p0.5_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # # nj-random(LAB+HED)-std0-p0.5-normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB)_n0.yaml' --nj-stdhyper -0.7 --nj-distribution normal --nj-p 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment add_nj-random-lab+hed-normal-std-0.7-p0.5_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # # nj-random(LAB+HED)-std0-p0.5-normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB)_n0.yaml' --nj-stdhyper -0.5 --nj-distribution normal --nj-p 1 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment add_nj-random-lab+hed-normal-std-0.5-p1_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # # nj-random(LAB+HED)-std0-p0.5-normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB)_n0.yaml' --nj-stdhyper -0.6 --nj-distribution normal --nj-p 1 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment add_nj-random-lab+hed-normal-std-0.6-p1_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # # nj-random(LAB+HED)-std0-p0.5-normal
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB)_n0.yaml' --nj-stdhyper -0.7 --nj-distribution normal --nj-p 1 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment add_nj-random-lab+hed-normal-std-0.7-p1_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp



############### 2.26补充实验 ##################
# Reinhard方法
python train.py $dataset_cn_rn_lab_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment cn-rn_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# nj-normal-p0.5
# std 0
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper 0 --nj-distribution normal --nj-p 0.5 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-lab-normal-std0-p0.5_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# nj-normal-p0.5
# std -0.8
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.8 --nj-distribution normal --nj-p 0.5 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-lab-normal-std-0.8-p0.5_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# nj-normal-p0.5
# std -0.7
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.7 --nj-distribution normal --nj-p 0.5 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-lab-normal-std-0.7-p0.5_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# nj-normal-p0.5
# std -0.5
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.5 --nj-distribution normal --nj-p 0.5 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-lab-normal-std-0.5-p0.5_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# nj-normal-p0.5
# std -0.3
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.3 --nj-distribution normal --nj-p 0.5 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-lab-normal-std-0.3-p0.5_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# nj-normal-p0.5
# hed
# std -0.8
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_HED_randomTrue_n0.yaml' --nj-stdhyper -0.8 --nj-distribution normal --nj-p 0.5 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-hed-normal-std-0.8-p0.5_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# nj-normal-p0.5
# hed
# std -0.7
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_HED_randomTrue_n0.yaml' --nj-stdhyper -0.7 --nj-distribution normal --nj-p 0.5 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-hed-normal-std-0.7-p0.5_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# nj-normal-p0.5
# hed
# std -0.5
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_HED_randomTrue_n0.yaml' --nj-stdhyper -0.5 --nj-distribution normal --nj-p 0.5 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-hed-normal-std-0.5-p0.5_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# nj-normal-p0.5
# hed
# std -0.3
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_HED_randomTrue_n0.yaml' --nj-stdhyper -0.3 --nj-distribution normal --nj-p 0.5 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-hed-normal-std-0.3-p0.5_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# nj-normal-p0.5
# hsv
# std -0.8
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_HSV_randomTrue_n0.yaml' --nj-stdhyper -0.8 --nj-distribution normal --nj-p 0.5 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-hsv-normal-std-0.8-p0.5_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# nj-normal-p0.5
# hsv
# std -0.7
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_HSV_randomTrue_n0.yaml' --nj-stdhyper -0.7 --nj-distribution normal --nj-p 0.5 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-hsv-normal-std-0.7-p0.5_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# nj-normal-p0.5
# hsv
# std -0.5
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_HSV_randomTrue_n0.yaml' --nj-stdhyper -0.5 --nj-distribution normal --nj-p 0.5 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-hsv-normal-std-0.5-p0.5_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# nj-normal-p0.5
# hsv
# std -0.3
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_HSV_randomTrue_n0.yaml' --nj-stdhyper -0.3 --nj-distribution normal --nj-p 0.5 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-hsv-normal-std-0.3-p0.5_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp
