model_name='mobilenetv3s'
dataset_path='/root/autodl-tmp/nine_class/standard'
dataset_cn_rn_hed_path='/root/autodl-tmp/nine_class/colornorm_hed'
b=128
workers=15
lr=0.1
num_classes=8

# ############# baseline ##############

# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# # ############ baseline+cj ##################

# BC
# Brightness [0.65, 1.35]
# Contrast [0.5, 1.5]
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_M_B.35-C.5_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# ############ nj baseline #############

# # nj-lab-std0
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper 0 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-lab-std0_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# nj-hed-std0
python train.py $dataset_cn_rn_hed_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_HED_randomTrue_n0.yaml' --nj-stdhyper 0 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-hed2-std0_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

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

# nj-cn
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper 0 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment cn_nj-lab-std0_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# nj-Macenko
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Macenko.yaml' \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-Macenko_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# ############ nj ablation #############

# # # nj-lab-std0.05
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper 0.05 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-lab-std0.05_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# # # nj-lab-std0.1
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper 0.1 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-lab-std0.1_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# # # nj-lab-std0.15
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper 0.15 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-lab-std0.15_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp


# # nj-hed-std0.05
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_HED_randomTrue_n0.yaml' --nj-stdhyper 0.05 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-hed-std0.05_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# # nj-hed-std0.1
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_HED_randomTrue_n0.yaml' --nj-stdhyper 0.1 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-hed-std0.1_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# # # nj-hed-std0.15
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_HED_randomTrue_n0.yaml' --nj-stdhyper 0.15 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-hed-std0.15_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

############################## nj+same distribution ##################################

# nj-uniform
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper 0 --nj-distribution uniform \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-lab-std0-uniform_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# nj-normal
# std 0
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper 0 --nj-distribution normal \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-lab-std0-normal_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# nj-normal
# std 0.05
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper 0.05 --nj-distribution normal \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-lab-std0.05-normal_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# nj-normal
# std 0.1
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper 0.1 --nj-distribution normal \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-lab-std0.1-normal_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp


# nj-laplace
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper 0 --nj-distribution laplace \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-lab-std0-laplace_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# nj-laplace
# std 0.05
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper 0.05 --nj-distribution laplace \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-lab-std0.05-laplace_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# nj-laplace
# std 0.1
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper 0.1 --nj-distribution laplace \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-lab-std0.1-laplace_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --amp

# ############ nj combination #############

# # nj-lab-std0+HSVmid0.3

# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0.3 0.3 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper 0 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-lab-std0_M_BC_HSVmid0.3_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp


# ############ nj Random #############
# nj-random(LAB+HED)-std0
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB)_n0.yaml' --nj-stdhyper 0 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random-lab-hed-std0_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# nj-random(LAB+HED)-std0.05
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB)_n0.yaml' --nj-stdhyper 0.05 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random-lab-hed-std0.05_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp


# hsvjitter0.3+hedjitter0.05
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0.3 0.3 --hflip 0.5 --vflip 0.5 --morphology --hed-jitter 0.05 --random-jitter \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_M_BC_random-hsv0.3-hed0.05$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp

# nj-random(LAB+HED+HSV)-std0
# python train.py $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config './norm_jitter/CRC_Random(HED+LAB+HSV)_n0.yaml' --nj-stdhyper 0 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment nj-random(lab+hed+hsv)-std0_M_BC_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --amp