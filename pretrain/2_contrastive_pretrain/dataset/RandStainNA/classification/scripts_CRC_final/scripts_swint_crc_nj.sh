model_name='swin_tiny_patch4_window7_224'
dataset_path='/root/autodl-tmp/nine_class/standard'
dataset_cn_rn_lab_path='/root/autodl-tmp/nine_class/colornorm'
b=512
workers=15
lr=0.001
num_classes=8
train_path='/root/autodl-tmp/3-RandStainNA/pytorch-image-models/train.py'

# # # ############ baseline+BC ##################

# # BC+Morphology
# # Brightness [0.65, 1.35]
# # Contrast [0.5, 1.5]
# python $train_path $dataset_path \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 1e-5 \
# --opt adamw --weight-decay 0.05 --clip-grad 1 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_M_B.35-C.5_$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# ############ nj baseline #############

# nj-lab-normal-p0.5
# std -0.5
python $train_path $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config '../norm_jitter/CRC_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.5 --nj-distribution normal --nj-p 0.5 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 1e-5 \
--opt adamw --weight-decay 0.05 --clip-grad 1 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-lab-normal-std-0.5-p0.5_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --native-amp

# nj-hed-normal-p0.5
# std -0.5
python $train_path $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config '../norm_jitter/CRC_HED_randomTrue_n0.yaml' --nj-stdhyper -0.5 --nj-distribution normal --nj-p 0.5 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 1e-5 \
--opt adamw --weight-decay 0.05 --clip-grad 1 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-hed-normal-std-0.5-p0.5_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --native-amp

# nj-hsv-normal-p0.5
# std -0.5
python $train_path $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config '../norm_jitter/CRC_HSV_randomTrue_n0.yaml' --nj-stdhyper -0.5 --nj-distribution normal --nj-p 0.5 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 1e-5 \
--opt adamw --weight-decay 0.05 --clip-grad 1 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-hsv-normal-std-0.5-p0.5_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --native-amp

# ############ nj Random #############

# nj-random(LAB+HED+HSV)-std0-p0.5-normal
python $train_path $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config '../norm_jitter/CRC_Random(HED+LAB+HSV)_n0.yaml' --nj-stdhyper 0 --nj-distribution normal --nj-p 0.5 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 1e-5 \
--opt adamw --weight-decay 0.05 --clip-grad 1 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-random-lab-hed-hsv-normal-std0-p0.5_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --native-amp

