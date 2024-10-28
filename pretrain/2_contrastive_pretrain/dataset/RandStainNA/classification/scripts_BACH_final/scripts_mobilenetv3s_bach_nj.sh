model_name='mobilenetv3s'
dataset_path='/root/autodl-tmp/BACH/standard'
dataset_gray_path='/root/autodl-tmp/BACH/gray'
b=128
workers=15
lr=0.1
num_classes=4
train_path='/root/autodl-tmp/3-RandStainNA/pytorch-image-models/train.py'
nj_lab='/root/autodl-tmp/3-RandStainNA/pytorch-image-models/norm_jitter/BACH/BACH_LAB_randomTrue_n0.yaml'
# # nj-lab-std0
python train.py $dataset_path \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0.35 0.5 0 0 --hflip 0.5 --vflip 0.5 --morphology --nj-config $nj_lab --nj-stdhyper -0.9 --nj-distribution normal --nj-p 0.5 \
--epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment nj-lab-normal-std-0.9-p0.5_M_BC_$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --native-amp
