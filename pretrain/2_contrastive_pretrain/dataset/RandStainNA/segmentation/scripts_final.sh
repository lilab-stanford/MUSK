ep=30
b=4
lr=2e-5

dataset_path='/root/autodl-tmp/MoNuSeg2018/standard'
dataset_aug_path='/root/autodl-tmp/MoNuSeg2018/standard_aug'
cn_path='/root/autodl-tmp/MoNuSeg2018/colornorm' 
cn_hed_path='/root/autodl-tmp/MoNuSeg2018/colornorm_hed' 
cn_hsv_path='/root/autodl-tmp/MoNuSeg2018/colornorm_hsv' 

# # baseline
python CIA.py \
--dataset $dataset_path \
--epochs $ep --batch-size $b \
--lr $lr \
--output baseline
--seed 97 \

# cn
python CIA.py \
--dataset $cn_path \
--epochs $ep --batch-size $b \
--lr $lr \
--output cn
--seed 97 \

# # cn-hed
python CIA.py \
--dataset $cn_hed_path \
--epochs $ep --batch-size $b \
--lr $lr \
--output cn-hed
--seed 97 \

# # cn-hsv
python CIA.py \
--dataset $cn_hsv_path \
--epochs $ep --batch-size $b \
--lr $lr \
--output cn-hsv
--seed 97 \

# # hsv light jitter
python CIA.py \
--dataset $dataset_path \
--color-jitter 0.35 0.5 0.1 0.1 \
--epochs $ep --batch-size $b \
--lr $lr \
--output hsvlight-p1
--seed 97 \

# hsv light jitter p=0.5
python CIA.py \
--dataset $dataset_path \
--color-jitter 0.35 0.5 0.1 0.1 --cj-p 0.5 \
--epochs $ep --batch-size $b \
--lr $lr \
--output hsvlight-p0.5
--seed 97 \

# hedlight-p1
python CIA.py \
--dataset $dataset_path \
--hed-jitter 0.05 \
--epochs $ep --batch-size $b \
--lr $lr \
--output hedlight-p1
--seed 97 \

# hedlight-p0.5
python CIA.py \
--dataset $dataset_path \
--hed-jitter 0.05 \
--epochs $ep --batch-size $b --cj-p 0.5 \
--lr $lr \
--output hedlight-p0.5
--seed 97 \

# labhsvlight-p1
python CIA.py \
--dataset $dataset_path \
--lab-jitter 0.1 0.1 0.1 \
--epochs $ep --batch-size $b \
--lr $lr \
--output labhsvlight-p1
--seed 97 \

# labhsvlight-p0.5
python CIA.py \
--dataset $dataset_path \
--lab-jitter 0.1 0.1 0.1 --cj-p 0.5 \
--epochs $ep --batch-size $b --cj-p 0.5 \
--lr $lr \
--output labhsvlight-p0.5
--seed 97 \

# ############## RandStainNA ############

python CIA.py \
--dataset $dataset_path \
--nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.9 --nj-distribution normal --nj-p 0.5 \
--epochs $ep --batch-size $b \
--lr $lr \
--output nj-lab-std-0.9-p0.5
--seed 97 \


############# 2.27 tta ##########

python CIA_tta.py \
--dataset $dataset_path \
--nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.9 --nj-distribution normal --nj-p 1 --nj-TTA 1 \
--epochs $ep --batch-size $b \
--lr $lr \
--output nj-tta1-O-lab-std-0.9-p1-thresh0.8 \
--seed 97 \

# python CIA_tta.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.9 --nj-distribution normal --nj-p 1 --nj-TTA 1 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-tta1-NO-lab-std-0.9-p1-thresh0.8 \
# --seed 97 \


############ 3.1 ################

# hedlight-p0.5-tta
python CIA_tta.py \
--dataset $dataset_path \
--hed-jitter 0.05 \
--epochs $ep --batch-size $b --cj-p 1 --nj-TTA 1 \
--lr $lr \
--output tta1-O-hedlight0.05-p1
--seed 97 \


# hsv light jitter p=1-tta1
python CIA_tta.py \
--dataset $dataset_path \
--color-jitter 0.35 0.5 0.1 0.1 --cj-p 1 --nj-TTA 1 \
--epochs $ep --batch-size $b \
--lr $lr \
--output tta1-O-hsvlight0.1-p1
--seed 97 \
