ep=30
b=4
lr=2e-5

dataset_path='/root/autodl-tmp/MoNuSeg2018/standard'
dataset_aug_path='/root/autodl-tmp/MoNuSeg2018/standard_aug'
cn_path='/root/autodl-tmp/MoNuSeg2018/colornorm' 
cn_hed_path='/root/autodl-tmp/MoNuSeg2018/colornorm_hed' 
cn_hsv_path='/root/autodl-tmp/MoNuSeg2018/colornorm_hsv' 
cn_a_path='/root/autodl-tmp/MoNuSeg2018/colornorm_A' 
cn_t_path='/root/autodl-tmp/MoNuSeg2018/colornorm_T' 

# # baseline
# python CIA.py \
# --dataset $dataset_path \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output baseline
# --seed 97 \

# cn
# python CIA.py \
# --dataset $cn_path \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output cn
# --seed 97 \

# # cn-hed
# python CIA.py \
# --dataset $cn_hed_path \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output cn-hed
# --seed 97 \

# # cn-hsv
# python CIA.py \
# --dataset $cn_hsv_path \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output cn-hsv
# --seed 97 \

# # cn_tempA
# python CIA.py \
# --dataset $cn_a_path \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output cn_A
# --seed 97 \

# # cn_tempT
# python CIA.py \
# --dataset $cn_t_path \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output cn_T
# --seed 97 \

# # hsv light jitter
# python CIA.py \
# --dataset $dataset_path \
# --color-jitter 0.35 0.5 0.1 0.1 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output hsvlight-p1
# --seed 97 \

# # hsv light jitter p=0.5
# python CIA.py \
# --dataset $dataset_path \
# --color-jitter 0.35 0.5 0.1 0.1 --cj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output hsvlight-p0.5
# --seed 97 \

# cn-hsv light jitter p=0.5
# python CIA.py \
# --dataset $cn_path \
# --color-jitter 0.35 0.5 0.1 0.1 --cj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output cn_hsvlight-p0.5
# --seed 97 \


# # hsv strong jitter p=0.5
# python CIA.py \
# --dataset $dataset_path \
# --color-jitter 0.35 0.5 0.5 0.5 --cj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output hsvstrong-p0.5
# --seed 97 \

# hsv strong jitter p=0.5
# python CIA.py \
# --dataset $cn_path \
# --color-jitter 0.35 0.5 0.5 0.5 --cj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output cn_hsvstrong-p0.5
# --seed 97 \

# # hedlight-p1
# python CIA.py \
# --dataset $dataset_path \
# --hed-jitter 0.05 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output hedlight-p1
# --seed 97 \

# # hedlight-p0.5
# python CIA.py \
# --dataset $dataset_path \
# --hed-jitter 0.05 \
# --epochs $ep --batch-size $b --cj-p 0.5 \
# --lr $lr \
# --output hedlight-p0.5
# --seed 97 \

# # hedstrong-p0.5
# python CIA.py \
# --dataset $dataset_path \
# --hed-jitter 0.2 \
# --epochs $ep --batch-size $b --cj-p 0.5 \
# --lr $lr \
# --output hedstrong0.2-p0.5
# --seed 97 \

# # lablight-p1
# python CIA.py \
# --dataset $dataset_path \
# --lab-jitter 0.05 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output lablight-p1
# --seed 97 \

# # lablight-p0.5
# python CIA.py \
# --dataset $dataset_path \
# --lab-jitter 0.05 \
# --epochs $ep --batch-size $b --cj-p 0.5 \
# --lr $lr \
# --output lablight-p0.5
# --seed 97 \

# labstrong-p0.5
# python CIA.py \
# --dataset $dataset_path \
# --lab-jitter 0.2 \
# --epochs $ep --batch-size $b --cj-p 0.5 \
# --lr $lr \
# --output labstrong-p0.5
# --seed 97 \

# # labhsvlight-p1
# python CIA.py \
# --dataset $dataset_path \
# --lab-jitter 0.1 0.1 0.1 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output labhsvlight-p1
# --seed 97 \

# # labhsvlight-p0.5
# python CIA.py \
# --dataset $dataset_path \
# --lab-jitter 0.1 0.1 0.1 --cj-p 0.5 \
# --epochs $ep --batch-size $b --cj-p 0.5 \
# --lr $lr \
# --output labhsvlight-p0.5
# --seed 97 \

# labhsvstrong-p0.5
# python CIA.py \
# --dataset $dataset_path \
# --lab-jitter 0.5 0.5 0.5 --cj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output labhsvstrong-p0.5
# --seed 97 \
# ############## RandStainNA ############

# python CIA.py \
# --dataset '/root/autodl-tmp/MoNuSeg2018/standard' \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.9 --nj-distribution normal --nj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-lab-std-0.9-p0.5
# --seed 97 \

# python CIA.py \
# --dataset '/root/autodl-tmp/MoNuSeg2018/standard' \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.8 --nj-distribution normal --nj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-lab-std-0.8-p0.5
# --seed 97 \

# python CIA.py \
# --dataset '/root/autodl-tmp/MoNuSeg2018/standard' \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.7 --nj-distribution normal --nj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-lab-std-0.7-p0.5
# --seed 97 \

# python CIA.py \
# --dataset $cn_path \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.7 --nj-distribution normal --nj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output cn_nj-lab-std-0.7-p0.5
# --seed 97 \

# python CIA.py \
# --dataset $cn_path \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.8 --nj-distribution normal --nj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output cn_nj-lab-std-0.8-p0.5
# --seed 97 \

# python CIA.py \
# --dataset $cn_path \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.5 --nj-distribution normal --nj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output cn_nj-lab-std-0.5-p0.5
# --seed 97 \

# python CIA.py \
# --dataset $cn_path \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.3 --nj-distribution normal --nj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output cn_nj-lab-std-0.3-p0.5
# --seed 97 \


# python CIA.py \
# --dataset '/root/autodl-tmp/MoNuSeg2018/standard' \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.5 --nj-distribution normal --nj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-lab-std-0.5-p0.5
# --seed 97 \

# python CIA.py \
# --dataset '/root/autodl-tmp/MoNuSeg2018/standard' \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.3 --nj-distribution normal --nj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-lab-std-0.3-p0.5
# --seed 97 \

# python CIA.py \
# --dataset '/root/autodl-tmp/MoNuSeg2018/standard' \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.1 --nj-distribution normal --nj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-lab-std-0.1-p0.5
# --seed 97 \

# ########### hed ###########
# python CIA.py \
# --dataset '/root/autodl-tmp/MoNuSeg2018/standard' \
# --nj-config './norm_jitter/MoNuSeg_HED_randomTrue_n0.yaml' --nj-stdhyper -0.9 --nj-distribution normal --nj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-hed-std-0.9-p0.5
# --seed 97 \

# python CIA.py \
# --dataset '/root/autodl-tmp/MoNuSeg2018/standard' \
# --nj-config './norm_jitter/MoNuSeg_HED_randomTrue_n0.yaml' --nj-stdhyper -0.8 --nj-distribution normal --nj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-hed-std-0.8-p0.5
# --seed 97 \

# python CIA.py \
# --dataset '/root/autodl-tmp/MoNuSeg2018/standard' \
# --nj-config './norm_jitter/MoNuSeg_HED_randomTrue_n0.yaml' --nj-stdhyper -0.7 --nj-distribution normal --nj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-hed-std-0.7-p0.5
# --seed 97 \


# python CIA.py \
# --dataset '/root/autodl-tmp/MoNuSeg2018/standard' \
# --nj-config './norm_jitter/MoNuSeg_HED_randomTrue_n0.yaml' --nj-stdhyper -0.5 --nj-distribution normal --nj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-hed-std-0.5-p0.5
# --seed 97 \

# python CIA.py \
# --dataset '/root/autodl-tmp/MoNuSeg2018/standard' \
# --nj-config './norm_jitter/MoNuSeg_HED_randomTrue_n0.yaml' --nj-stdhyper -0.3 --nj-distribution normal --nj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-hed-std-0.3-p0.5
# --seed 97 \

# python CIA.py \
# --dataset '/root/autodl-tmp/MoNuSeg2018/standard' \
# --nj-config './norm_jitter/MoNuSeg_HED_randomTrue_n0.yaml' --nj-stdhyper -0.1 --nj-distribution normal --nj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-hed-std-0.1-p0.5
# --seed 97 \

############ nj-dynamic #####################

# python CIA.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.7 --nj-distribution normal --nj-dynamic --dynamic-factor 1  \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-lab-std-0.7-dynamic1
# --seed 97 \

# python CIA.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.7 --nj-distribution normal --nj-dynamic --dynamic-factor 0.8  \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-lab-std-0.7-dynamic0.8
# --seed 97 \

# python CIA.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.7 --nj-distribution normal --nj-dynamic --dynamic-factor 0.5  \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-lab-std-0.7-dynamic0.5
# --seed 97 \

# python CIA.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_HED_randomTrue_n0.yaml' --nj-stdhyper -0.7 --nj-distribution normal --nj-dynamic --dynamic-factor 1  \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-hed-std-0.7-dynamic1
# --seed 97 \

# python CIA.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_HED_randomTrue_n0.yaml' --nj-stdhyper -0.7 --nj-distribution normal --nj-dynamic --dynamic-factor 0.8  \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-hed-std-0.7-dynamic0.8
# --seed 97 \

# python CIA.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_HED_randomTrue_n0.yaml' --nj-stdhyper -0.7 --nj-distribution normal --nj-dynamic --dynamic-factor 0.5  \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-hed-std-0.7-dynamic0.5
# --seed 97 \

########### 2.27 ############
# python CIA.py \
# --dataset $dataset_aug_path \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.7 --nj-distribution normal --nj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output aug_nj-lab-std-0.7-p0.5
# --seed 97 \

# python CIA.py \
# --dataset $dataset_aug_path \
# --nj-config './norm_jitter/MoNuSeg_HED_randomTrue_n0.yaml' --nj-stdhyper -0.7 --nj-distribution normal --nj-p 0.5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output aug_nj-hed-std-0.7-p0.5
# --seed 97 \

############# 2.27 tta ##########
# python CIA_tta.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.7 --nj-distribution normal --nj-p 0.5 --nj-TTA 3 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-tta3-O-lab-std-0.7-p0.5-thresh0.8-lr*0.9 \
# --seed 97 \

# python CIA_tta.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.7 --nj-distribution normal --nj-p 1 --nj-TTA 3 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-tta3-O-lab-std-0.7-p1-thresh0.8 \
# --seed 97 \

# python CIA_tta.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.8 --nj-distribution normal --nj-p 1 --nj-TTA 3 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-tta3-O-lab-std-0.8-p1-thresh0.8 \
# --seed 97 \

# python CIA_tta.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.9 --nj-distribution normal --nj-p 1 --nj-TTA 3 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-tta3-O-lab-std-0.9-p1-thresh0.8 \
# --seed 97 \

# python CIA_tta.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.9 --nj-distribution normal --nj-p 1 --nj-TTA 5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-tta5-O-lab-std-0.9-p1-thresh0.8 \
# --seed 97 \

# python CIA_tta.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.95 --nj-distribution normal --nj-p 1 --nj-TTA 3 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-tta3-O-lab-std-0.95-p1-thresh0.8 \
# --seed 97 \

# python CIA_tta.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.95 --nj-distribution normal --nj-p 1 --nj-TTA 5 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-tta5-O-lab-std-0.95-p1-thresh0.8 \
# --seed 97 \

# python CIA_tta.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.9 --nj-distribution normal --nj-p 1 --nj-TTA 1 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-tta1-O-lab-std-0.9-p1-thresh0.8 \
# --seed 97 \

# python CIA_tta.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.9 --nj-distribution normal --nj-p 1 --nj-TTA 1 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-tta1-NO-lab-std-0.9-p1-thresh0.8 \
# --seed 97 \

# # python CIA_tta.py \
# # --dataset $dataset_path \
# # --nj-config './norm_jitter/MoNuSeg_LAB_randomTrue_n0.yaml' --nj-stdhyper -0.9 --nj-distribution normal --nj-p 1 \
# # --epochs $ep --batch-size $b \
# # --lr $lr \
# # --output nj-lab-std-0.9-p1-thresh0.8 \
# # --seed 97 \


# python CIA_tta.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_HSV_randomTrue_n0.yaml' --nj-stdhyper -0.9 --nj-distribution normal --nj-p 1 --nj-TTA 3 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-tta3-O-hsv-std-0.9-p1-thresh0.8 \
# --seed 97 \

# python CIA_tta.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_HSV_randomTrue_n0.yaml' --nj-stdhyper -0.7 --nj-distribution normal --nj-p 1 --nj-TTA 3 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-tta3-O-hsv-std-0.7-p1-thresh0.8 \
# --seed 97 \

# python CIA_tta.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_LAB+HED+HSV_randomTrue_n0.yaml' --nj-stdhyper -0.9 --nj-distribution normal --nj-p 1 --nj-TTA 3 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-tta3-O-lab-HED-HSV-std-0.9-p1-thresh0.8 \
# --seed 97 \

################### 2.28 ######################
# python CIA_tta.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_HED_randomTrue_n0.yaml' --nj-stdhyper -0.9 --nj-distribution normal --nj-p 1 --nj-TTA 1 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-tta1-O-hed-std-0.9-p1-thresh0.8 \
# --seed 97 \

# python CIA_tta.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_HSV_randomTrue_n0.yaml' --nj-stdhyper -0.9 --nj-distribution normal --nj-p 1 --nj-TTA 1 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-tta1-O-hsv-std-0.9-p1-thresh0.8 \
# --seed 97 \

############ 3.1 ################
# python CIA_tta.py \
# --dataset $dataset_path \
# --nj-config './norm_jitter/MoNuSeg_LAB+HED+HSV_randomTrue_n0.yaml' --nj-stdhyper -0.9 --nj-distribution normal --nj-p 1 --nj-TTA 1 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output nj-tta1-O-lab+hed+hsv-std-0.9-p1-thresh0.8 \
# --seed 97 \

# hedlight-p0.5-tta
# python CIA_tta.py \
# --dataset $dataset_path \
# --hed-jitter 0.05 \
# --epochs $ep --batch-size $b --cj-p 1 --nj-TTA 1 \
# --lr $lr \
# --output tta1-O-hedlight0.05-p1
# --seed 97 \

# # hsv strong jitter p=0.5-tta1
# python CIA_tta.py \
# --dataset $dataset_path \
# --color-jitter 0.35 0.5 0.5 0.5 --cj-p 1 --nj-TTA 1 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output tta1-O-hsvstrong0.5-p1
# --seed 97 \

# hsv light jitter p=1-tta1
python CIA_tta.py \
--dataset $dataset_path \
--color-jitter 0.35 0.5 0.1 0.1 --cj-p 1 --nj-TTA 1 \
--epochs $ep --batch-size $b \
--lr $lr \
--output tta1-O-hsvlight0.1-p1
--seed 97 \

# labhsvlight-p1
# python CIA_tta.py \
# --dataset $dataset_path \
# --lab-jitter 0.1 0.1 0.1 \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output tta1-O-labhsvlight0.1-p1
# --seed 97 \