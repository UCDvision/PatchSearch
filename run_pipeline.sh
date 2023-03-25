#!/usr/bin/env bash

set -x
set -e

OUTPUT_DIR='== ROOT DIRECTORY TO SAVE EXPERIMENT RESULTS GOES HERE =='
CODE_DIR='== ROOT DIRECTORY THAT CONTAINS TRAINING DATA FILE LISTS GOES HERE =='
EXPERIMENT_ID='HTBA_trigger_10_targeted_n02106550'
EXP_DIR=$OUTPUT_DIR/$EXPERIMENT_ID/moco
EVAL_DIR=$EXP_DIR/linear
DEFENSE_DIR=$EXP_DIR/patch_search_iterative_search_test_images_size_1000_window_w_60_repeat_patch_1_prune_clusters_True_num_clusters_1000_per_iteration_samples_2_remove_0x25
FILTERED_DIR=$DEFENSE_DIR/patch_search_poison_classifier_topk_20_ensemble_5_max_iterations_2000_seed_4789
RATE='0.50'
SEED=4789

### STEP 1.1: pretrain the model
python main_moco_files_dataset_strong_aug.py \
    --seed $SEED \
    -a vit_base --epochs 200 -b 1024 \
    --stop-grad-conv1 --moco-m-cos \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --dist-url "tcp://localhost:$(( $RANDOM % 50 + 10000 ))" \
    --save_folder $EXP_DIR \
    $CODE_DIR/poison-generation/data/$EXPERIMENT_ID/train/loc_random_loc*_rate_${RATE}_targeted_True_*.txt

for seed in {4787..4791}
do
    ### STEP 1.2: train a linear layer for evaluating the pretrained model
    CUDA_VISIBLE_DEVICES=0 python main_lincls.py \
        --seed $seed \
        -a vit_base --lr 0.1 \
        --pretrained $EXP_DIR/checkpoint_0199.pth.tar \
        --train_file $CODE_DIR/poison-generation/$seed/train_ssl_0.01_filelist.txt \
        --val_file $CODE_DIR/poison-generation/val_ssl_filelist.txt \
        --save_folder $EVAL_DIR

    ### STEP 1.3: evaluate the trained linear layer with clean and poisoned val data
    CUDA_VISIBLE_DEVICES=0 python main_lincls.py \
        --seed $seed \
        -a vit_base --lr 0.1 \
        --conf_matrix \
        --resume $EVAL_DIR/checkpoint.pth.tar \
        --train_file $CODE_DIR/poison-generation/$seed/train_ssl_0.01_filelist.txt \
        --val_file $CODE_DIR/poison-generation/val_ssl_filelist.txt \
        --val_poisoned_file $CODE_DIR/poison-generation/data/$EXPERIMENT_ID/val_poisoned/loc_random_*.txt \
        --eval_id data_seed_${seed}_run_seed_${seed}
done

python parse_logs.py $EVAL_DIR | tee -a $EVAL_DIR/logs.eval_summary_seed

### STEP 2: run iterative search
for i in {1..2}
do
    ### STEP 2.1: calculate and cache the features if not done so and exit else run defense
    ### STEP 2.2: run defense if previous step cached the features else just run defense one more time
    ### need to break this into two steps since the combining them slows the defense
    python patch_search_iterative_search.py \
        --arch moco_vit_base \
        --weights $EXP_DIR/checkpoint_0199.pth.tar \
        --linear_weights $EVAL_DIR/checkpoint.pth.tar \
        --train_file $CODE_DIR/poison-generation/data/$EXPERIMENT_ID/train/loc_random_loc*_rate_${RATE}_targeted_True_*.txt \
        --val_file $CODE_DIR/poison-generation/val_ssl_filelist.txt \
        --prune_clusters \
        --use_cached_feats \
        --use_cached_poison_scores
done

### STEP 3: run poison classifier
CUDA_VISIBLE_DEVICES=0 python patch_search_poison_classifier.py \
    --print_freq 20 \
    --model_count 5 \
    --batch_size 32 \
    --eval_freq 20 \
    --max_iterations 2000 \
    --workers 8 \
    --seed ${SEED} \
    --train_file $CODE_DIR/poison-generation/data/$EXPERIMENT_ID/train/loc_random_loc*_rate_${RATE}_targeted_True_*.txt \
    --poison_dir $DEFENSE_DIR/all_top_poison_patches \
    --poison_scores $DEFENSE_DIR/poison-scores.npy \
    --eval_data "seed_${SEED}" \
    --topk_poisons 20

EXP_DIR=$FILTERED_DIR/moco
EVAL_DIR=$EXP_DIR/linear

### STEP 4.1: pretrain the model on training set filtered with PatchSearch
python main_moco_files_dataset_strong_aug.py \
    --seed $SEED \
    -a vit_base --epochs 200 -b 1024 \
    --icutmix --alpha 1.0 \
    --stop-grad-conv1 --moco-m-cos \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --dist-url "tcp://localhost:$(( $RANDOM % 50 + 10000 ))" \
    --save_folder $EXP_DIR \
    $FILTERED_DIR/filtered.txt

for seed in {4787..4791}
do
    ### STEP 4.2: train a linear layer for evaluating the pretrained model
    CUDA_VISIBLE_DEVICES=0 python main_lincls.py \
        --seed $seed \
        -a vit_base --lr 0.1 \
        --pretrained $EXP_DIR/checkpoint_0199.pth.tar \
        --train_file $CODE_DIR/poison-generation/$seed/train_ssl_0.01_filelist.txt \
        --val_file $CODE_DIR/poison-generation/val_ssl_filelist.txt \
        --save_folder $EVAL_DIR

    ### STEP 4.3: evaluate the trained linear layer with clean and poisoned val data
    CUDA_VISIBLE_DEVICES=0 python main_lincls.py \
        --seed $seed \
        -a vit_base --lr 0.1 \
        --conf_matrix \
        --resume $EVAL_DIR/checkpoint.pth.tar \
        --train_file $CODE_DIR/poison-generation/$seed/train_ssl_0.01_filelist.txt \
        --val_file $CODE_DIR/poison-generation/val_ssl_filelist.txt \
        --val_poisoned_file $CODE_DIR/poison-generation/data/$EXPERIMENT_ID/val_poisoned/loc_random_*.txt \
        --eval_id data_seed_${seed}_run_seed_${seed}
done

python parse_logs.py $EVAL_DIR | tee -a $EVAL_DIR/logs.eval_summary_seed

