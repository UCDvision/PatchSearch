# PatchSearch
Code for the CVPR '23 paper, "Defending Against Patch-based Data Poisoning Backdoor Attacks on Self-Supervised Learning"

## Installation

1. Clone the repository with `git clone https://github.com/UCDvision/PatchSearch`
2. Navigate into the clone repo with `cd PatchSearch`
3. Install the required packages with `conda env create -f environment.yml`
4. Activate the environment with `conda activate patch_search`

Note that we use a custom version of [pytorch-grad-cam](https://github.com/UCDvision/pytorch-grad-cam)
since the original does not return the model output which is required by our code.

## Data

Use the code from the [SS-Backdoor](https://github.com/UMBCvision/SSL-Backdoor#poison-generation) repository to generate clean and poisoned data.

## Pre-training

Now, you're ready to pre-train your models on poisoned data.
First, set a few variables pointing to training data files, training images, and experiment output directory.
Then, run the `main_moco_files_dataset_strong_aug.py` file as shown in the below command.:

```
OUTPUT_DIR='== ROOT DIRECTORY TO SAVE EXPERIMENT RESULTS GOES HERE =='
CODE_DIR='== ROOT DIRECTORY THAT CONTAINS TRAINING DATA FILE LISTS GOES HERE =='
EXPERIMENT_ID='HTBA_trigger_10_targeted_n02106550'
EXP_DIR=$OUTPUT_DIR/$EXPERIMENT_ID/moco
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

```

Adjust the batch size and learning rate based on your available GPU.
For training with `i-cutmix`, simply add the command line parameters `--icutmix --alpha 1.0` to the above pre-training command.

## Linear Evaluation

The pre-trained checkpoints are evaluated by running the below command. Note that it uses some of the variables set above.

```
### STEP 1.2: train a linear layer for evaluating the pretrained model
CUDA_VISIBLE_DEVICES=0 python main_lincls.py \
    --seed $SEED \
    -a vit_base --lr 0.1 \
    --pretrained $EXP_DIR/checkpoint_0199.pth.tar \
    --train_file linear_eval_files/train_ssl_0.01_filelist.txt \
    --val_file linear_eval_files/val_ssl_filelist.txt \
    --save_folder $EVAL_DIR

### STEP 1.3: evaluate the trained linear layer with clean and poisoned val data
CUDA_VISIBLE_DEVICES=0 python main_lincls.py \
    --seed $SEED \
    -a vit_base --lr 0.1 \
    --conf_matrix \
    --resume $EVAL_DIR/checkpoint.pth.tar \
    --train_file linear_eval_files/train_ssl_0.01_filelist.txt \
    --val_file linear_eval_files/val_ssl_filelist.txt \
    --val_poisoned_file $CODE_DIR/$EXPERIMENT_ID/val_poisoned/loc_random_*.txt \
    --eval_id exp_${SEED}
```

## Running Iterative Search in PatchSearch

Now, we can run the iterative search part of PartSearch to find highly poisonous samples.
Use the below command with some of the variables set during pre-trainig.
The poisons will be stored in the directory `all_top_poison_patches` located inside the experiment output directory of below command.

```
EVAL_DIR=$EXP_DIR/linear

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
        --train_file $CODE_DIR/$EXPERIMENT_ID/train/loc_random_loc*_rate_${RATE}_targeted_True_*.txt \
        --val_file linear_eval_files/val_ssl_filelist.txt \
        --prune_clusters \
        --use_cached_feats \
        --use_cached_poison_scores
done
```

## Running Poison Classifier in PatchSearch

Once we have the list of highly poisonous patches, we can build a classifier to detect them with the below command.
The filtered training data is listed in the `filtered.txt`.

```
DEFENSE_DIR=$EXP_DIR/patch_search_iterative_search_test_images_size_1000_window_w_60_repeat_patch_1_prune_clusters_True_num_clusters_1000_per_iteration_samples_2_remove_0x25

### STEP 3: run poison classifier
CUDA_VISIBLE_DEVICES=0 python patch_search_poison_classifier.py \
    --print_freq 20 \
    --model_count 5 \
    --batch_size 32 \
    --eval_freq 20 \
    --max_iterations 2000 \
    --workers 8 \
    --seed ${SEED} \
    --train_file $CODE_DIR/$EXPERIMENT_ID/train/loc_random_loc*_rate_${RATE}_targeted_True_*.txt \
    --poison_dir $DEFENSE_DIR/all_top_poison_patches \
    --poison_scores $DEFENSE_DIR/poison-scores.npy \
    --eval_data "seed_${SEED}" \
    --topk_poisons 20
```

## Post-Defense Pre-training and Evaluation

Finally, run the below command to run pre-training on cleaned up data and evaluate the resulting model.

```
FILTERED_DIR=$DEFENSE_DIR/patch_search_poison_classifier_topk_20_ensemble_5_max_iterations_2000_seed_4789
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

### STEP 4.2: train a linear layer for evaluating the pretrained model
CUDA_VISIBLE_DEVICES=0 python main_lincls.py \
    --seed $seed \
    -a vit_base --lr 0.1 \
    --pretrained $EXP_DIR/checkpoint_0199.pth.tar \
    --train_file linear_eval_files/train_ssl_0.01_filelist.txt \
    --val_file linear_eval_files/val_ssl_filelist.txt \
    --save_folder $EVAL_DIR
                                                                                                                                                                                                            
### STEP 4.3: evaluate the trained linear layer with clean and poisoned val data
CUDA_VISIBLE_DEVICES=0 python main_lincls.py \
    --seed $seed \
    -a vit_base --lr 0.1 \
    --conf_matrix \
    --resume $EVAL_DIR/checkpoint.pth.tar \
    --train_file linear_eval_files/train_ssl_0.01_filelist.txt \
    --val_file linear_eval_files/val_ssl_filelist.txt \
    --val_poisoned_file $CODE_DIR/$EXPERIMENT_ID/val_poisoned/loc_random_*.txt \
    --eval_id exp_${seed}
```
