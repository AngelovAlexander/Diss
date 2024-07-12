#!/bin/bash
set -e
set -x

CUDA_VISIBLE_DEVICES=0 python -u craft_interpretability.py \
    --model_path 'models/cub/model.pt' \
    --train_dataset_path 'Data/Cub_new_model/train_dataset.pt' \
    --test_dataset_path 'Data/Cub_new_model/unlabelled_train_examples_test.pt' \
    --save_shap_values_path './' \
    --shap_image_plot_name "patch_size" \
    --dataset_name "cub" \
    --patch_size 64 \
