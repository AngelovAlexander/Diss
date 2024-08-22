#!/bin/bash
set -e
set -x

CUDA_VISIBLE_DEVICES=0 python -u craft_interpretability.py \
    --model_path 'models/herbarium19/model.pt' \
    --train_dataset_path 'Data/Herbarium19/train_dataset.pt' \
    --test_dataset_path 'Data/Herbarium19/unlabelled_train_examples_test.pt' \
    --save_shap_values_path './' \
    --shap_image_plot_name "patch_size" \
    --dataset_name "herb" \
    --patch_size 64 \
