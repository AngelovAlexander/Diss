#!/bin/bash
set -e
set -x

CUDA_VISIBLE_DEVICES=0 python -u craft_interpretability.py \
    --model_path 'models/cub/model_try.pt' \
    --train_dataset_path 'Data/Cub_new/train_dataset.pt' \
    --test_dataset_path 'Data/Cub_new/unlabelled_train_examples_test.pt' \
    --save_shap_values_path './' \
    --shap_image_plot_name "shap_image_plot2.png" \
