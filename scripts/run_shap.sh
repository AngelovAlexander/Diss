#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=0 python -u shap_benchmark.py \
    --model_path 'models/herbarium19/model.pt' \
    --train_dataset_path 'Data/Herbarium19/train_dataset.pt' \
    --test_dataset_path 'Data/Herbarium19/unlabelled_train_examples_test.pt' \
    --save_shap_values_path './' \
    --shap_image_plot_name "shap_image_plot_category_img_herb.png" \
