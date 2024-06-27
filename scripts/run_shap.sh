#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=0 python -u shap_benchmark.py \
    --model_path './' \
    --train_dataset_path './' \
    --test_dataset_path './' \
    --save_shap_values_path './' \
    --shap_image_plot_name "shap_image_plot.png" \
    --exp_name herb19_simgcd
