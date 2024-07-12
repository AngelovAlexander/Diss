from craft.craft_torch import Craft

import argparse
import os
import numpy as np
import shap
import torch
import torchvision
import torch.nn as nn
from output_model_SimGCD import OutputSimGCD, OutputProjSimGCD
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from config import osr_split_dir
from model import DINOHeadNew, LatentToLogitMLP
from math import ceil
from shap_benchmark import divide_between_classes
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from data_split import create_folder


def show(img, **kwargs):
    img = np.array(img)
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    img -= img.min();img /= img.max()
    plt.imshow(img, **kwargs); plt.axis('off')

def craft_interpretability(model, train_dataset, test_dataset, dataset_name, plot_name, in_dim, out_dim, norm_last_layer=True, patch_size = 64):
    model = model.to("cuda")
    class_id = None
    proj = OutputProjSimGCD(model)
    last_layer = nn.utils.weight_norm(nn.Linear(256, out_dim, bias=False)) ### Change in_dim as 256
    last_layer.weight_g.data.fill_(1)
    if norm_last_layer:
        last_layer.weight_g.requires_grad = False
    logits = last_layer
    logits.to("cuda")
    craft = Craft(input_to_latent=proj,
              latent_to_logit=logits,
              number_of_concepts=10,
              patch_size=patch_size,
              batch_size=patch_size)

    dataset_labels = list(train_dataset.keys())
    if not os.path.isdir("results/craft_train_" + dataset_name):
        create_folder("results/craft_train_" + dataset_name)

    if not os.path.isdir("results/craft_train_" + dataset_name + "/" + str(patch_size)):
        create_folder("results/craft_train_" + dataset_name + "/" + str(patch_size))

    for i in range(len(dataset_labels)):
        class_dataset = train_dataset[dataset_labels[i]]
        class_dataset = class_dataset.to("cuda")
        class_dataset = class_dataset.requires_grad_(True)

        crops, crops_u, w = craft.fit(class_dataset)
        crops = np.moveaxis(crops.detach().cpu().numpy(), 1, -1)

        importances = craft.estimate_importance(class_dataset, class_id=dataset_labels[i])
        images_u = craft.transform(class_dataset)
        plt.bar(range(len(importances)), importances)
        plt.xticks(range(len(importances)))
        plt.title("Concept Importance")
        plt.savefig("results/craft_train_" + dataset_name + "/" + str(patch_size) + "/" + plot_name + "_concept_importance_class_" + str(dataset_labels[i]) + ".png")
        plt.clf()

        most_important_concepts = np.argsort(importances)[::-1][:5]
    
        nb_crops = 10
        for c_id in most_important_concepts:

            best_crops_ids = np.argsort(crops_u[:, c_id])[::-1][:nb_crops]
            best_crops = crops[best_crops_ids]
            for j in range(nb_crops):
                plt.subplot(ceil(nb_crops/5), 5, j+1)
                show(best_crops[j])
            plt.savefig("results/craft_train_" + dataset_name + "/" + str(patch_size) + "/" + plot_name + "_concept_" + str(c_id) + "_class_" + str(dataset_labels[i]) + ".png")
            plt.clf()   

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='shap_bnchmark', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--train_dataset_path', type=str)
    parser.add_argument('--test_dataset_path', type=str)
    parser.add_argument('--save_shap_values_path', type=str)
    parser.add_argument('--shap_image_plot_name', type=str)
    parser.add_argument('--dataset_name', default="herb", type=str)
    parser.add_argument('--patch_size', default=64, type=int)
    args = parser.parse_args()

    device = torch.device('cuda:0')
    feat_dim = 768
    num_mlp_layers = 3
    if args.dataset_name == "herb":
        path_splits = os.path.join(osr_split_dir, 'herbarium_19_class_splits.pkl')
    elif args.dataset_name == "cub":
        path_splits = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
    else:
        raise NotImplementedError

    with open(path_splits, 'rb') as handle:
        class_splits = pickle.load(handle)

    if args.dataset_name == "herb":
        train_classes = class_splits['Old']
        unlabeled_classes = class_splits['New']
    elif args.dataset_name == "cub":
        train_classes = class_splits['known_classes']
        open_set_classes = class_splits['unknown_classes']
        unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']
    else:
        raise NotImplementedError
    
    mlp_out_dim = len(train_classes) + len(unlabeled_classes)
    

    with open(args.train_dataset_path, 'rb') as tr_dataset:
        train_dataset = pickle.load(tr_dataset)
    
    with open(args.test_dataset_path, 'rb') as test_dataset:
        test_dataset = pickle.load(test_dataset)

    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    projector = DINOHeadNew(in_dim=feat_dim, out_dim=mlp_out_dim, nlayers=num_mlp_layers)
    model = nn.Sequential(backbone, projector).to(device)

    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict["model"])
    model.eval()

    train_classes = divide_between_classes(train_dataset)
    test_classes = divide_between_classes(test_dataset)
    craft_interpretability(model, train_classes, test_classes, args.dataset_name, args.shap_image_plot_name, feat_dim, mlp_out_dim, args.patch_size)
