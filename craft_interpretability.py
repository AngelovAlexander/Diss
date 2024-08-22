from craft.craft_torch import Craft

import argparse
import os
import numpy as np
import shap
import cv2
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


def show(img, ax = None, **kwargs):
    img = np.array(img)
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    img = img.astype(np.float64)
    img -= img.min();img /= img.max()
    if ax:
      ax.imshow(img, **kwargs); ax.axis('off')
    else:
      plt.imshow(img, **kwargs); plt.axis('off')

from matplotlib.colors import ListedColormap
import matplotlib
import colorsys

def get_alpha_cmap(cmap):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    else:
        c = np.array((cmap[0]/255.0, cmap[1]/255.0, cmap[2]/255.0))

        cmax = colorsys.rgb_to_hls(*c)
        cmax = np.array(cmax)
        cmax[-1] = 1.0

        cmax = np.clip(np.array(colorsys.hls_to_rgb(*cmax)), 0, 1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [c,cmax])

    alpha_cmap = cmap(np.arange(256))
    alpha_cmap[:,-1] = np.linspace(0, 0.85, 256)
    alpha_cmap = ListedColormap(alpha_cmap)

    return alpha_cmap

def plot_legend(cmaps, most_important_concepts, crops, crops_u):
  for i, c_id in enumerate(most_important_concepts):
    cmap = cmaps[i]
    plt.subplot(1, len(most_important_concepts), i+1)

    best_crops_id = np.argsort(crops_u[:, c_id])[::-1][0]
    best_crop = crops[best_crops_id]

    show(best_crop)

    plt.tight_layout()
    plt.show()


def concept_attribution_maps(images_preprocessed, images_u, cmaps, most_important_concepts, id, percentile=30):
    img = images_preprocessed[id]
    u = images_u[id]
    img = img.detach().cpu().numpy()

    show(img)

    for i, c_id in enumerate(most_important_concepts):

        cmap = cmaps[i]
        heatmap = u[c_id]

        # only show concept if excess N-th percentile
        sigma = np.percentile(images_u[:,c_id].flatten(), percentile)
        heatmap = heatmap * np.array(heatmap > sigma, np.float32)

        heatmap_reshaped = np.full((img.shape[1], img.shape[2]), heatmap)
        show(heatmap_reshaped, cmap=cmap, alpha=0.7)

    plt.show()

def concept_attribution_maps2(images_preprocessed, images_u, cmaps, most_important_concepts, id, percentile=90):
    img = images_preprocessed[id]
    u = images_u[id]  # u is 1-dimensional for each image

    # Image dimensions
    _, height, width = img.shape

    plt.figure(figsize=(12, 6))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(img.detach().cpu().numpy(), (1, 2, 0)))
    plt.title("Original Image")
    plt.axis('off')

    # Concept Attribution Map
    plt.subplot(1, 2, 2)
    plt.imshow(np.transpose(img.detach().cpu().numpy(), (1, 2, 0)))
    plt.title("Concept Attribution Map")

    combined_heatmap = np.zeros((height, width))

    for i, c_id in enumerate(most_important_concepts):
        concept_importance = u[c_id]

        # Showing concept if excess N-th percentile
        sigma = np.percentile(images_u[:, c_id], percentile)
        if concept_importance > sigma:
            # Creating a heatmap
            heatmap = np.full((height, width), concept_importance)
            combined_heatmap += heatmap

    # Normalize the combined heatmap
    if combined_heatmap.max() > 0:
        combined_heatmap = combined_heatmap / combined_heatmap.max()

    # Applying colormap
    heatmap_colored = plt.cm.jet(combined_heatmap)
    heatmap_colored[..., 3] = combined_heatmap * 0.5

    plt.imshow(heatmap_colored)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def craft_interpretability(model, train_dataset, test_dataset, dataset_name, plot_name, in_dim, out_dim, norm_last_layer=True, patch_size = 64):
    model = model.to("cuda")
    class_id = None
    proj = OutputProjSimGCD(model)
    last_layer = nn.utils.weight_norm(nn.Linear(256, out_dim, bias=False))
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

    dataset_labels = list(train_dataset.keys())#[:1]
    if not os.path.isdir("results/train_" + dataset_name + "/Craft"):
        create_folder("results/train_" + dataset_name + "/Craft")

    if not os.path.isdir("results/train_" + dataset_name + "/Craft/" + str(patch_size)):
        create_folder("results/train_" + dataset_name + "/Craft/" + str(patch_size))
    
    cmaps = [
        get_alpha_cmap((54, 197, 240)),
        get_alpha_cmap((210, 40, 95)),
        get_alpha_cmap((236, 178, 46)),
        get_alpha_cmap((15, 157, 88)),
        get_alpha_cmap((84, 25, 85))
    ]

    for i in range(len(dataset_labels)):
        class_dataset = train_dataset[dataset_labels[i]]
        class_dataset = class_dataset.to("cuda")
        class_dataset = class_dataset.requires_grad_(True)

        crops, crops_u, w = craft.fit(class_dataset)
        crops = np.moveaxis(crops.detach().cpu().numpy(), 1, -1)

        importances = craft.estimate_importance(class_dataset, class_id=dataset_labels[i])
        images_u = craft.transform(class_dataset)
        plt.figure(figsize=(10, 10))
        plt.bar(range(len(importances)), importances)
        plt.xticks(range(len(importances)))
        plt.title("Concept Importance")
        plt.tight_layout()
        plt.savefig("results/train_" + dataset_name + "/Craft/" + str(patch_size) + "/" + plot_name + "_concept_importance_class_" + str(dataset_labels[i]) + ".png")
        plt.clf()

        most_important_concepts = np.argsort(importances)[::-1][:5]
    
        nb_crops = 10
        for c_id in most_important_concepts:

            best_crops_ids = np.argsort(crops_u[:, c_id])[::-1][:nb_crops]
            best_crops = crops[best_crops_ids]
            for j in range(nb_crops):
                plt.subplot(ceil(nb_crops/5), 5, j+1)
                show(best_crops[j])
            plt.tight_layout()
            plt.savefig("results/train_" + dataset_name + "/Craft/" + str(patch_size) + "/" + plot_name + "_concept_" + str(c_id) + "_class_" + str(dataset_labels[i]) + ".png")
            plt.clf()
    
    
        plot_legend(cmaps, most_important_concepts, crops, crops_u)
        plt.savefig("results/train_" + dataset_name + "/Craft/" + str(patch_size) + "/" + plot_name + "_legend_" + str(dataset_labels[i]) + ".png")
        plt.clf()
        #concept_attribution_maps2(class_dataset, images_u, cmaps, most_important_concepts, 0)
        #plt.show()
        #plt.savefig("results/train_" + dataset_name + "/Craft/" + str(patch_size) + "/" + plot_name + "_map0_" + str(dataset_labels[i]) + ".png")
        #plt.clf()
        #concept_attribution_maps2(class_dataset, images_u, cmaps, most_important_concepts, 1)
        #plt.show()
        #plt.savefig("results/train_" + dataset_name + "/Craft/" + str(patch_size) + "/" + plot_name + "_map1_" + str(dataset_labels[i]) + ".png")
        #plt.clf()
        #concept_attribution_maps2(class_dataset, images_u, cmaps, most_important_concepts, 2)
        #plt.savefig("results/train_" + dataset_name + "/Craft/" + str(patch_size) + "/" + plot_name + "_map2_" + str(dataset_labels[i]) + ".png")
        #plt.clf()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='craft_interpretability', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    #test_classes = divide_between_classes(test_dataset)
    craft_interpretability(model, train_classes, test_dataset, args.dataset_name, args.shap_image_plot_name, feat_dim, mlp_out_dim, args.patch_size)
