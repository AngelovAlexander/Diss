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

def divide_between_classes(dataset):
    # Groups all images of the same type in a single Tensor
    classes = {}
    for data in dataset:
        if data[1] not in classes:
            classes[data[1]] = torch.stack(data[0])
        else:
            classes[data[1]] = torch.cat([classes.get(data[1]), torch.stack(data[0])])
    return classes

def show(img, **kwargs):
    img = np.array(img)
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    img -= img.min();img /= img.max()
    plt.imshow(img, **kwargs); plt.axis('off')

def craft_interpretability(model, train_dataset, test_dataset, plot_name, in_dim, out_dim, norm_last_layer=True):
    model = model.to("cuda")
    print(model)
    output_model = OutputSimGCD(model)
    proj = OutputProjSimGCD(model)
    c1 = train_dataset[list(train_dataset.keys())[4]]
    c2 = train_dataset[list(train_dataset.keys())[7]]
    c3 = train_dataset[list(train_dataset.keys())[18]]
    c4 = train_dataset[list(train_dataset.keys())[52]]
    c5 = train_dataset[list(train_dataset.keys())[1]]
    c6 = train_dataset[list(train_dataset.keys())[63]]
    c7 = train_dataset[list(train_dataset.keys())[23]]
    c = torch.cat([c1, c2, c3, c4, c5, c6, c7])
    c = c.to("cuda")
    c = c.requires_grad_(True)
    output_model.to("cuda")
    hidden_dims = [128, 64]

    latent_to_logit_model = LatentToLogitMLP(256, hidden_dims, out_dim)
    latent_to_logit_model.to("cuda")
    #h.to("cuda")
    #last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False)) ### Change in_dim as 256
    #last_layer.weight_g.data.fill_(1)
    #if norm_last_layer:
    #    last_layer.weight_g.requires_grad = False
    #logits = last_layer
    #logits.to("cuda")
    craft = Craft(input_to_latent=proj,
              latent_to_logit=latent_to_logit_model,
              number_of_concepts=10,
              patch_size=56,
              batch_size=56)
    crops, crops_u, w = craft.fit(c)

    
    crops = np.moveaxis(crops.detach().cpu().numpy(), 1, -1)

    print(crops.shape, crops_u.shape, w.shape)

    print(list(train_dataset.keys())[4])
    print(c.shape)
    c1 = c.detach().clone()
    importances = craft.estimate_importance(c, class_id=list(train_dataset.keys())[4])
    images_u = craft.transform(c)

    print(images_u.shape)

    #plt.bar(range(len(importances)), importances)
    #plt.xticks(range(len(importances)))
    #plt.title("Concept Importance")

    most_important_concepts = np.argsort(importances)[::-1][:5]

    for c_id in most_important_concepts:
        print("Concept", c_id, " has an importance value of ", importances[c_id])
    
    nb_crops = 10
    for c_id in most_important_concepts:

        best_crops_ids = np.argsort(crops_u[:, c_id])[::-1][:nb_crops]
        best_crops = crops[best_crops_ids]
        print("Len")
        print(len(best_crops))
        print("Concept", c_id, " has an importance value of ", importances[c_id])
        for i in range(nb_crops):
            plt.subplot(ceil(nb_crops/5), 5, i+1)
            show(best_crops[i])
        plt.show()
        print('\n\n')
    plt.savefig("results/" + plot_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='shap_bnchmark', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--train_dataset_path', type=str)
    parser.add_argument('--test_dataset_path', type=str)
    parser.add_argument('--save_shap_values_path', type=str)
    parser.add_argument('--shap_image_plot_name', type=str)
    args = parser.parse_args()

    device = torch.device('cuda:0')
    feat_dim = 768
    num_mlp_layers = 3
    #herb_path_splits = os.path.join(osr_split_dir, 'herbarium_19_class_splits.pkl')
    cub_path_splits = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')

    with open(cub_path_splits, 'rb') as handle:
        class_splits = pickle.load(handle)

    #train_classes = class_splits['Old']
    #unlabeled_classes = class_splits['New']
    train_classes = class_splits['known_classes']
    open_set_classes = class_splits['unknown_classes']
    unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']
    
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
    craft_interpretability(model, train_classes, test_dataset, args.shap_image_plot_name, feat_dim, mlp_out_dim)
