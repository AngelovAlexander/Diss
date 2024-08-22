import argparse
import os
import numpy as np
import shap
import torch
import torchvision
import torch.nn as nn
from output_model_SimGCD import OutputSimGCD
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from config import osr_split_dir
from model import DINOHead
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def divide_between_classes(dataset):
    # Groups all images of the same type in a single Tensor
    classes = {}
    for data in dataset:
        if data[1] not in classes:
            if type(data[0]) in [tuple, list]:
                classes[data[1]] = torch.stack(data[0])
            else:
                classes[data[1]] = data[0].unsqueeze(0)
        else:
            if type(data[0]) in [tuple, list]:
                classes[data[1]] = torch.cat([classes.get(data[1]), torch.stack(data[0])])
            else:
                classes[data[1]] = torch.cat([classes.get(data[1]), data[0].unsqueeze(0)])
    return classes


def shap_benchmark(model, train_dataset, test_dataset, plot_name):
    model = model.to("cuda")
    output_model = OutputSimGCD(model)
    c1 = train_dataset[list(train_dataset.keys())[4]]
    c = c.to("cuda")
    c = c.requires_grad_(True)
    output_model.to("cuda")
    e = shap.DeepExplainer(output_model, c)
    d = train_dataset[list(train_dataset.keys())[7]][0].unsqueeze(0)
    d = d.to("cuda")
    d = d.requires_grad_(True)
    y = d.detach().clone()
    y = y.detach().cpu().numpy()
    y = y.transpose(0,2,3,1)
    shap_values = e.shap_values(c, check_additivity=False)
    category_shap_values = np.mean(shap_values[0], axis=0)

    num_images_to_show = 5
    images_to_show = c[:num_images_to_show]
    images_to_show = images_to_show.detach().cpu().numpy()
    images_to_show = images_to_show.transpose(0,2,3,1)
    shap_values_to_show = category_shap_values

    # Use shap.image_plot to visualize the SHAP values on the images
    shap.image_plot(shap_values_to_show, images_to_show[0])
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
    herb_path_splits = os.path.join(osr_split_dir, 'herbarium_19_class_splits.pkl')
    #cub_path_splits = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')

    with open(herb_path_splits, 'rb') as handle:
        class_splits = pickle.load(handle)

    train_classes = class_splits['Old']
    unlabeled_classes = class_splits['New']
    #train_classes = class_splits['known_classes']
    #open_set_classes = class_splits['unknown_classes']
    #unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']
    
    mlp_out_dim = len(train_classes) + len(unlabeled_classes)
    

    with open(args.train_dataset_path, 'rb') as tr_dataset:
        train_dataset = pickle.load(tr_dataset)
    
    with open(args.test_dataset_path, 'rb') as test_dataset:
        test_dataset = pickle.load(test_dataset)

    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    projector = DINOHead(in_dim=feat_dim, out_dim=mlp_out_dim, nlayers=num_mlp_layers)
    model = nn.Sequential(backbone, projector).to(device)

    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict["model"])
    model.eval()

    train_classes = divide_between_classes(train_dataset)
    shap_benchmark(model, train_classes, test_dataset, args.shap_image_plot_name)
