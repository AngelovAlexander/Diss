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

def shap_benchmark(model, train_dataset, test_dataset, plot_name):

    def model_wrapper(images):
        _, logits = model(images)
        return logits

    model = model.to("cuda")
    output_model = OutputSimGCD(model)
    
    a = torch.stack(train_dataset[0][0])
    a = a.to("cuda")
    a = a.requires_grad_(True)
    b = a.detach().clone()
    output_model.to("cuda")
    e = shap.DeepExplainer(output_model, a)
    t_d = b#test_dataset[1][0].unsqueeze(0)
    t_d = t_d.to("cuda")
    t_d = t_d.requires_grad_(True)
    y = t_d.detach().clone()
    y = y.detach().cpu().numpy()
    y = y.transpose(0,2,3,1)
    shap_values = e.shap_values(t_d, check_additivity=False)
    shap_numpy = list(np.transpose(shap_values, (4, 0, 2, 3, 1)))

    shap_values = [shap_values[i, 0] for i in range(shap_values.shape[0])]
    shap.image_plot(shap_values, y[0])
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

    with open(herb_path_splits, 'rb') as handle:
        class_splits = pickle.load(handle)

    train_classes = class_splits['Old']
    unlabeled_classes = class_splits['New']
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

    shap_benchmark(model, train_dataset, test_dataset, args.shap_image_plot_name)
