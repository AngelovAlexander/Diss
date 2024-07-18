import argparse
import os
import numpy as np
import torch
import cv2
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from config import osr_split_dir
from model import DINOHeadNew
from shap_benchmark import divide_between_classes
from torchvision.transforms.functional import to_pil_image
from craft_interpretability import show

def visualize_attention(image, attn_weights, upscale_factor=16):
    image = np.transpose(image.detach().cpu().numpy(), (1, 2, 0))

    average_attention = attn_weights.mean(dim=1) # Result is 197x197
    average_attention = average_attention[0].mean(dim=0)
    average_attention = average_attention[1:]
    attention_map = average_attention.reshape(14, 14)  # Result is 14x14
    attention_map = attention_map.detach().cpu().numpy()
    # Create a mask for values above 0.01
    mask = attention_map > 0.01
        
    # Apply the mask to attention weights
    attention_map[~mask] = 0

    attention_map_resized = cv2.resize(attention_map, (244, 244), interpolation=cv2.INTER_LINEAR)

    plt.figure(figsize=(8, 8))
    show(image)
    plt.imshow(attention_map_resized, cmap='jet', alpha=0.5)  # 'jet' colormap, 50% transparency
    plt.colorbar()
    plt.title('Attention Map Overlay')
    plt.axis('off')
    plt.show()
    plt.savefig("results/attention_last.png")
    plt.clf()

def visualize_attention_patch(image, attention_maps, layer_idx, upscale_factor=16):
    # Extract the attention map for the specified layer
    attn_map = attention_maps[0]  # Take the first element of the batch

    # Assuming attention map shape is [num_heads, num_patches, num_patches]
    attn_map = attn_map.mean(dim=1)  # Average over heads

    # Remove the class token attention
    attn_map = attn_map[0].mean(dim=0)
    attn_map = attn_map[1:]
    print(attn_map.shape)
    
    # Reshape and normalize attention map
    attn_map = attn_map.reshape(14, 14)  # Assuming 14x14 patches for ViT
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    
    # Resize attention map to match the image size
    attn_map = np.array(to_pil_image(attn_map.unsqueeze(0)))
    attn_map = Image.fromarray(attn_map).resize(
        (244, 244),
        resample=Image.BILINEAR
    )
    attn_map = np.array(attn_map)
    
    # Plot image and overlay attention heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(image.detach().squeeze().permute(1, 2, 0).cpu().numpy())  # Assuming image is a tensor
    plt.imshow(attn_map, cmap='Reds', alpha=0.6)
    plt.colorbar()
    plt.title(f"Layer {layer_idx} Attention Visualization")
    plt.axis('off')
    plt.show()
    plt.savefig("results/attention_special2_" + str(layer_idx) + ".png")
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

    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    projector = DINOHeadNew(in_dim=feat_dim, out_dim=mlp_out_dim, nlayers=num_mlp_layers)
    model = nn.Sequential(backbone, projector).to(device)

    state_dict = torch.load(args.model_path)
    
    model.load_state_dict(state_dict["model"])
    model.eval()

    #train_classes = divide_between_classes(train_dataset)
    #dataset_labels = list(train_classes.keys())[:10]
    #image = train_classes[dataset_labels[9]]#torch.stack(train_dataset[0][0])
    image = torch.stack(train_dataset[0][0])
    image = image.to("cuda")
    image = image.requires_grad_(True)

    attention_outputs = []

    # Run inference
    with torch.no_grad():
        _ = model(image)
        attn_weights = model[0].get_last_selfattention(image[0].unsqueeze(0))
        attention_outputs.append(attn_weights)
        print(attn_weights.shape)

    visualize_attention_patch(image[0], attention_outputs, layer_idx=56, upscale_factor=4)
    visualize_attention(image[0], attn_weights)