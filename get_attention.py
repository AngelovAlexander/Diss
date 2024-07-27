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
import pickle
from evaluate_attention import *
from itertools import combinations
from data_split import create_folder
from model import PatchClassifier
import torch.nn.functional as F

def prepare_attention_map(attn_weights, image_shape):
    average_attention = attn_weights.mean(dim=1) # Result is 197x197
    average_attention = average_attention[0].mean(dim=0)
    average_attention = average_attention[1:]
    attention_map = average_attention.reshape(14, 14)  # Result is 14x14
    attention_map = attention_map.detach().cpu().numpy()

    attention_map_resized = cv2.resize(attention_map, (image_shape[1], image_shape[2]), interpolation=cv2.INTER_LINEAR)
    attention_map_resized = (attention_map_resized - attention_map_resized.min()) / (attention_map_resized.max() - attention_map_resized.min())
    return attention_map_resized

def visualize_individual_attention(image, attn_weights, id, class_idx):
    image = np.transpose(image, (1, 2, 0))

    #with open('results/craft_train_herb/try/attn_map_before_norm.pt', 'wb') as attn_map:
    #    pickle.dump(attention_map_resized, attn_map, pickle.HIGHEST_PROTOCOL)

    # Normalize the attention map
    
    #with open('results/craft_train_herb/try/img.pt', 'wb') as img_file:
    #    pickle.dump(image, img_file, pickle.HIGHEST_PROTOCOL)
    
    #with open('results/craft_train_herb/try/attn_map.pt', 'wb') as attn_map:
    #    pickle.dump(attention_map_resized, attn_map, pickle.HIGHEST_PROTOCOL)

    # Apply threshold
    #attention_map_resized[attention_map_resized > (attention_map_resized.min() + (attention_map_resized.min() / attention_map_resized.max()))] = 0

    # Normalize the attention map
    #attention_map_resized = (attention_map_resized - attention_map_resized.min()) / (attention_map_resized.max() - attention_map_resized.min())


    plt.figure(figsize=(8, 8))
    show(image)
    plt.imshow(attn_weights, cmap='jet', alpha=0.5)  # 'jet' colormap, 50% transparency
    plt.colorbar()
    plt.title('Attention Map Overlay')
    plt.axis('off')
    plt.show()
    if not os.path.isdir("results/train_herb/Self-Attention/Unlabeled/" + str(class_idx)):
        create_folder("results/train_herb/Self-Attention/Unlabeled/" + str(class_idx))
    plt.savefig("results/train_herb/Self-Attention/Unlabeled/" + str(class_idx) + "/attention_herb_last_" + str(id) + ".png")
    plt.clf()

def visualize_attention_patch(image, attention_maps, layer_idx):
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
    attn_map = Image.fromarray(attn_map).resize((244, 244),resample=Image.BILINEAR)
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

def embed_patches_with_attention(train_classes, visualize=False):
    dataset_labels = list(train_classes.keys())
    print("Label")
    print(dataset_labels[0])

    print("random_padding")
    
    images = train_classes[dataset_labels[0]]#dataset_labels[9]]#torch.stack(train_dataset[0][0])
    #images = torch.stack(train_dataset[0][0])
    images = images.to("cuda")
    images = images.requires_grad_(True)

    attention_outputs = []
    cur_id = 0
    #constraints = []
    all_patches = []
    embeddings = []

    idx = 0

    with torch.no_grad():
        _ = model(images)
        for img in images:
            attn_weights = model[0].get_last_selfattention(img.unsqueeze(0))
            attention_outputs.append(attn_weights)
    
            img = img.detach().cpu().numpy()
            attn_map = prepare_attention_map(attn_weights, img.shape)
            if visualize:
                visualize_individual_attention(img, attn_map, idx, dataset_labels[0])
            idx += 1
            
            patches = get_attended_patches(img, attn_map, threshold = 0.4, modification_type="static_padding", visualize = False)
            print(patches.shape)
            all_patches.append(patches)
            #print(np.vstack(patches).shape)
            #cur_img_embeddings = create_embeddings(patches)
            #constraints.extend(list(combinations(np.arange(cur_id, cur_id + cur_img_embeddings.shape[0]), 2)))
            #cur_id += cur_img_embeddings.shape[0]
            #print(cur_img_embeddings.shape)
            #embeddings.append(cur_img_embeddings)
    all_patches = np.vstack(all_patches)
    return all_patches
    """
    print(constraints)
    print(cur_id)
    embeddings = np.vstack(embeddings)
    print(embeddings.shape)

    labels = optimised_CoExDBSCAN(embeddings, eps=0.25, min_samples = 2, cannot_link_constraints = constraints)
    print(labels.shape)
    visualize_clusters(embeddings, labels)
    print(f"Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
    print(f"Number of noise points: {np.sum(labels == -1)}")
    ev = evaluate_optimised_CoExDBSCAN(embeddings, labels)
    print(ev)
    ev = evaluate_optimised_CoExDBSCAN(embeddings, labels, metric = "davies_bouldin_score")
    print(ev)"""

def classify_patches(model, patches):
    model.eval()
    with torch.no_grad():
        logits = model(patches)
        probabilities = F.softmax(logits, dim=1)
        _, predicted = torch.max(probabilities, 1)
    return predicted#[model.class_names[idx] for idx in predicted]

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
        labeled_classes = class_splits['Old']
        unlabeled_classes = class_splits['New']
    elif args.dataset_name == "cub":
        labeled_classes = class_splits['known_classes']
        open_set_classes = class_splits['unknown_classes']
        unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']
    else:
        raise NotImplementedError
    
    mlp_out_dim = len(labeled_classes) + len(unlabeled_classes)
    

    with open(args.train_dataset_path, 'rb') as tr_dataset:
        train_dataset = pickle.load(tr_dataset)

    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    projector = DINOHeadNew(in_dim=feat_dim, out_dim=mlp_out_dim, nlayers=num_mlp_layers)
    model = nn.Sequential(backbone, projector).to(device)

    state_dict = torch.load(args.model_path)
    
    model.load_state_dict(state_dict["model"])
    model.eval()
    train_classes = divide_between_classes(train_dataset)
    patches = embed_patches_with_attention(train_classes)
    print(patches.shape)
    patch_classifier = PatchClassifier(model)
    patches = torch.from_numpy(np.transpose(patches, (0, 3, 1, 2)))
    patches = patches.to("cuda")
    patches = patches.requires_grad_(True)
    results = classify_patches(patch_classifier, patches)
    print(results)
