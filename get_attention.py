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
from model import PatchClassifier, get_params_groups
from output_model_SimGCD import OutputProjSimGCD
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch.optim import SGD
from sklearn.ensemble import RandomForestClassifier
from torch.cuda.amp import autocast
from sklearn.decomposition import PCA

def prepare_attention_map(attn_weights, image_shape):
    average_attention = attn_weights.mean(dim=1) # Result is 197x197
    average_attention = average_attention[0].mean(dim=0)
    average_attention = average_attention[1:]
    attention_map = average_attention.reshape(14, 14)  # Result is 14x14
    attention_map = attention_map.detach().cpu().numpy()

    attention_map_resized = cv2.resize(attention_map, (image_shape[1], image_shape[2]), interpolation=cv2.INTER_LINEAR)
    attention_map_resized = (attention_map_resized - attention_map_resized.min()) / (attention_map_resized.max() - attention_map_resized.min())
    return attention_map_resized

def visualize_individual_attention(image, attn_weights, id, class_idx, dataset_name):
    if dataset_name not in ["cub", "herb"]:
        raise Exception("The dataset_name should be cub or herb!")
    image = np.transpose(image, (1, 2, 0))

    if not os.path.isdir("results/train_" + dataset_name + "/Self-Attention/Study/" + str(class_idx)):
        create_folder("results/train_" + dataset_name + "/Self-Attention/Study/" + str(class_idx))
    plt.figure(figsize=(8, 8))
    show(image)
    plt.axis('off')
    plt.show()
    plt.savefig("results/train_" + dataset_name + "/Self-Attention/Study/" + str(class_idx) + "/img_" + str(id) + ".png")
    plt.clf()

    plt.figure(figsize=(8, 8))
    show(image)
    plt.imshow(attn_weights, cmap='jet', alpha=0.5)  # 'jet' colormap, 50% transparency
    #plt.colorbar()
    #plt.title('Attention Map Overlay')
    
    plt.axis('off')
    plt.show()
    #if not os.path.isdir("results/train_" + dataset_name + "/Self-Attention/Study/" + str(class_idx)):
    #    create_folder("results/train_" + dataset_name + "/Self-Attention/Study/" + str(class_idx))
    plt.savefig("results/train_" + dataset_name + "/Self-Attention/Study/" + str(class_idx) + "/attention_" + dataset_name + "_last_" + str(id) + ".png")
    plt.clf()

def visualize_attention_patch(image, attention_maps, layer_idx):
    # Extract the attention map for the specified layer
    attn_map = attention_maps[0]  # Taking the first element of the batch

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

def embed_patches_with_hog(model, train_classes, dataset_name):
    if dataset_name not in ["cub", "herb"]:
        raise Exception("The dataset_name should be cub or herb!")
    dataset_labels = list(train_classes.keys())
    one_shot_train_data = []
    one_shot_train_label = []

    all_embeddings = []
    train_label = []
    all_constraints = []
    for i in range(len(dataset_labels)):
        images = train_classes[dataset_labels[i]]
        images = images.to("cuda")
        images = images.requires_grad_(True)

        cur_id = 0
        constraints = []
        embeddings = []

        idx = 0

        with torch.no_grad():
            _ = model(images)
            for img in images:
                attn_weights = model[0].get_last_selfattention(img.unsqueeze(0))
        
                img = img.detach().cpu().numpy()
                attn_map = prepare_attention_map(attn_weights, img.shape)
                idx += 1
                
                patches = get_attended_patches(img, attn_map, threshold = 0.5, modification_type="resize", visualize = False)
                cur_img_embeddings = create_embeddings(patches)
                constraints.extend(list(combinations(np.arange(cur_id, cur_id + cur_img_embeddings.shape[0]), 2)))
                cur_id += cur_img_embeddings.shape[0]
                embeddings.append(cur_img_embeddings)
        embeddings = np.vstack(embeddings)
        
        
        all_constraints.extend(constraints)
        train_label.append(np.full(embeddings.shape[0], dataset_labels[i]).flatten())
        all_embeddings.append(embeddings)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    train_label = np.concatenate(train_label, axis=0)
    
    return all_embeddings, train_label, all_constraints

def embed_patches_with_attention(model, train_classes, dataset_name, visualize=False):
    if dataset_name not in ["cub", "herb"]:
        raise Exception("The dataset_name should be cub or herb!")
    dataset_labels = list(train_classes.keys())
    one_shot_train_data = []
    one_shot_train_label = []

    all_patches = []
    all_labels = []
    q = 0
    for i in range(len(dataset_labels)):
        images = train_classes[dataset_labels[i]]
        images = images.to("cuda")
        images = images.requires_grad_(True)

        attention_outputs = []
        cur_id = 0
        #constraints = []
        all_patches_per_class = []
        embeddings = []

        idx = 0

        with torch.no_grad():
            _ = model(images)
            for img in images:
                q += 1
                attn_weights = model[0].get_last_selfattention(img.unsqueeze(0))
                attention_outputs.append(attn_weights)
        
                img = img.detach().cpu().numpy()
                attn_map = prepare_attention_map(attn_weights, img.shape)
                if visualize:
                    visualize_individual_attention(img, attn_map, idx, dataset_labels[i], dataset_name)
                idx += 1
                
                #patches = get_attended_patches(img, attn_map, threshold = 0.5, modification_type="resize", visualize = False)
                patches = get_patches_with_fixed_size(img, attn_map, patch_size=(64, 64), threshold=0.5, visualize=False)
                all_patches_per_class.append(patches)
                #cur_img_embeddings = create_embeddings(patches)
                #constraints.extend(list(combinations(np.arange(cur_id, cur_id + cur_img_embeddings.shape[0]), 2)))
                #cur_id += cur_img_embeddings.shape[0]
                #embeddings.append(cur_img_embeddings)
        all_patches_per_class = np.vstack(all_patches_per_class)
        random_indices = np.random.choice(all_patches_per_class.shape[0], size=int(0.8 * all_patches_per_class.shape[0]), replace=False)
        
        # One random index
        #random_idx = np.random.randint(all_patches_per_class.shape[0])
        
        # Extract randomly one sample per class for training the 1 shot patch learning
        one_shot_train_data.append(all_patches_per_class[random_indices])
        one_shot_train_label.append(np.full(int(0.8 * all_patches_per_class.shape[0]), dataset_labels[i]).flatten())
        mask = np.ones(all_patches_per_class.shape[0], dtype=bool)
        mask[random_indices] = False
        all_patches_per_class =all_patches_per_class[mask]
        # One random index
        #all_patches_per_class = np.delete(all_patches_per_class, random_idx, axis=0)
        all_patches.append(all_patches_per_class)
        all_labels.append(np.full(all_patches_per_class.shape[0], dataset_labels[i]).flatten())
    # Converting them into arrays
    all_patches = np.concatenate(all_patches, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    one_shot_train_data = np.vstack(one_shot_train_data)
    # One random index
    #one_shot_train_label = np.array(one_shot_train_label)
    one_shot_train_label = np.concatenate(one_shot_train_label)
    return all_patches, all_labels, one_shot_train_data, one_shot_train_label

def classify_patches(model, patches):
    model.eval()
    with torch.no_grad():
        logits = model(patches)
        logits2 = logits.argmax(1).cpu().numpy()
        probabilities = F.softmax(logits, dim=1)
        _, predicted = torch.max(probabilities, 1)
    return predicted

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='get_attention', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--train_dataset_path', type=str)
    parser.add_argument('--test_dataset_path', type=str)
    parser.add_argument('--save_shap_values_path', type=str)
    parser.add_argument('--shap_image_plot_name', type=str)
    parser.add_argument('--dataset_name', default="herb", type=str)
    parser.add_argument('--patch_size', default=64, type=int)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
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
    
    
    all_patches, all_labels, one_shot_train_data, one_shot_train_label = embed_patches_with_attention(model, train_classes, args.dataset_name)
    
    
    embeddings = []
    embeddings_classifier = OutputProjSimGCD(model)#.to(device)
    one_shot_train_data = torch.from_numpy(np.transpose(one_shot_train_data, (0, 3, 1, 2))).to(device)
    one_shot_train_data = one_shot_train_data.requires_grad_(True)
    batches = torch.split(one_shot_train_data, 64)
    one_shot_train_label = torch.from_numpy(one_shot_train_label)
    label_batches = torch.split(one_shot_train_label, 64)
    torch.cuda.empty_cache()
    for batch in batches:
        with torch.no_grad():
            a = embeddings_classifier(batch)
        embeddings.append(a.cpu())
    
    embeddings = torch.cat(embeddings, dim=0)
    classifier = RandomForestClassifier(n_estimators=200, random_state=2)
    classifier.fit(embeddings, one_shot_train_label)

    all_patches = torch.from_numpy(np.transpose(all_patches, (0, 3, 1, 2))).to(device)
    all_patches = all_patches.requires_grad_(True)
    batches = torch.split(all_patches, 64)

    all_patches_embeddings = []
    for batch in batches:
        with torch.no_grad():
            a = embeddings_classifier(batch)
        all_patches_embeddings.append(a.cpu())
    all_patches_embeddings = torch.cat(all_patches_embeddings, dim=0)

    predicted_labels = classifier.predict(all_patches_embeddings)

    count = 0
    for i in range(all_labels.shape[0]):
        if all_labels[i] == predicted_labels[i]:
            count += 1
    print(count, all_labels.shape[0])
    print((count/all_labels.shape[0]) * 100)

