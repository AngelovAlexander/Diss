import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage import color, feature
from craft_interpretability import show
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from sklearn.metrics import silhouette_score, davies_bouldin_score
import cv2

# Class that creates partial functionality of a constrained set
class ConstraintSet:
    def __init__(self):
        self.cannot_link = set()

    def add_cannot_link(self, constraints):
        for i, j in constraints:
            self.cannot_link.add((min(i, j), max(i, j)))

    def can_be_in_same_cluster(self, cluster, point):
        for p in cluster:
            if (min(p, point), max(p, point)) in self.cannot_link:
                return False
        return True

def optimised_CoExDBSCAN(X, eps, min_samples, cannot_link_constraints=None):
    n_samples = X.shape[0]
    
    # Assigning IDs to samples
    sample_ids = np.arange(n_samples)
    
    # Finding neighbors based on a given radius
    neighbors = NearestNeighbors(radius=eps, algorithm='auto', metric='euclidean', n_jobs=-1).fit(X)
    neighborhoods = neighbors.radius_neighbors(X, eps, return_distance=False)
    
    # Divide samples into core samples and other samples
    core_samples, non_core_samples = find_core_and_non_core_samples(neighborhoods, min_samples)
    
    constraint_set = ConstraintSet()
    
    constraint_set.add_cannot_link(cannot_link_constraints)
    
    # Clustering
    labels = np.full(n_samples, -1, dtype=int)
    current_label = 0
    for i in range(n_samples):
        if labels[i] != -1 or not non_core_samples[i]:
            continue
        
        cluster = []
        cluster_ids = []
        stack = [i]
        while stack:
            v = stack.pop()
            if labels[v] == -1:
                if constraint_set.can_be_in_same_cluster(cluster_ids, sample_ids[v]):
                    cluster.append(v)
                    cluster_ids.append(sample_ids[v])
                    labels[v] = current_label
                    if non_core_samples[v]:
                        stack.extend(neighborhoods[v])
        
        current_label += 1
    
    # Assigning non-core points to the cluster with the most neighbors
    for i in range(n_samples):
        if labels[i] == -1:
            cluster_counts = defaultdict(int)
            for neighbor in neighborhoods[i]:
                if labels[neighbor] != -1 and constraint_set.can_be_in_same_cluster([sample_ids[neighbor]], sample_ids[i]):
                    cluster_counts[labels[neighbor]] += 1
            if cluster_counts:
                labels[i] = max(cluster_counts, key=cluster_counts.get)
    
    return labels

def find_core_and_non_core_samples(neighborhoods, min_samples):
    n_samples = len(neighborhoods)
    core_samples = np.zeros(n_samples, dtype=bool)
    non_core_samples = np.zeros(n_samples, dtype=bool)
    neighborhood_sizes = np.array([len(neighbors) for neighbors in neighborhoods])
    
    core_samples = neighborhood_sizes >= min_samples
    
    for i in range(n_samples):
        if core_samples[i]:
            is_exemplar = np.all(neighborhood_sizes[i] >= neighborhood_sizes[neighborhoods[i][core_samples[neighborhoods[i]]]])
            non_core_samples[i] = is_exemplar
    
    return core_samples, non_core_samples

def counts_per_attention_interval(arr):
    # Count the number of pixels between attention levels
    bins = np.arange(0, 1.1, 0.1)
    counts, _ = np.histogram(arr, bins)
    return counts

def get_attended_patches(image, attn, threshold = 0.4, modification_type="random_padding", visualize = False):
    if modification_type not in ["resize", "static_padding", "random_padding"]:
        raise Exception("The modification type needs to be resize, static_padding or random_padding!")
    image = np.transpose(image, (1, 2, 0))
    mask = attn > threshold
    # Label the clusters
    labeled_image, num_features = label(mask)

    # Create a color map for visualization
    cmap = plt.get_cmap('jet')
    if visualize:
        # Visualize the image
        show(image, cmap=cmap)

    patches = []
    mean_dict = {}


    # Loop through each cluster and find bounding boxes
    for cluster_num in range(1, num_features + 1):
        cluster_mask = labeled_image == cluster_num
        coords = np.column_stack(np.where(cluster_mask))

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Normalize the image to be in the range [0,1] and extract the patch
        image -= image.min()
        image /= image.max()
        small_patch = image[y_min:y_max+1, x_min:x_max+1]
        # Getting the mean of the 16 most attended pixels
        # Used for ranking patches
        mean = np.mean(np.sort(small_patch.flatten())[-16:])
        if modification_type == "resize":
            patch = cv2.resize(small_patch, (224,224), interpolation=cv2.INTER_LINEAR)
        elif modification_type == "static_padding":
            patch = np.ones_like(image)
            patch[y_min:y_max+1, x_min:x_max+1] = small_patch
        else:
            patch = np.ones_like(image)
            y_start = np.random.randint(0, patch.shape[0] - small_patch.shape[0] + 1)
            x_start = np.random.randint(0, patch.shape[1] - small_patch.shape[1] + 1)

            patch[y_start:y_start+small_patch.shape[0], x_start:x_start+small_patch.shape[1]] = small_patch
        if num_features > 5:
            if mean in mean_dict:
                mean_dict[mean] = np.vstack([mean_dict[mean], np.expand_dims(patch, axis = 0)])
            else:
                mean_dict[mean] = np.expand_dims(patch, axis = 0)
        else:
            patches.append(np.expand_dims(patch, axis = 0))
        

        if visualize:
            # Draw rectangle around each cluster
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                            edgecolor='green', facecolor='none')
            plt.gca().add_patch(rect)

    if visualize:
        plt.axis('off')
        plt.show()
        plt.savefig("results/attention_with_patches.png")
        plt.clf()
    sorted_means = sorted(mean_dict.keys())
    # Just using the top 5 most attended patches per image
    available_patches = 5
    if num_features > 5:
        while available_patches > 0:
            cur_patch = mean_dict[sorted_means.pop(-1)]
            if available_patches - cur_patch.shape[0] < 0:
                patches.append(cur_patch[:available_patches])
            patches.append(cur_patch)
            available_patches -= cur_patch.shape[0]
    patches = np.vstack(patches)
    return patches

def plot_patches(patches):
    # Plot each patch individually
    num_patches = len(patches)
    fig, axes = plt.subplots(1, num_patches, figsize=(15, 5))

    for i, patch in enumerate(patches):
        if num_patches == 1:
            ax = axes
        else:
            ax = axes[i]
        show(patch, ax, cmap='jet')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig("results/attention_patches.png")
    plt.clf()

def create_patch_embedding(patch, hog_pixels_per_cell=(16, 16), n_bins=32):
    # Convert to grayscale if it's a color image
    if patch.ndim == 3:
        patch_gray = color.rgb2gray(patch)
    else:
        patch_gray = patch

    # Ensure patch is large enough for HOG
    min_size = max(hog_pixels_per_cell) * 2
    if patch_gray.shape[0] < min_size or patch_gray.shape[1] < min_size:
        patch_gray = np.pad(patch_gray, ((0, max(0, min_size - patch_gray.shape[0])),
                                         (0, max(0, min_size - patch_gray.shape[1]))),
                            mode='constant')

    mean = np.mean(patch_gray)
    std = np.std(patch_gray)

    # Histogram of oriented gradients (HOG)
    hog_features = feature.hog(patch_gray, pixels_per_cell=hog_pixels_per_cell,
                               cells_per_block=(1, 1), visualize=False, feature_vector=True)

    # Ensure HOG features have a fixed length; truncate or pad to 64 features
    hog_features = hog_features[:64]
    hog_features = np.pad(hog_features, (0, max(0, 64 - len(hog_features))), mode='constant')

    # Color histogram (if color image)
    if patch.ndim == 3:
        color_hist = np.histogram(patch.flatten(), bins=n_bins, range=(0, 256))[0]
    else:
        color_hist = np.zeros(n_bins)

    embedding = np.concatenate([[mean, std], hog_features, color_hist])

    return embedding

def create_embeddings(patches):
    embeddings = []
    for patch in patches:
        embedding = create_patch_embedding(patch)
        embeddings.append(embedding)
    return np.array(embeddings)

def evaluate_optimised_CoExDBSCAN(X, labels, metric = "silhouette_score"):
    if metric == "silhouette_score":
        return silhouette_score(X, labels)
    elif metric == "davies_bouldin_score":
        return davies_bouldin_score(X, labels)
    else:
        raise ValueError("The evaluation metric should be silhouette_score or davies_bouldin_score.")

def visualize_clusters(X, labels):
    """Visualize the clustering results"""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    colors = plt.cm.jet(np.linspace(0, 1, n_clusters))

    plt.figure(figsize=(10, 8))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'k'  # Black for noise points
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.title(f'CoExDBSCAN Clustering\nNumber of clusters: {n_clusters}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    plt.savefig("results/clustering_results.png")
    plt.clf()