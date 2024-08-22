import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
from sklearn.cluster import DBSCAN, KMeans

def cluster_one_category(embeddings, all_labels, label, reducer="TSNE", clustering_method = "DBSCAN"):
    a = np.where(all_labels == label)[0]
    start_index = np.where(all_labels == label)[0][0]

    end_index = np.where(all_labels == label)[0][-1] + 1

    class_embeddings = embeddings[start_index:end_index]
    if reducer == "TSNE":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, class_embeddings.shape[0] - 1))
    else:
        reducer = PCA(n_components=2)
    reduced_embeddings = reducer.fit_transform(class_embeddings)
    if clustering_method == "DBSCAN":
        dbscan = DBSCAN(eps=1.1, min_samples=3)
        labels = dbscan.fit_predict(reduced_embeddings)
    else:
        kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(reduced_embeddings)
        labels = kmeans.labels_
    
    plt.figure(figsize=(10, 6))
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        xy = reduced_embeddings[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], edgecolors='k', label=f'Cluster {k}')

    plt.title('DBSCAN Clustering')
    plt.legend()
    plt.savefig("results/test/" + reducer.lower() + "_cub_class_1_static_padding_" + clustering_method.lower() + "kmeans.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    with open('results/train_cub/embeddings_from_hog_static_padding.pt', 'rb') as embeddings_file:
        embeddings = pickle.load(embeddings_file)
    
    with open('results/train_cub/embedding_labels_from_hog_static_padding.pt', 'rb') as embedding_labels_file:
        all_labels = pickle.load(embedding_labels_file)
    
    cluster_one_category(embeddings, all_labels, 1)
