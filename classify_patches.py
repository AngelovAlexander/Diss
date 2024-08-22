import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

if __name__ == "__main__":
    with open('results/train_cub/embeddings_from_hog.pt', 'rb') as embeddings_file:
        embeddings = pickle.load(embeddings_file)
    
    with open('results/train_cub/embedding_labels_from_hog.pt', 'rb') as embedding_labels_file:
        all_labels = pickle.load(embedding_labels_file)
        
    
    random_indices = np.random.choice(all_labels.shape[0], size=int(0.8 * all_labels.shape[0]), replace=False)

    train_labels = all_labels[random_indices]

    mask = np.ones(all_labels.shape[0], dtype=bool)
    mask[random_indices] = False
    test_labels = all_labels[mask]

    train_embeddings = embeddings[random_indices]
    test_embeddings = embeddings[mask]

    classifier = RandomForestClassifier(n_estimators=1000, random_state=4)
    print("Fit")
    classifier.fit(train_embeddings, train_labels)

    print("Predict")
    predicted_labels = classifier.predict(test_embeddings)

    count = 0
    for i in range(test_labels.shape[0]):
        if test_labels[i] == predicted_labels[i]:
            count += 1
    print(count, test_labels.shape[0])
    print((count/test_labels.shape[0]) * 100)