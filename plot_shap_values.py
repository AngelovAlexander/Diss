import shap
import numpy as np

shap_values = np.load("/home/s2602230/SimGCD/shap_values.npy")

#train_slice = [train_dataset[i] for i in range(1, 3)]
#features = [data[0] for data in train_slice]
#features_array = np.array(features)
shap.image_plot(shap_values)