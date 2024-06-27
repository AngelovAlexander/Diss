import numpy as np
import shap
import torch
import torchvision
from output_model_SimGCD import OutputSimGCD
import matplotlib.pyplot as plt
from PIL import Image

def shap_benchmark(model, train_dataset, test_dataset):

    def model_wrapper(images):
        _, logits = model(images)
        return logits

    model = model.to("cuda")
    output_model = OutputSimGCD(model)
    
    a = train_dataset[0][0][0].unsqueeze(0)
    a = a.to("cuda")
    a = a.requires_grad_(True)
    output_model.to("cuda")
    e = shap.DeepExplainer(output_model, a)
    t_d = test_dataset[0][0].unsqueeze(0)
    t_d = t_d.to("cuda")
    t_d = t_d.requires_grad_(True)
    y = t_d.detach().clone()
    y = y.detach().cpu().numpy()
    y = y.transpose(0,2,3,1)
    shap_values = e.shap_values(t_d, check_additivity=False)
    np.save("/home/s2602230/SimGCD/shap_values.npy", shap_values)
    shap_numpy = list(np.transpose(shap_values, (4, 0, 2, 3, 1)))

    shap_values = [shap_values[i, 0] for i in range(shap_values.shape[0])]
    shap.image_plot(shap_values, y[0])
    plt.savefig("shap_image_plot.png")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='shap_bnchmark', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--test_dataset_path', type=str)
    parser.add_argument('--save_shap_values_path', type=str)

    parser.add_argument('--shap_image_plot_name', type=str)

    args = parser.parse_args()
    #model = 
    #train_dataset = 
    #test_dataset = 
    #shap_benchmark(model, train_dataset, test_dataset)
