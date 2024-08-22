import matplotlib.pyplot as plt
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot_graph', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file_names', type=str)
    args = parser.parse_args()
    names = args.file_names
    files = names.split(", ")
    avg_loss = []
    for file_name in files:
        temp_losses = []
        with open(file_name) as train_file:
            for line in train_file:
                if "Avg Loss: " in line:
                    temp_losses.append(float(line.split("Avg Loss: ")[-1][:-2]))
        temp_losses = np.array(temp_losses)
        avg_loss.append(temp_losses)
    avg_loss = np.stack(avg_loss, axis=0)
    mean_values = np.mean(avg_loss, axis=0)
    min_values = np.min(avg_loss, axis=0)
    max_values = np.max(avg_loss, axis=0)

    plt.plot(np.arange(1, len(mean_values) + 1), mean_values)
    plt.fill_between(np.arange(1, len(mean_values) + 1), min_values, max_values, color='lightgreen', alpha=0.5)
    #plt.plot(np.arange(75, len(mean_values) + 1), mean_values[74:])
    #plt.fill_between(np.arange(75, len(mean_values) + 1), min_values[74:], max_values[74:], color='lightgreen', alpha=0.5)
    plt.title("CUB's Average loss per epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.show()
    plt.savefig("results/losses_cub.png")