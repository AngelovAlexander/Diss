import cv2
import argparse
import matplotlib.pyplot as plt
from craft_interpretability import show
import numpy as np
import matplotlib.image as mpimg

def combine_two_imgs_horizontal(*images):
    return cv2.hconcat(images)
def combine_two_imgs_vertical(*images):
    return cv2.vconcat(images)

def normalize_img(image):
    image = np.array(image)
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)

    image = image.astype(np.float64)
    image -= image.min()
    image /= image.max()
    return image

def plot_axis(images, ax, category):
    ax.imshow(combine_two_imgs_horizontal(*images))
    #ax.set_title('Category ' + str(category))
    ax.axis('off')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='combine_images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--images_path', type=str)
    parser.add_argument('--image_names', type=str)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    img_names = args.image_names.split(",")
    imgs = [mpimg.imread(args.images_path + i_n) for i_n in img_names]
    save_paths = args.save_path.split(",")

    num_images_per_subplot = 2
    num_subplots = len(imgs) // num_images_per_subplot

    fig, axs = plt.subplots(1, num_subplots, figsize=(20, 5))

    for i in range(num_subplots):
        start_idx = i * num_images_per_subplot
        end_idx = start_idx + num_images_per_subplot
        plot_axis(imgs[start_idx:end_idx], axs[i], i + 1)

    plt.show()
    plt.savefig(save_paths[0])
    plt.clf()