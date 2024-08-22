import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from scipy.ndimage import label, find_objects

def detect_red_regions(overlaid_img, red_threshold=0.5, saturation_threshold=0.2):
    # Convert to HSV color space
    hsv_img = color.rgb2hsv(overlaid_img)
    
    # Extract Hue and Saturation channels
    hue = hsv_img[:,:,0]
    saturation = hsv_img[:,:,1]
    
    # Create mask for reddish colors (including both red ends of the hue spectrum)
    red_mask = ((hue < 0.05) | (hue > 0.95)) & (saturation > saturation_threshold)
    
    return red_mask.astype(float)

def extract_attended_patches(original_img, attn_mask, patch_size=64):
    # Label connected regions
    labeled, num_features = label(attn_mask)
    
    patches = []
    for obj in find_objects(labeled):
        y, x = obj[0].start, obj[1].start
        h, w = obj[1].stop - obj[1].start, obj[0].stop - obj[0].start
        
        # Calculate center of the region
        center_y, center_x = y + h // 2, x + w // 2
        
        # Define patch boundaries
        y1 = max(0, center_y - patch_size // 2)
        x1 = max(0, center_x - patch_size // 2)
        y2 = min(original_img.shape[0], y1 + patch_size)
        x2 = min(original_img.shape[1], x1 + patch_size)
        
        # Extract patch from original image
        patch = original_img[y1:y2, x1:x2]
        
        # Ensure patch is of the desired size
        if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
            patches.append(patch)
    
    return patches

def crop_images(original_img_path, overlaid_img_path)

    original_img = cv2.imread(original_img_path)
    overlaid_img = cv2.imread(overlaid_img_path)
    
    # Extract attention weights
    red_mask = detect_red_regions(overlaid_img)

    # Extract patches
    patches = extract_attended_patches(original_img, red_mask, patch_size=64)

    # Visualize results
    fig, axes = plt.subplots(1, len(patches) + 3, figsize=(20, 4))

    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Overlaid image
    axes[1].imshow(overlaid_img)
    axes[1].set_title('Overlaid Image')
    axes[1].axis('off')

    # Extracted attention weights
    axes[2].imshow(attn_weights)
    axes[2].set_title('Extracted Attention Weights')
    axes[2].axis('off')

    # Extracted patches
    for i, patch in enumerate(patches):
        axes[i+3].imshow(patch)
        axes[i+3].set_title(f'Patch {i+1}')
        axes[i+3].axis('off')

    plt.tight_layout()
    plt.savefig("results/patch_0.png")