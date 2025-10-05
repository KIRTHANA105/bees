import numpy as np
import cv2

IMG_SIZE = 224
PATCH_SIZE = 56
SEQUENCE_LENGTH = 16

def extract_image_patches(image, patch_size=56):
    h, w = image.shape[:2]
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            if i + patch_size <= h and j + patch_size <= w:
                patch = image[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
    return np.array(patches)

def preprocess_image_to_patches(img):
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    patches = extract_image_patches(img_resized, PATCH_SIZE)

    if len(patches) > SEQUENCE_LENGTH:
        patches = patches[:SEQUENCE_LENGTH]
    elif len(patches) < SEQUENCE_LENGTH:
        padding = np.zeros((SEQUENCE_LENGTH - len(patches), PATCH_SIZE, PATCH_SIZE, 3))
        patches = np.concatenate([patches, padding], axis=0)

    return patches / 255.0
