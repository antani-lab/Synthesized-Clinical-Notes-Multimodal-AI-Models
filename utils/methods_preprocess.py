import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import shutil
from PIL import Image
import cv2
import skimage
from skimage import io, color, exposure
from skimage.filters import threshold_multiotsu
import utils_mask

import albumentations as A

from enum_multi import TYPE_IMAGE

def get_contrast():
	limit_bright = 0.1
	limit_contrast = 0.5
	pipeline_transform = A.Compose([
	A.CLAHE(clip_limit = 1.0, tile_grid_size = (8,8), always_apply = True),
	#A.RandomBrightnessContrast(brightness_limit=(limit_bright, limit_bright), contrast_limit=(limit_contrast, limit_contrast),brightness_by_max=True, always_apply = True) 
			])
	return pipeline_transform

def get_sharpening():

	pipeline_transform = A.Compose([
		A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply = True)
	])
	return pipeline_transform

def find_non_black_square_region(image, threshold = 10):
    # Threshold for considering a pixel as "black" (adjustable)

    # Find where the image is not black (any pixel with values greater than threshold)
    mask = np.any(image > threshold, axis=2)

    # Find coordinates of non-black pixels
    non_black_coords = np.argwhere(mask)

    # If there are no non-black pixels, return None
    if non_black_coords.size == 0:
        return None

    # Determine the bounding box of non-black pixels
    min_y, min_x = np.min(non_black_coords, axis=0)
    max_y, max_x = np.max(non_black_coords, axis=0)

    # Calculate the height and width of the bounding box
    height = max_y - min_y
    width = max_x - min_x

    # Determine the size of the square (max of height and width)
    square_size = max(height, width)

    # Center the square around the middle of the bounding box
    center_y = (min_y + max_y) // 2
    center_x = (min_x + max_x) // 2

    # Calculate new square coordinates
    half_size = square_size // 2
    start_y = max(0, center_y - half_size)
    end_y = min(image.shape[0], center_y + half_size)
    start_x = max(0, center_x - half_size)
    end_x = min(image.shape[1], center_x + half_size)

    # Return the square region coordinates
    return start_y, end_y, start_x, end_x

def crop_if_too_big(image_np, PERCENTAGE_TO_CUT = 10, THRESHOLD_X = 1000, THRESHOLD_Y = 1000):

    PERCENTAGE_TO_CUT = int(PERCENTAGE_TO_CUT / 2)
    
    size_y = image_np.shape[0]
    size_x = image_np.shape[1]

    if (size_x > THRESHOLD_X):

        start_x = int(size_x * (PERCENTAGE_TO_CUT / 100))
        end_x = int(size_x * (1 - (PERCENTAGE_TO_CUT / 100)))

    else:

        start_x = 0
        end_x = size_x

    if (size_y > THRESHOLD_Y):

        start_y = int(size_y * (PERCENTAGE_TO_CUT / 100))
        end_y = int(size_y * (1 - (PERCENTAGE_TO_CUT / 100)))

    else:

        start_y = 0
        end_y = size_y

    cropped_image = image_np[start_y:end_y, start_x:end_x]
    return cropped_image


def eval_mask(image_np, classes = 3):

	greyscale_np = skimage.color.rgb2gray(image_np)
	#greyscale_image = Image.fromarray((greyscale_np * 255).astype(np.uint8))

	thresholds = threshold_multiotsu(greyscale_np, classes = classes)
	mask_np = np.digitize(greyscale_np, bins=thresholds)

	mask_np = np.array(mask_np, dtype=bool)

	return mask_np


def crop_large_side(image_np):

    size_y = image_np.shape[0]
    size_x = image_np.shape[1]

    diff = size_y - size_x

    start_x = 0
    end_x = size_x
    start_y = 0
    end_y = size_y

    surfaces = []

    if (diff >= 0):
        #section left
        start_y = 0
        end_y = start_y + int(size_y / 3) - 1
        
        cropped_image = image_np[start_y:end_y, :]
        mask_np = eval_mask(cropped_image, 2) * 255

        unique, counts = np.unique(mask_np, return_counts=True)
        #print("mask_np_left: " + str(len(mask_np_left)) + ", " + str(dict(zip(unique, counts_left))))
        surfaces.append(counts[0])

        #section central
        start_y = int(size_y / 3)
        end_y = start_y + int(size_y / 3) - 1
        
        cropped_image = image_np[start_y:end_y, :]
        mask_np = eval_mask(cropped_image, 2) * 255

        unique, counts = np.unique(mask_np, return_counts=True)
        #print("mask_np_left: " + str(len(mask_np_left)) + ", " + str(dict(zip(unique, counts_left))))
        surfaces.append(counts[0])

        #section right
        start_y = int(size_y / 3 * 2)
        end_y = start_y + int(size_y / 3) - 1
        
        cropped_image = image_np[start_y:end_y, :]
        mask_np = eval_mask(cropped_image, 2) * 255

        unique, counts = np.unique(mask_np, return_counts=True)
        #print("mask_np_left: " + str(len(mask_np_left)) + ", " + str(dict(zip(unique, counts_left))))
        surfaces.append(counts[0])

        max_surf = np.argmax(surfaces)

        if (max_surf == 0):
            start_x = 0
            end_x = size_x

            start_y = 0
            end_y = start_y + size_x

        elif (max_surf == 1):
            start_x = 0
            end_x = size_x

            diff_to_apply = int(diff / 2)
            start_y = diff_to_apply
            end_y = start_y + size_x

        elif (max_surf == 2):
            start_x = 0
            end_x = size_x

            start_y = -size_x
            end_y = size_y

    else:

        diff = abs(diff)
        #section left
        start_x = 0
        end_x = start_x + int(size_x / 3) - 1
        
        cropped_image = image_np[:, start_x:end_x]
        mask_np = eval_mask(cropped_image, 2) * 255

        unique, counts = np.unique(mask_np, return_counts=True)
        #print("mask_np_left: " + str(len(mask_np_left)) + ", " + str(dict(zip(unique, counts_left))))
        surfaces.append(counts[0])

        #section central
        start_x = int(size_x / 3)
        end_x = start_x + int(size_x / 3) - 1
        
        cropped_image = image_np[:, start_x:end_x]
        mask_np = eval_mask(cropped_image, 2) * 255

        unique, counts = np.unique(mask_np, return_counts=True)
        #print("mask_np_left: " + str(len(mask_np_left)) + ", " + str(dict(zip(unique, counts_left))))
        surfaces.append(counts[0])

        #section right
        start_x = int(size_x / 3 * 2)
        end_x = start_x + int(size_x / 3) - 1
        
        cropped_image = image_np[:, start_x:end_x]
        mask_np = eval_mask(cropped_image, 2) * 255

        unique, counts = np.unique(mask_np, return_counts=True)
        #print("mask_np_left: " + str(len(mask_np_left)) + ", " + str(dict(zip(unique, counts_left))))
        surfaces.append(counts[0])

        max_surf = np.argmax(surfaces)

        if (max_surf == 0):
            start_y = 0
            end_y = size_y

            start_x = 0
            end_x = start_x + size_y

        elif (max_surf == 1):
            start_y = 0
            end_y = size_y

            diff_to_apply = int(diff / 2)
            start_x = diff_to_apply
            end_x = start_x + size_y

        elif (max_surf == 2):
            start_y = 0
            end_y = size_y

            start_x = -size_y
            end_x = size_x

    #print(max_surf)
    cropped_image = image_np[start_y : end_y, start_x : end_x]

    return cropped_image



def find_vignette_boundaries(mean_values, threshold):
    """Finds the first and last positions where the mean values exceed the threshold."""
    above_threshold = np.where(mean_values > threshold)[0]
    if above_threshold.size == 0:
        return 0, len(mean_values)  # If no points exceed threshold, keep the full length
    return above_threshold[0], above_threshold[-1]

def analyze_diagonals(image, threshold):
    """Analyzes both main diagonals of the image to detect vignette boundaries."""
    h, w, _ = image.shape
    min_dim = min(h, w)

    # Extract pixel values along both diagonals
    diagonal1_pixels = image[np.arange(min_dim), np.arange(min_dim)]
    diagonal2_pixels = image[np.arange(min_dim), np.arange(w - 1, w - min_dim - 1, -1)]

    # Compute mean intensity for each pixel along both diagonals
    diagonal1_mean = np.mean(diagonal1_pixels, axis=1)
    diagonal2_mean = np.mean(diagonal2_pixels, axis=1)

    # Get start and end points where vignette fades out on both diagonals
    start1, end1 = find_vignette_boundaries(diagonal1_mean, threshold)
    start2, end2 = find_vignette_boundaries(diagonal2_mean, threshold)

    # Calculate restrictive cropping box from detected boundaries
    top_crop = max(start1, start2)
    bottom_crop = min(end1, end2)
    left_crop = max(start1, start2)
    right_crop = min(end1, end2)

    return top_crop, bottom_crop, left_crop, right_crop

def crop_image_by_vignette(image, threshold):
    """Crops the image based on vignette boundaries detected along the diagonals."""
    h, w, _ = image.shape

    # Get cropping boundaries
    top_crop, bottom_crop, left_crop, right_crop = analyze_diagonals(image, threshold)

    # Crop the RGB image using the computed boundaries
    cropped_image = image[top_crop:h - (min(h, w) - bottom_crop), left_crop:w - (min(h, w) - right_crop)]

    return cropped_image

def crop_image(image_np, PERCENTAGE_TO_CUT = 10):

    PERCENTAGE_TO_CUT = int(PERCENTAGE_TO_CUT / 2)

    size_y = image_np.shape[0]
    size_x = image_np.shape[1]

    start_x = int(size_x * (PERCENTAGE_TO_CUT / 100))
    end_x = int(size_x * (1 - (PERCENTAGE_TO_CUT / 100)))

    start_y = int(size_y * (PERCENTAGE_TO_CUT / 100))
    end_y = int(size_y * (1 - (PERCENTAGE_TO_CUT / 100)))

    cropped_image = image_np[start_y:end_y, start_x:end_x]
    return cropped_image


def dullrazor(img, lowbound=15, showimgs=False, filterstruc=3, inpaintmat=3):
    #grayscale
    imgtmp1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #applying a blackhat
    filterSize =(filterstruc, filterstruc)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize) 
    imgtmp2 = cv2.morphologyEx(imgtmp1, cv2.MORPH_BLACKHAT, kernel)

    #0=skin and 255=hair
    ret, mask = cv2.threshold(imgtmp2, lowbound, 255, cv2.THRESH_BINARY)
    
    #inpainting
    img_final = cv2.inpaint(img, mask, inpaintmat ,cv2.INPAINT_TELEA)
    
    
    return img_final

def cut_bottom_line(img_np, PERC = 0.9):

    new_y = int(img_np.shape[0] * PERC)
    new_img = np.copy(img_np)

    new_img = img_np[:new_y, :, :]
    return new_img

if __name__ == "__main__":
    pass