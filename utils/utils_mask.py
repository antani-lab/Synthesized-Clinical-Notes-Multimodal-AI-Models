import sys, getopt
import numpy as np 
import pandas as pd
from scipy.ndimage import distance_transform_edt

import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import label
import skimage
from skimage import io, color, exposure
from skimage.exposure import match_histograms

import albumentations as A
from skimage.filters import threshold_multiotsu, threshold_yen, threshold_li, threshold_local
import scipy.ndimage as nd
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation, binary_erosion, binary_closing
import random
import math
from enum_multi import PORTION
from scipy.ndimage import convolve

def keep_largest_region(mask):
    # Label connected components
    labeled_mask, num_features = nd.label(mask == False)
    
    # Find the size of each region
    region_sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
    
    # Find the label of the largest region
    largest_region_label = np.argmax(region_sizes) + 1  # labels start from 1
    
    # Create a new mask where all regions except the largest one are set to True
    new_mask = (labeled_mask == largest_region_label)  # Keep the largest region as False, rest are True
    return new_mask

#create a numpy array mask with True corresponding to skin, False corresponding to lesion
def eval_mask(image, classes = 3):

	thresholds = threshold_multiotsu(image, classes = classes)
	mask_np = np.digitize(image, bins=thresholds)
	mask_np = np.array(mask_np, dtype=bool)

	return mask_np

def eval_mask_otsu(image, classes = 2):

	thresholds = threshold_multiotsu(image, classes = classes)
	mask_np = np.digitize(image, bins=thresholds)
	mask_np = np.array(mask_np, dtype=bool)

	return mask_np 

def eval_mask_yen(image):

	thresholds = threshold_yen(image)
	mask_np = image > thresholds
	mask_np = np.array(mask_np, dtype=bool)

	return mask_np

def eval_mask_li(image):

	thresholds = threshold_li(image)
	mask_np = image > thresholds
	mask_np = np.array(mask_np, dtype=bool)

	return mask_np

def eval_mask_gaussian_otsu(image, sigma = 1.5):

	smooth = skimage.filters.gaussian(image, sigma = sigma)
	thresh = smooth > skimage.filters.threshold_otsu(smooth)
	fill = nd.binary_fill_holes(thresh)
	mask_np = skimage.segmentation.clear_border(fill)
	mask_np = np.array(mask_np, dtype=bool)

	return mask_np

def get_first_true_corners(arr):
	X, Y = arr.shape
	corner_coords = {}

	# Top-left corner
	row_idx = np.where(arr[0, :])[0]  # First row
	col_idx = np.where(arr[:, 0])[0]  # First column (from top-left)
	corner_coords["top_left"] = (
		(0, row_idx[0]) if len(row_idx) > 0 else (0, 0),
		(col_idx[0], 0) if len(col_idx) > 0 else (0, 0)
	)

	# Top-right corner
	row_idx = np.where(arr[0, :])[0]  # First row
	col_idx = np.where(arr[:, -1])[0]  # Last column (from top-right)
	corner_coords["top_right"] = (
		(0, row_idx[-1]) if len(row_idx) > 0 else (0,223),
		(col_idx[0], Y-1) if len(col_idx) > 0 else (0,223)
	)

	# Bottom-left corner
	row_idx = np.where(arr[-1, :])[0]  # Last row
	col_idx = np.where(arr[:, 0])[0]  # First column (from top-left)
	corner_coords["bottom_left"] = (
		(X-1, row_idx[0]) if len(row_idx) > 0 else (223, 0),
		(col_idx[-1], 0) if len(col_idx) > 0 else (223, 0)
	)

	# Bottom-right corner
	row_idx = np.where(arr[-1, :])[0]  # Last row
	col_idx = np.where(arr[:, -1])[0]  # Last column (from top-right)
	corner_coords["bottom_right"] = (
		(X-1, row_idx[-1]) if len(row_idx) > 0 else (223,223),
		(col_idx[-1], Y-1) if len(col_idx) > 0 else (223,223)
	)

	return corner_coords


def evaluate_corners(arr, THRESH = 0.1):
    
	X_copy = np.copy(arr)
	coords = get_first_true_corners(X_copy)
	#print(coords)
	THRESHOLD = 0.90
	PATCH_SIZE = 224
	#top left
	top_left = coords['top_left']
	square_1 = X_copy[:top_left[0][1], :top_left[0][1]]
	square_2 = X_copy[:top_left[1][0], :top_left[1][0]]

	perc_1 = np.mean(square_1)
	perc_2 = np.mean(square_2)
      
	if (perc_1 > THRESH and perc_2 > THRESH):
		if (top_left[0][1] > top_left[1][0]):
			X_copy[:top_left[0][1], :top_left[0][1]] = True
		else:
			X_copy[:top_left[1][0], :top_left[1][0]] = True
	#print(perc_1, perc_2)

	top_right = coords['top_right']
	square_1 = X_copy[:PATCH_SIZE-1-top_right[0][1], top_right[0][1]:]
	square_2 = X_copy[:top_right[1][0], PATCH_SIZE-1-top_right[1][0]:]
	perc_1 = np.mean(square_1)
	perc_2 = np.mean(square_2)
	if (perc_1 > THRESH and perc_2 > THRESH):
		if (PATCH_SIZE-1-top_right[0][1] > top_right[1][0]):
			X_copy[:PATCH_SIZE-1-top_right[0][1], top_right[0][1]:] = True
		else:
			X_copy[:top_right[1][0], PATCH_SIZE-1-top_right[1][0]:] = True
	#print(perc_1, perc_2)

	#bottom left
	bottom_left = coords['bottom_left']
	square_1 = X_copy[PATCH_SIZE-1-bottom_left[0][1]:, :bottom_left[0][1]]
	square_2 = X_copy[bottom_left[1][0]:, :PATCH_SIZE-1-bottom_left[1][0]]
	perc_1 = np.mean(square_1)
	perc_2 = np.mean(square_2)
	if (perc_1 > THRESH and perc_2 > THRESH):
		if (bottom_left[0][1] > PATCH_SIZE-1-bottom_left[1][0]):
			X_copy[PATCH_SIZE-1-bottom_left[0][1]:, :bottom_left[0][1]] = True
		else:
			X_copy[bottom_left[1][0]:, :PATCH_SIZE-1-bottom_left[1][0]] = True
	#print(perc_1, perc_2)

	#top right
	bottom_right = coords['bottom_right']
	square_1 = X_copy[bottom_right[0][1]: , bottom_right[0][1]:]
	square_2 = X_copy[bottom_right[1][0]: , bottom_right[1][0]:]

	perc_1 = np.mean(square_1)
	perc_2 = np.mean(square_2)
	if (perc_1 > THRESH and perc_2 > THRESH):
		if (PATCH_SIZE-1-bottom_right[0][1] > PATCH_SIZE-1-bottom_right[1][0]):
			X_copy[PATCH_SIZE-1-bottom_right[0][1]: , bottom_right[0][1]:] = True
		else:
			X_copy[bottom_right[1][0]: , bottom_right[1][0]:] = True
	#print(perc_1, perc_2)
		
	return X_copy


def is_vignette(mask, p = 0.8, PERC_TRUE = 0.99) -> tuple:
	
	flag_center = False
	flag_outside = False


	Y_dim, X_dim = mask.shape
	cx, cy = X_dim // 2, Y_dim // 2  # Center coordinates

	side_from_center_Y = int((Y_dim * p) / 2)
	side_from_center_X = int((X_dim * p) / 2)

	# Extract the central region
	center = mask[cy - side_from_center_Y : cy + side_from_center_Y, 
					cx - side_from_center_X : cx + side_from_center_X]

	# Compute percentage of True values in central region
	center_true_percentage = np.mean(center)  # Mean of boolean array gives ratio of True values

	# Compute percentage of True values in complementary region
	complementary_mask = mask.copy()
	complementary_mask[cy - side_from_center_Y : cy + side_from_center_Y, 
						cx - side_from_center_X : cx + side_from_center_X] = False  # Zero out central region
	complementary_true_percentage = np.mean(complementary_mask)

	flag_center = center_true_percentage >= PERC_TRUE or center_true_percentage <= (1-PERC_TRUE)
	flag_outside = complementary_true_percentage <= (1 - PERC_TRUE)

	flag = flag_center or flag_outside

	return flag


def mask_coverage(mask):
	"""Returns the proportion of True pixels in the mask."""
	return np.mean(mask)

def largest_mask_region(mask):
	"""Finds the size of the largest connected region of True pixels."""
	labeled_mask, num_features = label(mask)  # Label connected components
	if num_features == 0:
		return 0
	region_sizes = np.bincount(labeled_mask.ravel())[1:]  # Exclude background (label 0)
	return np.max(region_sizes)

def mask_compactness(mask):
	"""Computes compactness: perimeter^2 / area."""
	perimeter = np.sum(nd.binary_dilation(mask) ^ mask)  # Count edge pixels
	area = np.sum(mask)  # Count True pixels
	return (perimeter ** 2) / (area + 1e-6)  # Avoid division by zero



def smooth_mask(mask, sigma = 1):
    """
    Smooths a binary mask to make it more rounded using Gaussian blur and thresholding.

    Parameters:
        mask (np.ndarray): A (X, X) boolean NumPy array representing the mask.
        sigma (float): The standard deviation for Gaussian blur (higher values increase smoothness).

    Returns:
        np.ndarray: The modified binary mask with rounded edges.
    """
    smoothed = gaussian_filter(mask.astype(float), sigma=sigma)  # Apply Gaussian blur
    return smoothed > 0.5  # Convert back to binary using thresholding

def expand_false_regions(mask, N=1, p=0.5):
	"""
	Expands False regions in the mask by randomly converting True pixels 
	adjacent to False regions into False.
	
	Parameters:
	mask (ndarray): Binary NumPy array of shape (X, Y), where False represents the mask.
	N (int): Neighborhood distance to consider for expansion.
	p (float): Probability of turning an adjacent True pixel into False.
	
	Returns:
	ndarray: Updated mask with expanded False regions.
	"""
	X, Y = mask.shape
	expanded_mask = mask.copy()

	# Find False pixels
	false_positions = np.argwhere(mask == False)

	# Define the expansion directions
	offsets = [(dx, dy) for dx in range(-N, N+1) for dy in range(-N, N+1) if (dx, dy) != (0, 0)]

	# Loop through each False pixel
	for x, y in false_positions:
		for dx, dy in offsets:
			nx, ny = x + dx, y + dy
			if 0 <= nx < X and 0 <= ny < Y and mask[nx, ny]:  # If neighbor is True
				if np.random.rand() < p:  # Random chance to turn it False
					expanded_mask[nx, ny] = False
	
	return expanded_mask

def apply_mask(image, mask):
    """
    Applies a binary mask to an image, setting all False regions to black.

    Parameters:
    - image: NumPy array of shape (X, Y, 3) representing an RGB image.
    - mask: NumPy array of shape (X, Y) with True/False values.

    Returns:
    - The masked image with False regions set to black.
    """
    return image * mask[..., None]  # Expands mask to (X, Y, 1) for broadcasting


def create_perturbation_mask(mask, perturb_min = 0.95, perturb_max = 1.2, img_size = 224):
    
	mask = mask.astype(int)
	random_mask = np.random.uniform(perturb_min, perturb_max, (img_size, img_size))

	return mask * random_mask

def expand_false_towards_corners(mask, N=1, p=0.5):
	"""
	Expands False regions in the mask towards the corners, avoiding the center.

	Parameters:
	mask (ndarray): Binary NumPy array (X, Y) where False represents the mask.
	N (int): Expansion range.
	p (float): Probability of converting a selected True pixel to False.

	Returns:
	ndarray: Updated mask with False regions expanded towards the corners.
	"""
	X, Y = mask.shape
	expanded_mask = mask.copy()

	# Compute center of the image
	cx, cy = X // 2, Y // 2

	# Find False pixels
	false_positions = np.argwhere(mask == False)

	for x, y in false_positions:
		# Determine direction towards the nearest corner
		dx = -1 if x < cx else 1  # Move up if above center, down if below
		dy = -1 if y < cy else 1  # Move left if left of center, right if right

		for _ in range(N):  # Expand in multiple steps if needed
			nx, ny = x + dx, y + dy
			if 0 <= nx < X and 0 <= ny < Y and mask[nx, ny]:  # If neighbor is True
				if np.random.rand() < p:  # Random chance to turn it False
					expanded_mask[nx, ny] = False
				x, y = nx, ny  # Move further in the same direction

	return expanded_mask


def get_mask(img_np, hsv_np, lab_np):
	
	images = []
	scores = []
	flags = []

	greyscale_np = skimage.color.rgb2gray(img_np)

	#otsu grey, S, V, L, a, b

	mask = eval_mask_otsu(greyscale_np)
	perc_1 = np.mean(mask)
	images.append(mask)
	score = mask_compactness(mask)
	scores.append(score)
	flag = is_vignette(mask)
	flags.append(flag)

	mask = ~eval_mask_otsu(hsv_np[:,:,1])
	perc_2 = np.mean(mask)
	images.append(mask)
	score = mask_compactness(mask)
	scores.append(score)
	flag = is_vignette(mask)
	flags.append(flag)

	mask = eval_mask_otsu(hsv_np[:,:,2])
	perc_3 = np.mean(mask)
	images.append(mask)
	score = mask_compactness(mask)
	scores.append(score)
	flag = is_vignette(mask)
	flags.append(flag)

	#li
	mask = eval_mask_li(greyscale_np)
	perc_1 = np.mean(mask)
	images.append(mask)
	score = mask_compactness(mask)
	scores.append(score)
	flag = is_vignette(mask)
	flags.append(flag)

	mask = ~eval_mask_li(hsv_np[:,:,1])
	perc_2 = np.mean(mask)
	images.append(mask)
	score = mask_compactness(mask)
	scores.append(score)
	flag = is_vignette(mask)
	flags.append(flag)

	mask = eval_mask_li(hsv_np[:,:,2])
	perc_3 = np.mean(mask)
	images.append(mask)
	score = mask_compactness(mask)
	scores.append(score)
	flag = is_vignette(mask)
	flags.append(flag)

	#yen
	mask = eval_mask_yen(greyscale_np)
	perc_1 = np.mean(mask)
	images.append(mask)
	score = mask_compactness(mask)
	scores.append(score)
	flag = is_vignette(mask)
	flags.append(flag)

	mask = ~eval_mask_yen(hsv_np[:,:,1])
	perc_2 = np.mean(mask)
	images.append(mask)
	score = mask_compactness(mask)
	scores.append(score)
	flag = is_vignette(mask)
	flags.append(flag)

	mask = eval_mask_yen(hsv_np[:,:,2])
	perc_3 = np.mean(mask)
	images.append(mask)
	score = mask_compactness(mask)
	scores.append(score)
	flag = is_vignette(mask)
	flags.append(flag)

	"""
	flag_color = np.mean([perc_1, perc_2, perc_3]) > 0.5

	mask = eval_mask_yen(lab_np[:,:,1])
	flag_mask = np.mean(mask) > 0.5
	if (flag_color != flag_mask):
		mask = ~mask
	images.append(mask)
	score = mask_compactness(mask)
	scores.append(score)
	flag = is_vignette(mask)
	flags.append(flag)

	mask = eval_mask_yen(lab_np[:,:,2])
	flag_mask = np.mean(mask) > 0.5
	if (flag_color != flag_mask):
		mask = ~mask
	images.append(mask)
	score = mask_compactness(mask)
	scores.append(score)
	flag = is_vignette(mask)
	flags.append(flag)
	"""

	"""
	#gaussian
	mask = ~eval_mask_gaussian_otsu(greyscale_np)
	images.append(mask)
	score = mask_compactness(mask)
	scores.append(score)
	flag = is_vignette(mask)
	flags.append(flag)

	mask = ~eval_mask_gaussian_otsu(hsv_np[:,:,1])
	images.append(mask)
	score = mask_compactness(mask)
	scores.append(score)
	flag = is_vignette(mask)
	flags.append(flag)

	mask = eval_mask_gaussian_otsu(hsv_np[:,:,2])
	images.append(mask)
	score = mask_compactness(mask)
	scores.append(score)
	flag = is_vignette(mask)
	flags.append(flag)

	mask = ~eval_mask_gaussian_otsu(lab_np[:,:,1])
	images.append(mask)
	score = mask_compactness(mask)
	scores.append(score)
	flag = is_vignette(mask)
	flags.append(flag)

	mask = ~eval_mask_gaussian_otsu(lab_np[:,:,2])
	images.append(mask)
	score = mask_compactness(mask)
	scores.append(score)
	flag = is_vignette(mask)
	flags.append(flag)
	"""
	sorted_indeces = np.argsort(scores)#[:3]  # Sort and take the first two indices

	return sorted_indeces, images, scores, flags

def preprocessing_mask(mask, iterations_closing = 1, iterations_dilatation = 2):
	
	if (iterations_closing > 0):
		mask = ~binary_closing(~mask, iterations = iterations_closing)

	if (iterations_dilatation > 0):
		mask = ~binary_dilation(~mask, iterations = iterations_dilatation)

	return mask

def add_border_true(arr, P, S = 30):
	"""Sets the P-pixel-wide border of a boolean NumPy array to True.

	Args:
		arr (numpy.ndarray): A 2D boolean NumPy array.
		P (int): The border width to set to True.

	Returns:
		numpy.ndarray: The modified array with the border set to True.
	"""
	if P <= 0:
		return arr  # No modification needed

	x = arr.shape[0]
	start = (x - S) // 2  # Compute starting index for centering SxS
	end = start + S  # Compute ending index

	central_region = arr[start:end, start:end]  # Extract central region
	flag = np.all(central_region) 

	if (flag == False):
		arr[:P, :] = True  # Top border
		arr[-P:, :] = True  # Bottom border
		arr[:, :P] = True  # Left border
		arr[:, -P:] = True  # Right border

	return arr

def flood_fill(mask, labeled, i, j, label):
	"""
	Performs flood-fill (region-growing) to label a connected component.
	
	mask: (x, x) NumPy binary array (True = background, False = foreground).
	labeled: Array storing labeled components.
	i, j: Starting position for flood fill.
	label: The label to assign.
	
	Returns:
	- Size of the connected component.
	"""
	x, y = mask.shape
	stack = [(i, j)]
	size = 0

	while stack:
		ci, cj = stack.pop()
		if labeled[ci, cj] == 0 and not mask[ci, cj]:  # Unvisited and part of the False region
			labeled[ci, cj] = label
			size += 1
			# Add 4-connected neighbors
			for ni, nj in [(ci-1, cj), (ci+1, cj), (ci, cj-1), (ci, cj+1)]:
				if 0 <= ni < x and 0 <= nj < y and labeled[ni, nj] == 0:
					stack.append((ni, nj))

	return size

def find_largest_false_region(mask):
	"""
	Identifies the largest connected region of False values.
	
	mask: (x, x) NumPy binary array (True = background, False = foreground).
	
	Returns:
	- labeled: NumPy array with labeled components.
	- largest_label: Label corresponding to the largest False region.
	"""
	labeled = np.zeros_like(mask, dtype=int)
	largest_label = None
	largest_size = 0
	label = 1

	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if not mask[i, j] and labeled[i, j] == 0:  # Unvisited False pixel
				size = flood_fill(mask, labeled, i, j, label)
				if size > largest_size:
					largest_size = size
					largest_label = label
				label += 1

	return labeled, largest_label

def fill_inside_block(mask):
	"""
	Converts all True values inside the largest False block to False.
	
	mask: (x, x) NumPy binary array (True = background, False = foreground).
	
	Returns:
	- Updated mask with all internal islands set to False.
	"""
	labeled, largest_label = find_largest_false_region(mask)

	if largest_label is not None:
		mask[labeled == largest_label] = False  # Set all pixels inside the largest False block to False

	return mask


def mask_difference(mask_A, mask_B):
	return ~(mask_A ^ mask_B)

def expand_false_towards_corners(mask, N=10, p=0.5):
    """
    Expands False regions in the mask towards the corners, avoiding the center, with a probability
    that decreases as distance from the original False region increases.

    Parameters:
        mask (ndarray): Binary NumPy array (X, Y) where False represents the mask.
        N (int): Maximum expansion range.
        p (float): Initial probability of converting a selected True pixel to False.

    Returns:
        ndarray: Updated mask with False regions expanded towards the corners.
    """
    X, Y = mask.shape
    expanded_mask = mask.copy()

    # Compute the distance from each True pixel to the nearest False pixel
    #distance_map = distance_transform_edt(mask)

    # Find False pixel positions (original mask)
    false_positions = np.argwhere(mask == False)

    for x, y in false_positions:
        # Determine direction towards the nearest corner
        dx = -1 if x < X // 2 else 1  # Move up if above center, down if below
        dy = -1 if y < Y // 2 else 1  # Move left if left of center, right if right

        for step in range(1, N + 1):  # Expand up to N steps
            nx, ny = x + step * dx, y + step * dy
            if 0 <= nx < X and 0 <= ny < Y and expanded_mask[nx, ny]:  # If neighbor is True
                # Compute dynamic probability (decreasing with distance)
                probability = p * (1 - step / N)
                if np.random.rand() < probability:
                    expanded_mask[nx, ny] = False  # Convert pixel to False

    return expanded_mask

def random_border_erosion(mask, p=0.1):
    """
    Randomly erodes the border of False regions in a binary mask.

    Parameters:
        mask (ndarray): Binary NumPy array (X, Y) where False represents the mask.
        p (float): Probability of converting a False border pixel to True.

    Returns:
        ndarray: Updated mask with the border of False regions eroded.
    """
    eroded_mask = mask.copy()

    # Define a 3x3 kernel to detect False pixels with at least one True neighbor
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    # Count the number of True neighbors for each pixel
    true_neighbors = convolve(mask.astype(int), kernel, mode='constant', cval=1)

    # Border pixels are False pixels with at least one True neighbor
    border_pixels = (mask == False) & (true_neighbors > 0)

    # Randomly select which border pixels to erode
    random_mask = np.random.rand(*mask.shape) < p
    pixels_to_erode = border_pixels & random_mask

    # Convert selected pixels to True
    eroded_mask[pixels_to_erode] = True

    return eroded_mask


def random_border_erosion(mask, num_pixels=10, p=0.5):
    """
    Randomly erodes a given number of False border pixels in a binary mask, 
    with a probability for each border pixel.

    Parameters:
        mask (ndarray): Binary NumPy array (X, Y) where False represents the mask.
        num_pixels (int): Maximum number of False border pixels to erode.
        p (float): Probability of eroding each detected border pixel.

    Returns:
        ndarray: Updated mask with the border of False regions eroded.
    """
    eroded_mask = mask.copy()

    # Define a 3x3 kernel to detect False pixels with at least one True neighbor
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    # Count the number of True neighbors for each pixel
    true_neighbors = convolve(mask.astype(int), kernel, mode='constant', cval=1)

    # Find all border pixels (False pixels with at least one True neighbor)
    border_pixels = np.argwhere((mask == False) & (true_neighbors > 0))

    if len(border_pixels) == 0:
        return eroded_mask  # No erosion possible if no border pixels found

    # Apply probability filter to select some border pixels
    selected_mask = np.random.rand(len(border_pixels)) < p
    selected_pixels = border_pixels[selected_mask]

    # Limit the number of pixels to erode
    num_pixels = min(num_pixels, len(selected_pixels))

    if num_pixels > 0:
        # Randomly choose `num_pixels` from the selected pixels
        final_indices = np.random.choice(len(selected_pixels), num_pixels, replace=False)
        final_pixels = selected_pixels[final_indices]

        # Convert selected border pixels to True (erode)
        eroded_mask[final_pixels[:, 0], final_pixels[:, 1]] = True

    return eroded_mask

def switch_false_to_true(arr, p = 0.5):
    """
    Randomly switches a percentage 'p' of False values to True in the given NumPy array.

    Parameters:
        arr (np.ndarray): A (X, Y) boolean NumPy array.
        p (float): The percentage (0 to 1) of False values to switch to True.

    Returns:
        np.ndarray: The modified array.
    """
    arr_copy = np.copy(arr)
    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1")

    false_indices = np.argwhere(arr_copy == False)  # Get indices of False values
    num_to_switch = int(len(false_indices) * p)  # Number of False values to switch

    if num_to_switch > 0:
        selected_indices = false_indices[np.random.choice(len(false_indices), num_to_switch, replace=False)]
        arr_copy[selected_indices[:, 0], selected_indices[:, 1]] = True

    return arr_copy

def switch_true_to_false(arr, p):
    """
    Randomly switches a percentage 'p' of True values to False in the given NumPy array.

    Parameters:
        arr (np.ndarray): A (X, Y) boolean NumPy array.
        p (float): The percentage (0 to 1) of True values to switch to False.

    Returns:
        np.ndarray: The modified array.
    """
    arr_copy = np.copy(arr)
    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1")

    true_indices = np.argwhere(arr_copy == True)  # Get indices of True values
    num_to_switch = int(len(true_indices) * p)  # Number of True values to switch

    if num_to_switch > 0:
        selected_indices = true_indices[np.random.choice(len(true_indices), num_to_switch, replace=False)]
        arr_copy[selected_indices[:, 0], selected_indices[:, 1]] = False

    return arr_copy



if __name__ == "__main__":
	pass