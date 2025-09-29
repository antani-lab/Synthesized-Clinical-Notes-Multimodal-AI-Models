import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
import skimage
import colorsys
from skimage.color import rgb2hsv, hsv2rgb, rgb2lab, lab2rgb
from skimage.exposure import match_histograms
import utils_mask
import albumentations as A
from skimage.feature import canny

def get_pipeline_geometric(prob = 0.5, size = 224):
	
	pipeline_transform = A.Compose([
	A.VerticalFlip(p=prob),
	A.HorizontalFlip(p=prob),
	A.RandomRotate90(p=prob),
	A.RandomResizedCrop(size = (size, size), scale=(0.80, 0.95), interpolation = 2, p=prob),
	])
	
	return pipeline_transform

def get_pipeline_color(prob = 0.5):

	pipeline_transform = A.Compose([
			#A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=-40, val_shift_limit=20, p=prob),  # Shift towards brown
			A.RGBShift(r_shift_limit=(-50,10), g_shift_limit=(-50,10), b_shift_limit=(-50,10), p=prob),
			])
	
	return pipeline_transform


if __name__ == "__main__":
	pass