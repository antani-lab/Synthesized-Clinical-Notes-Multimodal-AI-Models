import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import shutil
from PIL import Image

import skimage
from skimage import io, color, exposure
import argparse

import shutil
from enum_type import TYPE_IMAGE
from methods_preprocess import find_non_black_square_region, crop_if_too_big, crop_large_side, crop_image
from methods_preprocess import crop_image_by_vignette, get_contrast, get_sharpening, dullrazor

parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-d', '--DATASET', help='dataset to pre-process',type=str, default='HAM10000')
parser.add_argument('-i', '--INPUT_FOLDER', help='path input data (including csv)',type=str, default='')
parser.add_argument('-o', '--OUTPUT_FOLDER', help='path output folder',type=str, default='')
parser.add_argument('-t', '--TYPE_IMAGE', help='dermoscopic/clinical',type=str, default='dermoscopic')
parser.add_argument('-c', '--CSV_FOLDER', help='path csv file (labels.csv)',type=str, default='')
parser.add_argument('-w', '--FLAG_OVERWRITE', help='if overwriting the images',type=str, default='False')

args = parser.parse_args()

DATASET = args.DATASET

new_patch_size = 224
threshold = 20
PERCENTAGE_TO_CUT = 20

type_image = args.TYPE_IMAGE
CSV_FOLDER = args.CSV_FOLDER

DATA_FLD = args.INPUT_FOLDER
OUTPUT_FLD = args.OUTPUT_FOLDER

IMG_FOLDER = DATA_FLD + "/" + DATASET + "/images/"
NEW_FOLDER = OUTPUT_FLD + "/" + DATASET + "/resized_images/"
TYPE_IMAGE_TO_USE = TYPE_IMAGE[type_image]

filename = args.CSV_FOLDER + "/" + DATASET + "/labels.csv"

list_images = os.listdir(IMG_FOLDER)

np.random.shuffle(list_images)

new_patch_size = 224
threshold = 10

flag_overright = args.FLAG_OVERWRITE

if (flag_overright == "False"):
	flag_overright = False
else:
	flag_overright = True

#only for atlas_dermatologico dataset, which includes a band on the bottom
def remove_band_down(img, PERC = 0.90):

	shapeX = img.shape[0]

	limit_X = int(shapeX * PERC)

	new_img = np.copy(img)

	new_img = new_img[:limit_X, :]

	return new_img

THRESHOLD = 20
THRESHOLD_X = 0.05
THRESHOLD_Y = 0.05

list_bad = []

contrast_transformation = get_contrast()
sharpening = get_sharpening()

threshold_black_square = 10
threshold_vignette = 0

for i in tqdm(range(len(list_images))):

	img_fname = list_images[i]
	filename = IMG_FOLDER + img_fname

	fname_new = NEW_FOLDER + img_fname

	if (('.jpg' in img_fname or '.png' in img_fname) and (os.path.exists(fname_new) == False or flag_overright)):
		img = Image.open(filename)
		image_np = np.asarray(img)
		flag_p = False
		try:
			new_patch_size = 224
			image_np = image_np[:,:,:3]
			cropped_img_np = image_np

			if (DATASET == 'atlas_dermatologico'):
				
				cropped_img_np = remove_band_down(cropped_img_np, PERC = 0.90)

			cropped_img_np = dullrazor(cropped_img_np, lowbound = 10, showimgs=False, filterstruc = 3, inpaintmat = 3)

			
			start_y, end_y, start_x, end_x = find_non_black_square_region(cropped_img_np, threshold_black_square)
			cropped_img_np = cropped_img_np[start_y:end_y, start_x:end_x]
			

			
			cropped_img_np = crop_image_by_vignette(cropped_img_np, threshold = threshold_vignette)
			
			cropped_img_np = crop_large_side(cropped_img_np)
			
			cropped_img_np = cropped_img_np[int(THRESHOLD_X * cropped_img_np.shape[0]) : int(cropped_img_np.shape[0] - (THRESHOLD_X * cropped_img_np.shape[0])),  
                                int(THRESHOLD_X * cropped_img_np.shape[1]) : int(cropped_img_np.shape[1] - (THRESHOLD_X * cropped_img_np.shape[1]))]


			cropped_img = Image.fromarray((cropped_img_np).astype(np.uint8))
			cropped_img = cropped_img.resize((new_patch_size, new_patch_size))
			cropped_img_np = np.asarray(cropped_img)

			fname_new = NEW_FOLDER + img_fname
			img = cropped_img_np
			io.imsave(fname_new, img)

		except Exception as e:
			print(e)
			list_bad.append(img_fname)
			print(img_fname)

src = IMG_FOLDER + "labels.csv"
dest = NEW_FOLDER + "labels.csv"

shutil.copy(src, dest)