from openai import OpenAI
import os
import pandas as pd
import numpy as np
import base64
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import cv2
import argparse
from tqdm import tqdm
import time
sys.path.append("../utils/")

from enum_multi import DATASET_TO_USE
import textwrap


parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-d', '--DATASET', help='application of DDCA',type=str, default='BCN20000')
parser.add_argument('-c', '--CSV_FOLDER', help='path of csv folder including metadata (including the dataset folder)',type=str, default='')
parser.add_argument('-i', '--INPUT_PATH', help='path of the folder where to images are stored (including the dataset folder)',type=str, default='')
parser.add_argument('-o', '--OUTPUT_PATH', help='path folder where to store output model (including the dataset folder)',type=str, default='')

args = parser.parse_args()

DATASET = args.DATASET
DATASET_IN_USE = DATASET_TO_USE[DATASET]

hasKey = False
maxlen = 250


CSV_FOLD = args.CSV_FOLDER+DATASET+'/'
csv_test_filename = CSV_FOLD + 'classes_subclasses_metadata_mapping.csv'
csv_file = pd.read_csv(csv_test_filename, sep = ',', header = None, dtype=str).values

labels_filename = CSV_FOLD + "labels.csv"
csv_data = pd.read_csv(labels_filename, sep = ',', header = None, dtype=str).values

IMG_FOLD = args.INPUT_PATH + DATASET + '/resized_images/'

FOLD_NOTES = args.OUTPUT_PATH + DATASET + '/clinical_notes/'
os.makedirs(FOLD_NOTES, exist_ok = True)


folder_store_reports = FOLD_NOTES + "short_reports/"
os.makedirs(folder_store_reports, exist_ok = True)

flag_overwrite = True

np.random.shuffle(csv_file)

def get_note_template(row):

    current_superclass = csv_file[i,2]
    current_class = csv_file[i,3]
    current_subclass = csv_file[i,4]

    if (current_class == current_subclass):
        class_subclass_suffix = ""
    else:
        class_subclass_suffix = " (" + current_subclass + ")"

    current_description = csv_file[i,5]
    current_match_class = csv_file[i,6]
    current_match_subclass = csv_file[i,7]

    if (current_class == current_match_class):
        specific_class_sentence = ""
    else:
        
        if (current_match_class == current_match_subclass):
            matching_subclass_suffix = ""
        else:
            matching_subclass_suffix = " (" + current_match_subclass + ")"

        specific_class_sentence = f"""
        The lesion falls within the {current_match_class}{matching_subclass_suffix} spectrum, representing a pattern consistent with this diagnostic group.
        """.strip()

    note = textwrap.dedent(f"""

    The image is classified within the {current_superclass} category. 
    Its primary diagnostic class is {current_class}{class_subclass_suffix}. 
    Morphologically, it is characterized by {current_description}. 
    {specific_class_sentence}
    """)

    note = note.strip()

    return note

for i in tqdm(range(len(csv_file))):

    fname = csv_file[i,0]

    fname_output = folder_store_reports + fname + ".txt"

    if (os.path.exists(fname_output) == False or flag_overwrite):

        note = get_note_template(csv_file[i])
    
        f = open(fname_output, "w")
        f.write(note)
        f.close() 
