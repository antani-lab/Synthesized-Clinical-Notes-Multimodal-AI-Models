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

parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-d', '--DATASET', help='application of DDCA',type=str, default='BCN20000')
parser.add_argument('-g', '--GPT', help='GPT to use',type=str, default='gpt-4o-mini')
parser.add_argument('-k', '--KEY_PATH', help='GPT to use',type=str, default='')

parser.add_argument('-c', '--CSV_FOLDER', help='path of csv folder including metadata (including the dataset folder)',type=str, default='')
parser.add_argument('-i', '--INPUT_PATH', help='path of the folder where to images are stored (including the dataset folder)',type=str, default='')
parser.add_argument('-o', '--OUTPUT_PATH', help='path folder where to store output model (including the dataset folder)',type=str, default='')

args = parser.parse_args()

DATASET = args.DATASET
DATASET_IN_USE = DATASET_TO_USE[DATASET]

GPT_TO_USE = args.GPT


def check_api(key):
    client = OpenAI(api_key=key)
    try:
        messages = [{'role': "user", 'content': "How are you today?"},]
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.1, max_tokens=2)
        return True

    except Exception as e:
        print(f"call failed: {e}")
        return False

hasKey = False
maxlen = 300


#load key
key_path = args.KEY_PATH
#"""
with open(key_path, "r") as f:
    key = f.read().strip()
    #print(key)
key_check = check_api(key)
if key_check:
    print("The API key has been set successfully. NIH \n")
    hasKey = True
    client = OpenAI(api_key=key)


CSV_FOLD = args.CSV_FOLDER+DATASET+'/'
csv_test_filename = CSV_FOLD + 'classes_subclasses_metadata_mapping.csv'
csv_file = pd.read_csv(csv_test_filename, sep = ',', header = None, dtype=str).values

labels_filename = CSV_FOLD + "labels.csv"
csv_data = pd.read_csv(labels_filename, sep = ',', header = None, dtype=str).values

IMG_FOLD = args.INPUT_PATH + DATASET + '/resized_images/'

FOLD_NOTES = args.OUTPUT_PATH + DATASET + '/clinical_notes/'
os.makedirs(FOLD_NOTES, exist_ok = True)


folder_store_reports = FOLD_NOTES + 'char/'
os.makedirs(folder_store_reports, exist_ok = True)


def get_filename(sample_id, csv_data):

    i = 0
    b = False
    fname = -1

    for i in range(len(csv_data)):

        current_fname = csv_data[i,0]
        if (sample_id in current_fname):
            fname = current_fname
            b = True
        else:
            i = i + 1

    return fname

def get_instruction_and_question(metadata, maxlen):

    current_superclass = metadata[2]
    current_class = metadata[3]
    current_subclass = metadata[4]
    current_description = metadata[5]
    current_match_class = metadata[6]
    current_match_subclass = csv_file[7]

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
        The lesion belongs to the {current_match_class}{matching_subclass_suffix} category.
        """.strip()

    instruction = f"""
                You are a dermatologist. Your task is to describe the content in the uploaded, using medical terminology.\
                Be stick to what you identify, do not explain nature of diseases and do not imply. \
                Your reply should be within {maxlen} words. \
                Underline explicitly that the lesion includes a {current_class}{class_subclass_suffix}, {current_superclass} and some morphological characteristics are {current_description}.
                {specific_class_sentence}.
                Do not summarize at the end.
                """

    question_1 = f"""
            What Characteristics of the dermatoscopic structure of the skin lesions you identify, among:\
            1) Type of Lesion: The report might describe whether the lesion is benign (e.g., mole, freckle) or suspicious for malignancy (e.g., melanoma, basal cell carcinoma). \
            Common types of lesions include: \
            Macules: Flat spots on the skin. \
            Papules: Raised, solid bumps. \
            Nodules: Larger, deeper lumps. \
            Plaques: Raised, flat areas. \
            Ulcers: Open sores. \
            Cysts: Fluid-filled sacs. \
            Vesicles/Bullae: Fluid-filled blisters. \
            2) Color: The color (e.g., brown, red, black, or pink) of the lesion is described, as changes in color can be significant for malignancy. \
            3) Border Definition: Whether the lesion's edges are well-defined or irregular (which can suggest malignancy). \
            4) Symmetry: Whether the lesion is symmetric or asymmetric. Asymmetry may raise concern for malignancy. \
            """


    return instruction, question_1

def encode_pil(img):

    with BytesIO() as buffer:
        img.save(buffer, format = "JPEG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

cont = 0

np.random.shuffle(csv_file)
flag_overwrite = False
max_tokens = 1000

for i in tqdm(range(len(csv_file))):

    fname = str(csv_file[i,0])
    sample_path = get_filename(fname, csv_data)

    fname_img = IMG_FOLD + sample_path
    flag_exist_path = os.path.exists(fname_img)


    new_fname = folder_store_reports + fname + '.txt'

    if ((os.path.exists(new_fname) == False and flag_exist_path) or flag_overwrite):

        label = int(csv_file[i,1])

        base64_img = encode_image(fname_img)
        instructions, question = get_instruction_and_question(csv_file[i], maxlen)
        #img = Image.open(fname)
        #img_np = np.asarray(img)

        messages_to_send = []
        messages_to_send.append({'role': "system", 'content': instructions})
        messages_to_send.append({"role": "user",
                    "content": [
                        { "type": "text", "text": question },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}",
                                "detail": "low"
                            },
                        },
                    ],
                })


        completion = client.chat.completions.create(
            model=GPT_TO_USE,
            #model="gpt-4o",
            messages=messages_to_send,
            max_tokens = max_tokens,
            temperature = 0.2
        )

        note = completion.choices[0].message.content
        print(note)

        
        f = open(new_fname, "w")
        f.write(note)
        f.close() 

        cont = cont + 1

        if (cont % 20 == 0):

            time.sleep(0.5)