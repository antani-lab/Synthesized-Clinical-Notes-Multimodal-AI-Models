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
from enum_multi import DATASET_TO_USE

parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-d', '--DATASET', help='application of DDCA',type=str, default='BCN20000')
parser.add_argument('-g', '--GPT', help='GPT to use',type=str, default='gpt-4o')

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
maxlen = 250

#load key
key_path = 'PATH_KEY.txt'

with open(key_path, "r") as f:
    key = f.read().strip()
    #print(key)
key_check = check_api(key)
if key_check:
    print("The API key has been set successfully. NIH \n")
    hasKey = True
    client = OpenAI(api_key=key)

def from_label_to_concept(label):

    concept = 'NONE'

    if (label == 0):
        concept = 'seborrheic keratosis'

    elif (label == 1):
        concept = 'dermatofibroma'

    elif (label == 2):
        concept = 'melanocytic nevus'

    elif (label == 3):
        concept = 'vascular lesion'

    elif (label == 4):
        concept = 'actinic keratosis'

    elif (label == 5):
        concept = 'basal cell cancer'

    elif (label == 6):
        concept = 'melanoma'
    
    return concept

#DATASET = 'Derm7pt'
#DATASET = 'BCN20000'

MAIN_FOLD = 'IMG_FLD'

csv_test_filename = MAIN_FOLD + 'labels.csv'
csv_file = pd.read_csv(csv_test_filename, sep = ',', header = None, dtype=str).values


FOLD_NOTES = 'CLINICAL_NOTE_FLD'
os.makedirs(FOLD_NOTES, exist_ok = True)


gpt_4o_mini_fld = FOLD_NOTES + 'gpt_4o_mini_fld/'
os.makedirs(gpt_4o_mini_fld, exist_ok = True)

gpt_4o_fld = FOLD_NOTES + 'gpt_4o_fld/'
os.makedirs(gpt_4o_fld, exist_ok = True)

if (GPT_TO_USE == "gpt-4o-mini"):
    folder_store_reports = gpt_4o_mini_fld
elif (GPT_TO_USE == "gpt-4o"):
    folder_store_reports = gpt_4o_fld

def get_instruction_and_question(label, maxlen):

    instruction = f"""
                You are a dermatologist. Your task is to describe the content in the uploaded, using medical terminology.\
                Be stick to what you identify, do not explain nature of diseases and do not imply. \
                Your reply should be within {maxlen} words. \
                Underline explicitly that the image includes a {label}.
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

for i in tqdm(range(len(csv_file))):

    fname = MAIN_FOLD + csv_file[i,0]
    fname_path = csv_file[i,0].replace('.jpg','')
    fname_path = fname_path.replace('.jpeg','')
    fname_path = fname_path.replace('.png','')

    new_fname = folder_store_reports + fname_path + '.txt'

    if (os.path.exists(new_fname) == False):
        label = csv_file[i,1]

        label_to_use = from_label_to_concept(label)

        base64_img = encode_image(fname)
        instructions, question = get_instruction_and_question(label_to_use, maxlen)
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
            max_tokens = 200,
            temperature = 0.3
        )

        note = completion.choices[0].message.content
        print(note)

        
        f = open(new_fname, "w")
        f.write(note)
        f.close() 

        cont = cont + 1

        if (cont % 200 == 0):

            time.sleep(1)