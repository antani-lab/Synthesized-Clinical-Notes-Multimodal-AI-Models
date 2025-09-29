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
key_path = 'KEY_PATH.txt'

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
        concept = 'seborrheic'

    elif (label == 1):
        concept = 'dermatofibroma'

    elif (label == 2):
        concept = 'melanocytic nevus'

    elif (label == 3):
        concept = 'vascular lesion'

    elif (label == 4):
        concept = 'aktinic keratosis'

    elif (label == 5):
        concept = 'basal cell cancer'

    elif (label == 6):
        concept = 'melanoma'
    
    return concept

#DATASET = 'Derm7pt'
#DATASET = 'BCN20000'

CSV_FOLD = 'CSV_PATH'
csv_test_filename = CSV_FOLD + 'classes_subclasses.csv'
csv_file = pd.read_csv(csv_test_filename, sep = ',', header = None, dtype=str).values

IMG_FOLD = 'IMG_PATH'

FOLD_NOTES = 'NOTE_PATH'
os.makedirs(FOLD_NOTES, exist_ok = True)


gpt_4o_mini_fld = FOLD_NOTES + 'gpt_4o_mini_fld_abcd/'
os.makedirs(gpt_4o_mini_fld, exist_ok = True)

gpt_4o_fld = FOLD_NOTES + 'gpt_4o_fld_abcd/'
os.makedirs(gpt_4o_fld, exist_ok = True)

if (GPT_TO_USE == "gpt-4o-mini"):
    folder_store_reports = gpt_4o_mini_fld
elif (GPT_TO_USE == "gpt-4o"):
    folder_store_reports = gpt_4o_fld

#ABCD rule: https://dermoscopedia.org/ABCD_rule

def get_superclass(label):

    superclass = -1

    
    if ('seborrheic' in label):
        prob_pre = np.random.rand(1)[0]
        if (prob_pre >= 0.8):
            superclass = 'benign'
        else:
            superclass = 'benign (non-malignant)'

    elif ('dermatofibroma' in label):
        prob_pre = np.random.rand(1)[0]
        if (prob_pre >= 0.2):
            superclass = 'benign'
        else:
            superclass = 'benign (non-malignant)'

    elif ('nev' in label):
        prob_pre = np.random.rand(1)[0]
        if (prob_pre >= 0.2):
            superclass = 'benign'
        else:
            superclass = 'benign (non-malignant)'

    elif ('vascular' in label):
        prob_pre = np.random.rand(1)[0]
        if (prob_pre >= 0.2):
            superclass = 'benign'
        else:
            superclass = 'benign (non-malignant)'

    elif ('actinic' in label):
        superclass = 'pre-cancerous'

    elif ('basal' in label):
        prob_pre = np.random.rand(1)[0]
        if (prob_pre >= 0.2):
            superclass = 'malignant'
        else:
            superclass = 'cancerous'

    elif ('melanoma' in label):
        prob_pre = np.random.rand(1)[0]
        if (prob_pre >= 0.2):
            superclass = 'malignant'
        else:
            superclass = 'cancerous'
    
    return superclass

def get_instruction_and_question(label, subclass, superclass, maxlen):

    flag_subclass = label == subclass

    if (flag_subclass == True):
        instruction = f"""
                    You are a dermatologist. Your task is to describe the content in the image, using dermatologic terminology.\
                    You have to compile the structured report, using only the options provided among brackets. Report the category (Symmetric lesion, Border lesion, Color lesion, Dermoscopic structure). \
                    Your reply should be within {maxlen} words. \
                    Underline explicitly that the image is {superclass} and includes a {label}. Do not summarize at the end.
                    """
    else:
        instruction = f"""
                    You are a dermatologist. Your task is to describe the content in the image, using dermatologic terminology.\
                    You have to compile the structured report, using only the options provided among brackets. Do not report the name of the field, just the answer. \
                    Your reply should be within {maxlen} words. \
                    Underline explicitly that the image is {superclass} and includes a {label}, specifically {subclass}. Do not summarize at the end.
                    """
        
    question = f"""
            Report the following characteristics, choosing only the options among brackets, using the notation letter) (.e.g A), B)):
            A) Asymmetry (only for the skin lesion)
            Symmetric: Color and structure are mirrored across both axes.
            Asymmetric: Uneven color or structure \n

            B) Border (only for the skin lesion)
            Sharp border: Clear, abrupt pigment cut-off at the edge.
            Indistinct border: Gradual or blurry transition at lesion edge.
            Mixed border: Combination of sharp and indistinct segments. \n

            C) Color (considering the lesion only, not the skin)
            White (lighter than surrounding skin), Red, Light brown, Dark brown, Blue-gray, Black \n

            D) Dermoscopic Structures
            Structureless areas: Uniform zones lacking visible patterns.
            Pigment network: Reticular or mesh-like pigmentation.
            Branched streaks: Irregular, atypical branching lines.
            Dots: Small, round dark spots.
            Globules: Larger, clustered round pigmented areas. \n
    """

    return instruction, question

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
    
    sample_path = str(csv_file[i,0])
    sample_class = csv_file[i,1]
    sample_subclass = csv_file[i,2]

    fname_img = IMG_FOLD + sample_path + '.jpg'
    fname_path = sample_path

    flag_exist_path = os.path.exists(fname_img)

    if (flag_exist_path == False):

        fname_img = fname_img.replace('.jpg','.jpeg')
        flag_exist_path = os.path.exists(fname_img)

        if (os.path.exists(fname_img) == False):

            fname_img = fname_img.replace('.jpeg','.png')
            flag_exist_path = os.path.exists(fname_img)


    new_fname = folder_store_reports + fname_path + '.txt'

    if (os.path.exists(new_fname) == False and flag_exist_path):
        label = csv_file[i,1]

        #label_to_use = from_label_to_concept(label)
        superclass = get_superclass(sample_class)
        base64_img = encode_image(fname_img)
        instructions, question = get_instruction_and_question(sample_class, sample_subclass, superclass, maxlen)
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
            temperature = 0.2
        )

        note = completion.choices[0].message.content
        print(note)

        
        f = open(new_fname, "w")
        f.write(note)
        f.close() 

        cont = cont + 1

        if (cont % 200 == 0):

            time.sleep(1)