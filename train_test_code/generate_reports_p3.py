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
key_path = 'PATHKEY.txt'

with open(key_path, "r") as f:
    key = f.read().strip()
    #print(key)
key_check = check_api(key)
if key_check:
    print("The API key has been set successfully. NIH \n")
    hasKey = True
    client = OpenAI(api_key=key)


#DATASET = 'Derm7pt'
#DATASET = 'BCN20000'

CSV_FOLD = 'DATA_PATH/'
csv_test_filename = CSV_FOLD + 'classes_subclasses.csv'
csv_file = pd.read_csv(csv_test_filename, sep = ',', header = None, dtype=str).values

IMG_FOLD = 'IMG_PATH'

FOLD_NOTES = 'CLINICAL_NOTE_PATH'
os.makedirs(FOLD_NOTES, exist_ok = True)


gpt_4o_mini_fld = FOLD_NOTES + 'gpt_4_mini_as_doctor/'
os.makedirs(gpt_4o_mini_fld, exist_ok = True)

gpt_4o_fld = FOLD_NOTES + 'gpt_4o_fld_as_doctor/'
os.makedirs(gpt_4o_fld, exist_ok = True)


if (GPT_TO_USE == "gpt-4o-mini"):
    folder_store_reports = gpt_4o_mini_fld
elif (GPT_TO_USE == "gpt-4o"):
    folder_store_reports = gpt_4o_fld



def get_instruction_and_question(maxlen):

    instruction = f"""
                    You are a dermatologist. Your task is to describe the content in the image, using dermatologic terminology.\
                    You have to write a report, analyzing the characteristics of the image and reporting the type of skin lesion. \
                    Your reply should be within {maxlen} words. \
                    Underline explicitly that the image includes a lesion, its type, if it is benign or malignant. 
                    """
        
    question = f"""
            Report the characteristic of the images, considering:
            A) Asymmetry or symmetry of the lesion. \n
            B) The border of the lesion. \n
            C) Color of the lesion. \n
            D) Dermoscopic Structures of the lesion. \n
            E) Type of lesion, selecting only one of the following classes (and subclasses if possible): \n
            Benign keratosis (seborrheic keratosis, solar lentigo, lichen planus) \n
            Melanocytic nevus (nevus, blue nevus, clark nevus, spitz nevus, compound nevus, halo nevus) \n
            Dermatofibroma (dermatofibroma) \n
            Vascular lesion (vascular lesion, pyogenic granuloma)\n
            Actinic keratosis (actinic keratosis, bowen's disease) \n
            Basal cell cancer (basal cell cancer) \n
            Melanoma (melanoma) \n
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

    #print(os.path.exists(new_fname), flag_exist_path)

    if (os.path.exists(new_fname) == False and flag_exist_path):
        label = csv_file[i,1]

        #label_to_use = from_label_to_concept(label)
        base64_img = encode_image(fname_img)
        instructions, question = get_instruction_and_question(maxlen)
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
            temperature = 0.1
        )

        note = completion.choices[0].message.content
        print(note)

        
        f = open(new_fname, "w")
        f.write(note)
        f.close() 

        cont = cont + 1

        if (cont % 100 == 0):

            time.sleep(1)