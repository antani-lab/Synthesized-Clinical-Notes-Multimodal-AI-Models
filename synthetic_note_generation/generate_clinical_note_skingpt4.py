import os

os.environ["HF_TOKEN"] = token

from huggingface_hub import snapshot_download
import torch
import sys
import json
import pandas as pd
from PIL import Image
import argparse
from tqdm import tqdm
import time
import numpy as np

import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
from types import SimpleNamespace

import yaml
# Add repo to path

repo_path = "../SkinGPT-4-main/"
sys.path.append(repo_path)

import sys

from skingpt4.models import skin_gpt4
from PIL import Image
from skingpt4.models.skin_gpt4 import skingpt4

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from skingpt4.common.config import Config
from skingpt4.common.dist_utils import get_rank
from skingpt4.common.registry import registry
from skingpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from skingpt4.datasets.builders import *
from skingpt4.models import *
from skingpt4.processors import *
from skingpt4.runners import *
from skingpt4.tasks import *
from omegaconf import OmegaConf
from enum_multi import DATASET_TO_USE
import numpy as np
import random
import textwrap

if torch.cuda.is_available():
	device = torch.device("cuda")
	print("working on gpu")
else:
	device = torch.device("cpu")
	print("working on cpu")
print(torch.backends.cudnn.version())

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
seed = 0
torch.manual_seed(seed)
#torch.use_deterministic_algorithms(mode=True)
if torch.cuda.is_available():
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
#np.random.seed(seed)
random.seed(seed)


parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-d', '--DATASET', help='application of DDCA',type=str, default='BCN20000')
parser.add_argument('-c', '--CSV_FOLDER', help='path of csv folder including metadata (including the dataset folder)',type=str, default='')
parser.add_argument('-i', '--INPUT_PATH', help='path of the folder where to images are stored (including the dataset folder)',type=str, default='')
parser.add_argument('-o', '--OUTPUT_PATH', help='path folder where to store output model (including the dataset folder)',type=str, default='')
parser.add_argument('-m', '--MEMORY', help='patch of huggingface cache',type=str, default='')

args = parser.parse_args()

HUGGINGFACE_CACHE = args.MEMORY
os.environ["HF_HOME"] = HUGGINGFACE_CACHE
os.environ["TRANSFORMERS_CACHE"] = HUGGINGFACE_CACHE
os.environ["HF_DATASETS_CACHE"] = HUGGINGFACE_CACHE + "/datasets"
os.environ["HF_HUB_CACHE"] = HUGGINGFACE_CACHE + "/hub"

token_fname = HUGGINGFACE_CACHE + "/token.txt"

with open(token_fname, 'r', encoding='utf-8', errors='ignore') as file:
    #with open(ID_txt, 'r') as file:
    token = file.read()
    file.close()

os.environ["HF_TOKEN"] = token

DATASET = args.DATASET
DATASET_IN_USE = DATASET_TO_USE[DATASET]

hasKey = False
maxlen = 300

#DATASET = 'Derm7pt'
#DATASET = 'BCN20000'


CSV_FOLD = args.CSV_FOLDER+DATASET+'/'
csv_test_filename = CSV_FOLD + 'classes_subclasses_metadata_mapping.csv'
csv_file = pd.read_csv(csv_test_filename, sep = ',', header = None, dtype=str).values

labels_filename = CSV_FOLD + "labels.csv"
csv_data = pd.read_csv(labels_filename, sep = ',', header = None, dtype=str).values

IMG_FOLD = args.INPUT_PATH + DATASET + '/resized_images/'

FOLD_NOTES = args.OUTPUT_PATH + DATASET + '/clinical_notes/'
os.makedirs(FOLD_NOTES, exist_ok = True)



folder_store_reports_abcd = FOLD_NOTES + 'skingpt4_abcd/'
os.makedirs(folder_store_reports_abcd, exist_ok = True)

folder_store_reports_char = FOLD_NOTES + 'skingpt4_char/'
os.makedirs(folder_store_reports_char, exist_ok = True)

folder_store_reports_doc = FOLD_NOTES + 'skingpt4_doc/'
os.makedirs(folder_store_reports_doc, exist_ok = True)

folder_store_reports_p1 = FOLD_NOTES + 'skingpt4_p1/'
os.makedirs(folder_store_reports_p1, exist_ok = True)

folder_store_reports_p2 = FOLD_NOTES + 'skingpt4_p2/'
os.makedirs(folder_store_reports_p2, exist_ok = True)

cont = 0
np.random.shuffle(csv_file)
np.random.seed(seed)
random.seed(seed)



MODEL_PATH = repo_path

cfg_path = MODEL_PATH + "skingpt4/configs/models/skingpt4_llama2_13bchat.yaml"
#cfg_path = MODEL_PATH + "skingpt4/configs/models/skingpt4_vicuna.yaml"

cfg = OmegaConf.load(cfg_path)

print(cfg)   # sanity check: should show model.arch, model.ckpt, datasets, etc.

arch = cfg.model.arch

# Use the repo’s registry
model_cls = registry.get_model_class(arch)

model = model_cls.from_config(cfg.model)   # this will also read llama_model path from the model config yaml

model = model.float().to("cuda").eval()


# Plain dict, not OmegaConf.create
vis_proc_cfg = {"name": "blip2_image_eval", "image_size": 224}

proc_cls = registry.get_processor_class(vis_proc_cfg["name"])
vis_processor = proc_cls.from_config(vis_proc_cfg)

chat = Chat(model, vis_processor, device="cuda")


cont = 0


def get_instruction_and_question_abcd(metadata, maxlen):

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
        The lesion falls within the {current_match_class}{matching_subclass_suffix} spectrum, representing a pattern consistent with this diagnostic group.
        """.strip()


    instruction = textwrap.dedent(f"""
                You are a dermatologist. Your task is to describe the content in the image, using dermatologic terminology.\
                You have to compile the structured report, using only the options provided among brackets. Report the category (Symmetric lesion, Border lesion, Color lesion, Dermoscopic structure). \
                Your reply should be within {maxlen} words. \
                Underline explicitly that the lesion is {current_superclass} and includes a {current_class}{class_subclass_suffix} and it includes {current_description}.
                {specific_class_sentence}.

                Do not summarize at the end.
                """)
    
        
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


def get_instruction_and_question_char(metadata, maxlen):

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

    question = f"""
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


    return instruction, question


def get_instruction_and_question_doc(maxlen):

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
            Squamous Cell Cancer \n
    """

    return instruction, question


def get_instruction_original(maxlen, metadata = None):
    possible_options = [
        "Describe this image in detail.",
        "Take a look at this image and describe what you notice.",
        "Please provide a detailed description of the picture.",
        "Could you describe the contents of this image for me?"
    ]

    prompt = random.choice(possible_options)

    if (metadata is not None):

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
                    Underline explicitly that the lesion includes a {current_class}{class_subclass_suffix}, {current_superclass} and some morphological characteristics are {current_description}.
                    {specific_class_sentence}.
                    """.strip()

        prompt = prompt + instruction

    return prompt

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


flag_overwrite = False
max_tokens = 1000
cont = 0

num_beams = 1
temperature = 0.2
max_new_tokens = max_tokens

for i in tqdm(range(len(csv_file))):
    
    sample_path = str(csv_file[i,0])
    sample_class = csv_file[i,1]
    sample_subclass = csv_file[i,2]

    fname = str(csv_file[i,0])
    sample_path = get_filename(fname, csv_data)

    fname_img = IMG_FOLD + sample_path
    flag_exist_path = os.path.exists(fname_img)

    new_fname_abcd = folder_store_reports_abcd + fname + '.txt'

    if (flag_exist_path):

        #abcd
        if (os.path.exists(new_fname_abcd) == False or flag_overwrite):
            print("abcd")
            

            #label_to_use = from_label_to_concept(label)

            instructions, question = get_instruction_and_question_abcd(csv_file[i], maxlen)
            
            prompt = instructions + question
            print(prompt)
            raw_image = Image.open(fname_img).convert("RGB")

            chat_state = None
            img_list = []

            #image_tensor = vis_processor(raw_image).unsqueeze(0).to("cuda")#.half()

            chat_state = CONV_VISION.copy()
            img_list = []   # ✅ manually put the tensor in the list

            # 2. Upload image (PIL image!)
            llm_message = chat.upload_img(raw_image, chat_state, img_list)

            # 3. Ask a question (note: this updates chat_state in-place)
            chat.ask(prompt, chat_state)

            note = chat.answer(conv=chat_state, 
                                img_list=img_list,
                                num_beams=num_beams,
                                temperature=temperature,
                                max_new_tokens=max_tokens,
                                )

            
            print(note)

            f = open(new_fname_abcd, "w")
            f.write(note[0])
            f.close() 

        new_fname_char = folder_store_reports_char + fname + '.txt'
        #char
        if (os.path.exists(new_fname_char) == False or flag_overwrite):
            print("char")
            label = csv_file[i,1]

            #label_to_use = from_label_to_concept(label)

            instructions, question = get_instruction_and_question_char(csv_file[i], maxlen)
            
            prompt = instructions + question

            raw_image = Image.open(fname_img).convert("RGB")

            chat_state = None
            img_list = []

            #image_tensor = vis_processor(raw_image).unsqueeze(0).to("cuda")#.half()

            chat_state = CONV_VISION.copy()
            img_list = []   # ✅ manually put the tensor in the list

            # 2. Upload image (PIL image!)
            llm_message = chat.upload_img(raw_image, chat_state, img_list)

            # 3. Ask a question (note: this updates chat_state in-place)
            chat.ask(prompt, chat_state)

            note = chat.answer(conv=chat_state, 
                                img_list=img_list,
                                num_beams=num_beams,
                                temperature=temperature,
                                max_new_tokens=max_tokens,
                                )[0]
            print(note)

            f = open(new_fname_char, "w")
            f.write(note)
            f.close() 
       

        new_fname_doc = folder_store_reports_doc + fname + '.txt'
        #char
        if (os.path.exists(new_fname_doc) == False or flag_overwrite):
            print("doc")
            label = csv_file[i,1]

            #label_to_use = from_label_to_concept(label)

            img_paths = [fname_img]

            instructions, question = get_instruction_and_question_doc(maxlen)
            
            prompt = instructions + question

            raw_image = Image.open(fname_img).convert("RGB")

            chat_state = None
            img_list = []

            #image_tensor = vis_processor(raw_image).unsqueeze(0).to("cuda")#.half()

            chat_state = CONV_VISION.copy()
            img_list = []   # ✅ manually put the tensor in the list

            # 2. Upload image (PIL image!)
            llm_message = chat.upload_img(raw_image, chat_state, img_list)

            # 3. Ask a question (note: this updates chat_state in-place)
            chat.ask(prompt, chat_state)

            note = chat.answer(conv=chat_state, 
                                img_list=img_list,
                                num_beams=num_beams,
                                temperature=temperature,
                                max_new_tokens=max_tokens,
                                )[0]
            print(note)
            
            f = open(new_fname_doc, "w")
            f.write(note)
            f.close() 
        

        new_fname_p1 = folder_store_reports_p1 + fname + '.txt'
        #char
        if (os.path.exists(new_fname_p1) == False or flag_overwrite):
            print("p1")
            label = csv_file[i,1]

            #label_to_use = from_label_to_concept(label)

            img_paths = [fname_img]

            prompt = get_instruction_original(maxlen, csv_file[i])
            
            raw_image = Image.open(fname_img).convert("RGB")

            chat_state = None
            img_list = []

            #image_tensor = vis_processor(raw_image).unsqueeze(0).to("cuda")#.half()

            chat_state = CONV_VISION.copy()
            img_list = []   # ✅ manually put the tensor in the list

            # 2. Upload image (PIL image!)
            llm_message = chat.upload_img(raw_image, chat_state, img_list)

            # 3. Ask a question (note: this updates chat_state in-place)
            chat.ask(prompt, chat_state)

            note = chat.answer(conv=chat_state, 
                                img_list=img_list,
                                num_beams=num_beams,
                                temperature=temperature,
                                max_new_tokens=max_tokens,
                                )[0]
            print(note)
            
            f = open(new_fname_p1, "w")
            f.write(note)
            f.close() 
       

        new_fname_p2 = folder_store_reports_p2 + fname + '.txt'
        #char
        if (os.path.exists(new_fname_p2) == False or flag_overwrite):
            print("p2")
            label = csv_file[i,1]

            #label_to_use = from_label_to_concept(label)

            img_paths = [fname_img]

            prompt = get_instruction_original(maxlen, None)
            
            raw_image = Image.open(fname_img).convert("RGB")

            chat_state = None
            img_list = []

            #image_tensor = vis_processor(raw_image).unsqueeze(0).to("cuda")#.half()

            chat_state = CONV_VISION.copy()
            img_list = []   # ✅ manually put the tensor in the list

            # 2. Upload image (PIL image!)
            llm_message = chat.upload_img(raw_image, chat_state, img_list)

            # 3. Ask a question (note: this updates chat_state in-place)
            chat.ask(prompt, chat_state)

            note = chat.answer(conv=chat_state, 
                                img_list=img_list,
                                num_beams=num_beams,
                                temperature=temperature,
                                max_new_tokens=max_tokens,
                                )[0]
            print(note)
            
            f = open(new_fname_p2, "w")
            f.write(note)
            f.close() 

        cont = cont + 1

        if (cont % 20 == 0):

            time.sleep(0.5)