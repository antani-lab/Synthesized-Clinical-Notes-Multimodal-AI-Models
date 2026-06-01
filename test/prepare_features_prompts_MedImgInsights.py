import torch
from transformers import AutoModel, AutoTokenizer
import os
import sys
from huggingface_hub import snapshot_download
import pandas as pd
import sklearn
from PIL import Image
import numpy as np
MedImgInsights_FOLDER = "../MedImageInsights/"
sys.path.append(MedImgInsights_FOLDER)

from medimageinsightmodel import MedImageInsight
import base64

import sys
import argparse
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
import sys
import random
sys.path.append("../utils/")
sys.path.append("../models/")

import utils_zero_shot_learning
from metrics_multiclass import accuracy_score, kappa_score, f1_scores, precisions, recalls

argv = sys.argv[1:]

print("CUDA current device " + str(torch.cuda.current_device()))
print("CUDA devices available " + str(torch.cuda.device_count()))

if torch.cuda.is_available():
	device = torch.device("cuda")
	print("working on gpu")
else:
	device = torch.device("cpu")
	print("working on cpu")
print(torch.backends.cudnn.version())
torch.backends.cudnn.benchmark = False

#algorithm parameters

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-b', '--batch_size', help='batch size bag level',type=int, default=128)
parser.add_argument('-d', '--DATASET', help='dataset to use',type=str, default='HAM10000')
parser.add_argument('-l', '--LLM', help='LLM to use: BioMedClip, MONET, Derm1M, MedImageInsights',type=str, default='MedImageInsights')
parser.add_argument('-f', '--DATA_FOLDER', help='path of the folder where to images are stored',type=str, default='')

args = parser.parse_args()

BATCH_SIZE = args.batch_size

EMBEDDING_bool = True
DATASET = args.DATASET
LLM_TO_USE = args.LLM

N_EXP = 0
seed = N_EXP
torch.manual_seed(seed)
#torch.use_deterministic_algorithms(mode=True)
if torch.cuda.is_available():
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

print("PARAMETERS")
print("DATASET: " + str(DATASET))

#OUTPUT_folder = args.output_folder
print("CREATE DIRECTORY WHERE MODELS WILL BE STORED")

MAIN_FOLDER = args.DATA_FOLDER
DATA_FOLDER = MAIN_FOLDER + '/datasets/'
CSV_FOLDER = MAIN_FOLDER + 'csv_folder/'

zero_shot_learning_fld = CSV_FOLDER + 'zero_shot_keywords/'

filename_keyword_classes = zero_shot_learning_fld + "keyword_classes.csv"
filename_keyword_subclasses = zero_shot_learning_fld + "keyword_subclasses.csv"
filename_keyword_classes_matching = zero_shot_learning_fld + "keyword_classes_matching.csv"
filename_keyword_subclasses_matching = zero_shot_learning_fld + "keyword_subclasses_matching.csv"

keyword_classes = pd.read_csv(filename_keyword_classes, sep = ',', header = None).values.tolist()
keyword_subclasses = pd.read_csv(filename_keyword_subclasses, sep = ',', header = None).values.tolist()
keyword_classes_matching = pd.read_csv(filename_keyword_classes_matching, sep = ',', header = None).values.tolist()
keyword_subclasses_matching = pd.read_csv(filename_keyword_subclasses_matching, sep = ',', header = None).values.tolist()

N_CLASSES = 8


features_dir = MAIN_FOLDER+DATASET+'/features_'+LLM_TO_USE+'/prompts/'
os.makedirs(features_dir, exist_ok=True)


list_terminology = utils_zero_shot_learning.list_terminology

#"""
#MODEL DEFINITION
#CNN BACKBONE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


repo_id = "lion-ai/MedImageInsights"
local_dir = MedImgInsights_FOLDER
# This will create a local folder `MedImageInsights` with all files, including large LFS files.
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # set to True if you're OK with symlinks
)

model_dir = MedImgInsights_FOLDER + "/2024.09.27/"

classifier = MedImageInsight(
    model_dir=model_dir,
    vision_model_name="medimageinsigt-v1.0.0.pt",
    language_model_name="language_model.pth",
)
classifier.device = device

print(classifier.device)
# Load model
classifier.load_model()

def read_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()


hidden_dim = 1024

keyword_classes = pd.read_csv(filename_keyword_classes, sep = ',', header = None).values.tolist()

features = []
filenames = []

for i in range(len(keyword_classes)):
	
	with torch.autocast(device_type='cuda', dtype=torch.float16):


		#keyword = 'the image suggests for melanoma, possibly malignat'
		keyword = keyword_classes[i][0]
		filenames.append(keyword)
		print(i, keyword)

		text_tokens = classifier.tokenize(
			keyword,
			padding='max_length',
			max_length=classifier.max_length,
			truncation=True,
			return_tensors='pt'
		)
		text_tokens = {k: v.to(classifier.device) for k, v in text_tokens.items()}

		with torch.no_grad():
			cls_np = classifier.model.encode_text(text_tokens).detach().cpu().numpy()


	try:
		cls_np = np.squeeze(cls_np)
	except:
		pass

	features.append(cls_np)

features = np.reshape(features, (len(keyword_classes), hidden_dim))

#print(filenames)

#save features
features_filename = features_dir + 'cls_reports_classes.npy'
with open(features_filename, 'wb') as f:
	np.save(f, features)

#save list
features_csv_filename = features_dir + 'list_keywords_classes.csv'

File = {'keywords':filenames}
df = pd.DataFrame(File,columns=['keywords'])

df.to_csv(features_csv_filename, index=False, header=False)


keyword_subclasses = pd.read_csv(filename_keyword_subclasses, sep = ',', header = None).values.tolist()

filenames = []
features = []

for i in range(len(keyword_subclasses)):
	
	with torch.autocast(device_type='cuda', dtype=torch.float16):


		#keyword = 'the image suggests for melanoma, possibly malignat'
		keyword = keyword_subclasses[i][0]
		filenames.append(keyword)
		print(i, keyword)

		text_tokens = classifier.tokenize(
			keyword,
			padding='max_length',
			max_length=classifier.max_length,
			truncation=True,
			return_tensors='pt'
		)
		text_tokens = {k: v.to(classifier.device) for k, v in text_tokens.items()}

		with torch.no_grad():
			cls_np = classifier.model.encode_text(text_tokens).detach().cpu().numpy()


	try:
		cls_np = np.squeeze(cls_np)
	except:
		pass

	features.append(cls_np)

features = np.reshape(features, (len(keyword_subclasses), hidden_dim))


#save features
features_filename = features_dir + 'cls_reports_subclasses.npy'
with open(features_filename, 'wb') as f:
	np.save(f, features)

#save list
features_csv_filename = features_dir + 'list_keywords_subclasses.csv'

File = {'keywords':filenames}
df = pd.DataFrame(File,columns=['keywords'])

df.to_csv(features_csv_filename, index=False, header=False)

keyword_classes_matching = pd.read_csv(filename_keyword_classes_matching, sep = ',', header = None).values.tolist()

filenames = []
features = []

for i in range(len(keyword_classes_matching)):
	
	with torch.autocast(device_type='cuda', dtype=torch.float16):


		#keyword = 'the image suggests for melanoma, possibly malignat'
		keyword = keyword_classes_matching[i][0]
		filenames.append(keyword)
		print(i, keyword)

		text_tokens = classifier.tokenize(
			keyword,
			padding='max_length',
			max_length=classifier.max_length,
			truncation=True,
			return_tensors='pt'
		)
		text_tokens = {k: v.to(classifier.device) for k, v in text_tokens.items()}

		with torch.no_grad():
			cls_np = classifier.model.encode_text(text_tokens).detach().cpu().numpy()


	try:
		cls_np = np.squeeze(cls_np)
	except:
		pass

	features.append(cls_np)

features = np.reshape(features, (len(keyword_classes_matching), hidden_dim))


#save features
features_filename = features_dir + 'cls_reports_classes_matching.npy'
with open(features_filename, 'wb') as f:
	np.save(f, features)

#save list
features_csv_filename = features_dir + 'list_keywords_classes_matching.csv'

File = {'keywords':filenames}
df = pd.DataFrame(File,columns=['keywords'])

df.to_csv(features_csv_filename, index=False, header=False)


keyword_subclasses_matching = pd.read_csv(filename_keyword_subclasses_matching, sep = ',', header = None).values.tolist()

filenames = []
features = []

for i in range(len(keyword_subclasses_matching)):
	
	with torch.autocast(device_type='cuda', dtype=torch.float16):


		#keyword = 'the image suggests for melanoma, possibly malignat'
		keyword = keyword_subclasses_matching[i][0]
		filenames.append(keyword)
		print(i, keyword)

		text_tokens = classifier.tokenize(
			keyword,
			padding='max_length',
			max_length=classifier.max_length,
			truncation=True,
			return_tensors='pt'
		)
		text_tokens = {k: v.to(classifier.device) for k, v in text_tokens.items()}

		with torch.no_grad():
			cls_np = classifier.model.encode_text(text_tokens).detach().cpu().numpy()


	try:
		cls_np = np.squeeze(cls_np)
	except:
		pass

	features.append(cls_np)

features = np.reshape(features, (len(keyword_subclasses_matching), hidden_dim))


#save features
features_filename = features_dir + 'cls_reports_subclasses_matching.npy'
with open(features_filename, 'wb') as f:
	np.save(f, features)

#save list
features_csv_filename = features_dir + 'list_keywords_subclasses_matching.csv'

File = {'keywords':filenames}
df = pd.DataFrame(File,columns=['keywords'])

df.to_csv(features_csv_filename, index=False, header=False)
