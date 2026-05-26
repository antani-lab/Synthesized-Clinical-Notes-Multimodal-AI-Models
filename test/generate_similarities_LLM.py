import sys, getopt
import torch
from torch.utils import data
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
from enum_multi import ALG, PHASE, TYPE_DATA, MOD, REPORTS, CONCEPTS
import utils_data
import random
from dataloader import ImbalancedDatasetSampler, Dataset_generate_features, filter_labels
from model import MultimodalArchitecture, MultimodalArchitecture_CLIP
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
from metrics_multiclass import accuracy_score, kappa_score, f1_scores, precisions, recalls
from tqdm import tqdm
import utils_txt
from sklearn.metrics.pairwise import cosine_similarity
from numba import jit, float32

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
parser.add_argument('-l', '--LLM', help='LLM to use: BioMedClip, MONET, Derm1M, MedImgInsights: ',type=str, default='BioMedClip')
parser.add_argument('-b', '--batch_size', help='batch size bag level',type=int, default=32)
parser.add_argument('-d', '--DATASET', help='dataset to use',type=str, default='HAM10000')
parser.add_argument('-i', '--CONCEPTS', help='concepts to analyze: class, subclass, class_matching, subclass_matching',type=str, default='classes')
parser.add_argument('-m', '--MAIN_FOLDER', help='path to main folder including image and csv folders',type=str, default='')

args = parser.parse_args()
LLM_TO_USE = args.LLM

BATCH_SIZE = args.batch_size
DATASET = args.DATASET

CONCEPT_TO_USE_str = args.CONCEPTS
CONCEPT_type = CONCEPTS[CONCEPT_TO_USE_str]


seed = 0
torch.manual_seed(seed)
#torch.use_deterministic_algorithms(mode=True)
if torch.cuda.is_available():
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

print("PARAMETERS")
print("DATASET: " + str(DATASET))

####PATH where find patches
#instance_dir = args.DATA_FOLDER


#OUTPUT_folder = args.output_folder
print("CREATE DIRECTORY WHERE MODELS WILL BE STORED")

MAIN_FOLDER = args.MAIN_FOLDER

DATA_FOLDER = MAIN_FOLDER + '/datasets/'
CSV_FOLDER = MAIN_FOLDER + '/csv_folder/'
#all_data = valid_dataset
target_img = None

features_dir = MAIN_FOLDER+DATASET+'/features_'+LLM_TO_USE+'/'
feat_imgs_dir = features_dir + 'images/'
os.makedirs(feat_imgs_dir, exist_ok = True)
feat_prompts_dir = features_dir + 'prompts/'
os.makedirs(feat_prompts_dir, exist_ok = True)

test_dataset_filename = CSV_FOLDER + DATASET + "/labels.csv"
test_dataset = pd.read_csv(test_dataset_filename, sep = ',', header = None).values


if (CONCEPT_type is CONCEPTS.classes):

	features_concept_path = feat_prompts_dir + '/cls_reports_classes.npy'

	try:
		with open(features_concept_path, 'rb') as f:
			concepts_embeddings = np.load(f)
	except Exception as e:
		print(e)

elif (CONCEPT_type is CONCEPTS.subclasses):

	features_concept_path = feat_prompts_dir + '/cls_reports_subclasses.npy'

	try:
		with open(features_concept_path, 'rb') as f:
			concepts_embeddings = np.load(f)
	except Exception as e:
		print(e)

elif (CONCEPT_type is CONCEPTS.classes_matching):

	features_concept_path = feat_prompts_dir + '/cls_reports_classes_matching.npy'

	try:
		with open(features_concept_path, 'rb') as f:
			concepts_embeddings = np.load(f)
	except Exception as e:
		print(e)

elif (CONCEPT_type is CONCEPTS.subclasses_matching):

	features_concept_path = feat_prompts_dir + '/cls_reports_subclasses_matching.npy'

	try:
		with open(features_concept_path, 'rb') as f:
			concepts_embeddings = np.load(f)
	except Exception as e:
		print(e)


#################LOAD



similarities_images = np.empty((len(test_dataset), concepts_embeddings.shape[0] + 1), dtype = 'object')

@jit(nopython=True)
def cosine_similarity_numba(u: np.ndarray, v: np.ndarray):
    assert u.shape[0] == v.shape[0]

    uv = 0.0
    uu = 0.0
    vv = 0.0

    for i in range(u.shape[0]):
        uv += u[i] * v[i]
        uu += u[i] * u[i]
        vv += v[i] * v[i]

    if uu != 0.0 and vv != 0.0:
        return uv / np.sqrt(uu * vv)
    else:
        return 1.0

for i in tqdm(range(len(test_dataset))):
	
	ID_img = test_dataset[i,0]
	
	fname = ID_img.split('/')[-1]

	fname = fname.split('.')[0]

	features_filename_img = feat_imgs_dir + fname + '.npy'

	try:
		with open(features_filename_img, 'rb') as f:
			cls_img = np.load(f)
		
	except Exception as e:
		print(e)
	
	row = np.empty(concepts_embeddings.shape[0] + 1, dtype='object')  # Preallocate array
	row[0] = fname

	for e, c in enumerate(concepts_embeddings):
		#print(cls_img.dtype, c.dtype)
		#sim = cosine_similarity(cls_img.reshape(1, -1), c.reshape(1, -1)).item() 
		sim = cosine_similarity_numba(cls_img.astype(np.float32), c.astype(np.float32))
		#print(sim)
		row[e+1] = sim

	similarities_images[i] = row


similarities_images = np.reshape(similarities_images, (len(test_dataset), concepts_embeddings.shape[0] + 1))
print(similarities_images)

if (CONCEPT_type is CONCEPTS.classes):
	filename_to_store = feat_imgs_dir + 'feature_similarities_keyword_classes.csv'

elif (CONCEPT_type is CONCEPTS.subclasses):
	filename_to_store = feat_imgs_dir + 'feature_similarities_keyword_subclasses.csv'

elif (CONCEPT_type is CONCEPTS.classes_matching):
	filename_to_store = feat_imgs_dir + 'feature_similarities_keyword_classes_matching.csv'

elif (CONCEPT_type is CONCEPTS.subclasses_matching):
	filename_to_store = feat_imgs_dir + 'feature_similarities_keyword_subclasses_matching.csv'

# Creating column names dynamically
columns = ['filenames'] + [f'class_{i}' for i in range(concepts_embeddings.shape[0])]

# Constructing the data dictionary
File = {'filenames': similarities_images[:, 0]}
for i in range(1, concepts_embeddings.shape[0] + 1):
	File[f'class_{i - 1}'] = similarities_images[:, i]

# Creating the DataFrame
df = pd.DataFrame(File, columns=columns)

fmt = ['%s'] + ['%.4f'] * (similarities_images.shape[1] - 1)

np.savetxt(filename_to_store, df.values, fmt='%s',delimiter=',')
