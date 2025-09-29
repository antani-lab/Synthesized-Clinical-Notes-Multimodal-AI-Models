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
from enum_multi import ALG, PHASE, TYPE_DATA, MOD, REPORTS
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
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-c', '--CNN', help='cnn_to_use',type=str, default='densenet121')
parser.add_argument('-e', '--EPOCHS', help='epochs to train',type=int, default=10)
parser.add_argument('-z', '--hidden_space', help='hidden_space_size',type=int, default=128)
parser.add_argument('-b', '--batch_size', help='batch size bag level',type=int, default=32)
parser.add_argument('-o', '--output_folder', help='path folder where to store output model',type=str, default='')
parser.add_argument('-i', '--DATA_FOLDER', help='path of the folder where to images are stored',type=str, default='')
parser.add_argument('-f', '--CSV_FOLDER', help='folder where csv including IDs and classes are stored',type=str, default='True')
parser.add_argument('-d', '--DATASET', help='dataset to use',type=str, default='HAM10000')
parser.add_argument('-w', '--PROBLEM', help='type of PROBLEM:binary (nevus vs melanoma), multiclass (sebo vs nevus vs basal vs melanoma)',type=str, default='multiclass')
parser.add_argument('-m', '--MODALITY', help='modalities to use: img, txt, multimodal',type=str, default='multimodal')
parser.add_argument('-t', '--TYPE', help='reports training: all, abcd, char, short, doc, meta, random',type=str, default='all')
parser.add_argument('-k', '--KEYWORDS', help='train on keywords',type=str, default='False')
parser.add_argument('-a', '--AUGMENTATION', help='report augmentation',type=str, default='False')

args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
CNN_TO_USE = args.CNN
BATCH_SIZE = args.batch_size
EPOCHS = args.EPOCHS
EPOCHS_str = EPOCHS
PROBLEM = args.PROBLEM
MODALITY = args.MODALITY

EMBEDDING_bool = True
DATASET = args.DATASET

hidden_space_len = args.hidden_space

REPORTS_TRAINING = args.TYPE
REPORT_AUGMENTATION = args.AUGMENTATION

if (REPORT_AUGMENTATION == 'True'):
	REPORT_AUGMENTATION = True
else:
	REPORT_AUGMENTATION = False

flag_KEYWORDS = args.KEYWORDS

if (flag_KEYWORDS == 'True'):
	flag_KEYWORDS = True
else:
	flag_KEYWORDS = False

seed = N_EXP
torch.manual_seed(seed)
#torch.use_deterministic_algorithms(mode=True)
if torch.cuda.is_available():
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

print("PARAMETERS")
print("N_EPOCHS: " + str(EPOCHS_str))
print("CNN used: " + str(CNN_TO_USE))
print("DATASET: " + str(DATASET))
print("N_EXP: " + str(N_EXP_str))
print("REPORTS: " + str(REPORTS))

####PATH where find patches
#instance_dir = args.DATA_FOLDER


#OUTPUT_folder = args.output_folder
print("CREATE DIRECTORY WHERE MODELS WILL BE STORED")


OUTPUT_folder = 'PLACEHOLDER_MODEL_WEIGHTS'
models_path = OUTPUT_folder
os.makedirs(models_path, exist_ok=True)
models_path = models_path + 'multimodal/'
os.makedirs(models_path, exist_ok=True)
models_path = models_path + MODALITY + '/'
os.makedirs(models_path, exist_ok=True)

models_path = models_path + REPORTS_TRAINING + '/'
os.makedirs(models_path, exist_ok=True)

if (flag_KEYWORDS):
	models_path = models_path + 'keywords/'
	os.makedirs(models_path, exist_ok=True)
else:
	models_path = models_path + 'no_keywords/'
	os.makedirs(models_path, exist_ok=True)

if (REPORT_AUGMENTATION):
	models_path = models_path + 'report_augmentation/'
	os.makedirs(models_path, exist_ok=True)
else:
	models_path = models_path + 'no_report_augmentation/'
	os.makedirs(models_path, exist_ok=True)

models_path = models_path + PROBLEM + '/'
os.makedirs(models_path, exist_ok=True)
models_path = models_path+CNN_TO_USE+'/'
os.makedirs(models_path, exist_ok=True)
models_path = models_path+'N_EXP_'+N_EXP_str+'/'
os.makedirs(models_path, exist_ok=True)


N_CLASSES = 5

flag_all = PHASE.all
MAIN_FLD = 'PLACEHOLDER'
MAIN_FOLDER = MAIN_FLD + '/datasets/'
CSV_FOLDER = MAIN_FLD + '/csv_folder/'
#all_data = valid_dataset
target_img = None

test_dataset = utils_data.get_specific_dataset(MAIN_FOLDER, DATASET, PHASE.all, None)

print(test_dataset.shape)

features_dir = MAIN_FOLDER+DATASET+'/features_'+CNN_TO_USE+'/'+MODALITY+'/'
os.makedirs(features_dir, exist_ok = True)

features_dir = features_dir + REPORTS_TRAINING + '/'
os.makedirs(features_dir, exist_ok=True)

if (flag_KEYWORDS):
	features_dir = features_dir + 'keywords/'
	os.makedirs(features_dir, exist_ok=True)
else:
	features_dir = features_dir + 'no_keywords/'
	os.makedirs(features_dir, exist_ok=True)

if (REPORT_AUGMENTATION):
	features_dir = features_dir + 'report_augmentation/'
	os.makedirs(features_dir, exist_ok=True)
else:
	features_dir = features_dir + 'no_report_augmentation/'
	os.makedirs(features_dir, exist_ok=True)

features_dir = features_dir+'/N_EXP_'+N_EXP_str+'/'
os.makedirs(features_dir, exist_ok = True)

feat_imgs_dir = features_dir + 'images/'
os.makedirs(feat_imgs_dir, exist_ok = True)

feat_prompts_dir = models_path + 'prompts/'
os.makedirs(feat_prompts_dir, exist_ok = True)

features_concept_path = feat_prompts_dir + '/cls_reports.npy'

try:
    with open(features_concept_path, 'rb') as f:
        concepts_embeddings = np.load(f)
    

except Exception as e:
    print(e)

#################LOAD



similarities_images = np.empty((len(test_dataset), concepts_embeddings.shape[0] + 1), dtype = 'object')

MODALITY = MOD[MODALITY]

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
filename_to_store = feat_imgs_dir + 'feature_similarities.csv'

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
