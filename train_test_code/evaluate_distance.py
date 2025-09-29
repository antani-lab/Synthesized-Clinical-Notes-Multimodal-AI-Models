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
from enum_multi import ALG, PHASE, TYPE_DATA, MOD
import utils_data
import random
from dataloader import ImbalancedDatasetSampler, Dataset_instance, filter_labels
from model import MultimodalArchitecture
import json
from metrics_multiclass import accuracy_score, kappa_score, f1_scores, precisions, recalls
from tqdm import tqdm
import sklearn
import utils_zero_shot_learning
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
parser.add_argument('-b', '--batch_size', help='batch size bag level',type=int, default=128)
parser.add_argument('-o', '--output_folder', help='path folder where to store output model',type=str, default='')
parser.add_argument('-i', '--DATA_FOLDER', help='path of the folder where to images are stored',type=str, default='')
parser.add_argument('-f', '--CSV_FOLDER', help='folder where csv including IDs and classes are stored',type=str, default='True')
parser.add_argument('-d', '--DATASET', help='dataset to use',type=str, default='HAM10000')
parser.add_argument('-w', '--PROBLEM', help='type of PROBLEM:binary (nevus vs melanoma), multiclass (sebo vs nevus vs basal vs melanoma)',type=str, default='binary')
parser.add_argument('-m', '--MODALITY', help='modalities to use: img, txt, multimodal',type=str, default='img')
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

OUTPUT_MODEL = '/PATH/MODEL/'
MODEL_FLD = OUTPUT_MODEL + MODALITY + '/'

MODEL_FLD = MODEL_FLD + REPORTS_TRAINING + '/'

if (flag_KEYWORDS):
	MODEL_FLD = MODEL_FLD + 'keywords/'
else:
	MODEL_FLD = MODEL_FLD + 'no_keywords/'

if (REPORT_AUGMENTATION):
	MODEL_FLD = MODEL_FLD + 'report_augmentation/'
else:
	MODEL_FLD = MODEL_FLD + 'no_report_augmentation/'

MODEL_FLD = MODEL_FLD + PROBLEM + '/' +CNN_TO_USE+'/'+'N_EXP_'+N_EXP_str+'/'
MODEL_FLD = MODEL_FLD+'checkpoints/test/distance/'
os.makedirs(MODEL_FLD, exist_ok = True)

print("DATASET: " + str(DATASET))
print("N_EXP: " + str(N_EXP_str))
print("MODALITY: " + str(MODALITY))
print("TYPE_DOC: " + str(REPORTS_TRAINING))
print("flag_KEYWORDS: " + str(flag_KEYWORDS))


MAIN_FLD = '/PATH/MAIN_FLD/'

DATASET_FLD = MAIN_FLD + '/datasets/'
CSV_FOLDER = MAIN_FLD + '/csv_folder/'

fname_test = CSV_FOLDER + DATASET + '/labels_test.csv'
test_dataset = pd.read_csv(fname_test, sep = ',', header = None).values

#"""
TOT_CLASSES = 7
#set_to_filter = [1,3,4]
set_to_filter = [1, 3]
test_dataset = filter_labels(test_dataset, set_to_filter) 
unique, counts = np.unique(test_dataset[:,1], return_counts=True)
N_CLASSES = TOT_CLASSES - len(set_to_filter)
#"""

FEAT_FOLDER = DATASET_FLD+DATASET+'/features_'+CNN_TO_USE+'/'+MODALITY+'/'
FEAT_FOLDER = FEAT_FOLDER + REPORTS_TRAINING + '/'

if (flag_KEYWORDS):
	FEAT_FOLDER = FEAT_FOLDER + 'keywords/'
else:
	FEAT_FOLDER = FEAT_FOLDER + 'no_keywords/'

if (REPORT_AUGMENTATION):
	FEAT_FOLDER = FEAT_FOLDER + 'report_augmentation/'
else:
	FEAT_FOLDER = FEAT_FOLDER + 'no_report_augmentation/'

FEAT_FOLDER = FEAT_FOLDER+'/N_EXP_'+N_EXP_str+'/'

feat_img_fld = FEAT_FOLDER + '/images/'
feat_rep_short_fld = FEAT_FOLDER + '/reports_shorts/'
feat_rep_abcd_fld = FEAT_FOLDER + '/reports_abcd/'
feat_rep_char_fld = FEAT_FOLDER + '/reports_char/'
feat_rep_doc_fld = FEAT_FOLDER + '/reports_doc/'

def save_values(MODEL_PATH, DISTANCE, DATASET, filenames, values):

    fname_file = MODEL_PATH + DISTANCE + '_' + DATASET + '.csv'

    File = {'filenames' : filenames, 'values': values}
    df = pd.DataFrame(File,columns=['filenames', 'values'])
    np.savetxt(fname_file, df.values, fmt='%s',delimiter=',')


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
    
@jit(nopython=True)
def euclidean_distance_numba(u: np.ndarray, v: np.ndarray):

    assert u.shape[0] == v.shape[0]

    dist = np.float32(0.0)

    for i in range(u.shape[0]):
        diff = (u[i] - v[i]).item()
        dist += diff * diff

    return np.sqrt(dist)


def cosine_similarity(a, b, X = 128) -> float:
    # Convert dtype to float32
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)

    # Reshape if possible
    a = a.reshape(-1)
    b = b.reshape(-1)

    # Trim or pad if needed (to exactly X)
    if a.size != X:
        if a.size > X:
            a = a[:X]
        else:
            a = np.pad(a, (0, X - a.size), mode='constant')
    if b.size != X:
        if b.size > X:
            b = b[:X]
        else:
            b = np.pad(b, (0, X - b.size), mode='constant')

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def euclidean_distance(a, b, X = 128) -> float:
    # Convert dtype to float32
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)

    # Reshape if possible
    a = a.reshape(-1)
    b = b.reshape(-1)

    # Trim or pad if needed (to exactly X)
    if a.size != X:
        if a.size > X:
            a = a[:X]
        else:
            a = np.pad(a, (0, X - a.size), mode='constant')
    if b.size != X:
        if b.size > X:
            b = b[:X]
        else:
            b = np.pad(b, (0, X - b.size), mode='constant')

    return np.linalg.norm(a - b)

MAX_ELEMENT = len(test_dataset)

filenames = np.empty((MAX_ELEMENT), dtype=str)

embeddings_imgs = np.empty((MAX_ELEMENT, 128))
embeddings_abcd = np.empty((MAX_ELEMENT, 128))
embeddings_char = np.empty((MAX_ELEMENT, 128))
embeddings_short = np.empty((MAX_ELEMENT, 128))
embeddings_doc = np.empty((MAX_ELEMENT, 128))

dist_euclidean_abcd = np.empty((MAX_ELEMENT))
dist_euclidean_char = np.empty((MAX_ELEMENT))
dist_euclidean_short = np.empty((MAX_ELEMENT))
dist_euclidean_doc = np.empty((MAX_ELEMENT))

dist_cosine_abcd = np.empty((MAX_ELEMENT))
dist_cosine_char = np.empty((MAX_ELEMENT))
dist_cosine_short = np.empty((MAX_ELEMENT))
dist_cosine_doc = np.empty((MAX_ELEMENT))

for i in tqdm(range(len(test_dataset))):

    fname_sample = test_dataset[i,0].split('.')[0]

    fname_img = feat_img_fld + fname_sample + '.npy'
    fname_short = feat_rep_short_fld + fname_sample + '.npy'
    fname_abcd = feat_rep_abcd_fld + fname_sample + '.npy'
    fname_char = feat_rep_char_fld + fname_sample + '.npy'
    fname_doc = feat_rep_doc_fld + fname_sample + '.npy'

    try:
        with open(fname_img, 'rb') as f:
            cls_img = np.load(f)
        
        with open(fname_short, 'rb') as f:
            cls_short = np.load(f)

        with open(fname_abcd, 'rb') as f:
            cls_abcd = np.load(f)

        with open(fname_char, 'rb') as f:
            cls_char = np.load(f)

        with open(fname_doc, 'rb') as f:
            cls_doc = np.load(f)

        #embeddings_imgs[i] = cls_img
        #embeddings_abcd[i] = cls_abcd
        #embeddings_short[i] = cls_short
        #embeddings_char[i] = cls_char
        #embeddings_doc[i] = cls_doc

        """
        try:
            dist_euclidean_abcd[i] = euclidean_distance_numba(cls_img.astype(np.float32), cls_abcd.astype(np.float32))
        except:
            dist_euclidean_abcd[i] = euclidean_distance(cls_img.astype(np.float32), cls_abcd.astype(np.float32))

        try:
            dist_euclidean_char[i] = euclidean_distance_numba(cls_img.astype(np.float32), cls_char.astype(np.float32))
        except:
            dist_euclidean_char[i] = euclidean_distance(cls_img.astype(np.float32), cls_char.astype(np.float32))
        try:
            dist_euclidean_short[i] = euclidean_distance_numba(cls_img.astype(np.float32), cls_short.astype(np.float32))
        except:
            dist_euclidean_short[i] = euclidean_distance(cls_img.astype(np.float32), cls_short.astype(np.float32))
        try:
            dist_euclidean_doc[i] = euclidean_distance_numba(cls_img.astype(np.float32), cls_doc.astype(np.float32))
        except:
            dist_euclidean_doc[i] = euclidean_distance(cls_img.astype(np.float32), cls_doc.astype(np.float32))
        """

        try:
            dist_cosine_abcd[i] = cosine_similarity_numba(cls_img.astype(np.float32), cls_abcd.astype(np.float32))
        except:
            dist_cosine_abcd[i] = cosine_similarity(cls_img.astype(np.float32), cls_abcd.astype(np.float32))

        try:
            dist_cosine_char[i] = cosine_similarity_numba(cls_img.astype(np.float32), cls_char.astype(np.float32))
        except:
            dist_cosine_char[i] = cosine_similarity(cls_img.astype(np.float32), cls_char.astype(np.float32))

        try:
            dist_cosine_short[i] = cosine_similarity_numba(cls_img.astype(np.float32), cls_short.astype(np.float32))
        except:
            dist_cosine_short[i] = cosine_similarity(cls_img.astype(np.float32), cls_short.astype(np.float32))

        try:
            dist_cosine_doc[i] = cosine_similarity_numba(cls_img.astype(np.float32), cls_doc.astype(np.float32))
        except:
            dist_cosine_doc[i] = cosine_similarity(cls_img.astype(np.float32), cls_doc.astype(np.float32))
    except Exception as e:
        #print(e)
        pass

"""
avg_euclidean_abcd = np.mean(dist_euclidean_abcd)
avg_euclidean_char = np.mean(dist_euclidean_char)
avg_euclidean_short = np.mean(dist_euclidean_short)
avg_euclidean_doc = np.mean(dist_euclidean_doc)
"""
avg_cosine_abcd = np.mean(dist_cosine_abcd)
avg_cosine_char = np.mean(dist_cosine_char)
avg_cosine_short = np.mean(dist_cosine_short)
avg_cosine_doc = np.mean(dist_cosine_doc)

#print(avg_euclidean_abcd, avg_euclidean_char, avg_euclidean_short, avg_euclidean_doc)
print(avg_cosine_abcd, avg_cosine_char, avg_cosine_short, avg_cosine_doc)

"""
save_values(MODEL_FLD, 'euclidean_abcd', DATASET, filenames, dist_euclidean_abcd)
save_values(MODEL_FLD, 'euclidean_short', DATASET, filenames, dist_euclidean_short)
save_values(MODEL_FLD, 'euclidean_char', DATASET, filenames, dist_euclidean_char)
save_values(MODEL_FLD, 'euclidean_doc', DATASET, filenames, dist_euclidean_doc)
"""
save_values(MODEL_FLD, 'cosine_abcd', DATASET, filenames, dist_cosine_abcd)
save_values(MODEL_FLD, 'cosine_char', DATASET, filenames, dist_cosine_char)
save_values(MODEL_FLD, 'cosine_short', DATASET, filenames, dist_cosine_short)
save_values(MODEL_FLD, 'cosine_doc', DATASET, filenames, dist_cosine_doc)
