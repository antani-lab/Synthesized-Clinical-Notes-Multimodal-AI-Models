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
sys.path.append("../utils/")
sys.path.append("../models/")

from enum_multi import ALG, PHASE, TYPE_DATA, MOD, REPORTS, CONCEPTS
import utils_data
import random
from dataloader import ImbalancedDatasetSampler, Dataset_instance, filter_labels
from model import MultimodalArchitecture
import json
from metrics_multiclass import accuracy_score, kappa_score, f1_scores, precisions, recalls
from tqdm import tqdm
import sklearn
import utils_zero_shot_learning

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
parser.add_argument('-z', '--hidden_space', help='hidden_space_size',type=int, default=128)
parser.add_argument('-o', '--output_folder', help='path folder where to store output model',type=str, default='')
parser.add_argument('-f', '--CSV_FOLDER', help='folder where csv including IDs and classes are stored',type=str, default='True')
parser.add_argument('-d', '--DATASET', help='dataset to use',type=str, default='HAM10000')
parser.add_argument('-w', '--PROBLEM', help='type of PROBLEM:binary (nevus vs melanoma), multiclass (sebo vs nevus vs basal vs melanoma)',type=str, default='binary')
parser.add_argument('-m', '--MODALITY', help='modalities to use: img, txt, multimodal',type=str, default='img')
parser.add_argument('-t', '--TYPE', help='reports training: all, abcd, char, short, doc, meta, random',type=str, default='meta')
parser.add_argument('-k', '--KEYWORDS', help='train on keywords',type=str, default='False')
parser.add_argument('-a', '--AUGMENTATION', help='report augmentation',type=str, default='False')
parser.add_argument('-i', '--CONCEPTS', help='concepts to analyze: classes, subclasses, classes_matching, subclasses_matching',type=str, default='classes_matching')
parser.add_argument('-x', '--DATA_FOLDER', help='path of the folder where to images are stored',type=str, default='')

args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
CNN_TO_USE = args.CNN
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

CONCEPT_TO_USE_str = args.CONCEPTS
CONCEPT_TO_USE = CONCEPTS[CONCEPT_TO_USE_str]

seed = N_EXP
torch.manual_seed(seed)
#torch.use_deterministic_algorithms(mode=True)
if torch.cuda.is_available():
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


MAIN_FLD = args.DATA_FOLDER
CSV_FLD = MAIN_FLD + "/csv_folder/"
DATASET_FLD = MAIN_FLD + "/datasets/"

MODEL_WEIGHTS_FLD = MAIN_FLD + "/model_weights/multimodal/"

OUTPUT_FLD = MODEL_WEIGHTS_FLD + MODALITY+"/"+REPORTS_TRAINING+"/"

if (flag_KEYWORDS):
	OUTPUT_FLD = OUTPUT_FLD + "keywords/"
else:
	OUTPUT_FLD = OUTPUT_FLD + "no_keywords/"

if (REPORT_AUGMENTATION):
	OUTPUT_FLD = OUTPUT_FLD + "report_augmentation/"
else:
	OUTPUT_FLD = OUTPUT_FLD + "no_report_augmentation/"

OUTPUT_FLD = OUTPUT_FLD + "/multiclass/"+CNN_TO_USE+"/N_EXP_"+N_EXP_str+"/checkpoints/test/zero_shot/"
os.makedirs(OUTPUT_FLD, exist_ok = True)

zero_shot_learning_folder = CSV_FLD + "zero_shot_keywords/"

if (CONCEPT_TO_USE is CONCEPTS.classes):
    csv_filename = zero_shot_learning_folder + "keyword_classes.csv"
    keywords = pd.read_csv(csv_filename, sep = ',', header = None).values.squeeze()

elif (CONCEPT_TO_USE is CONCEPTS.subclasses):
    csv_filename = zero_shot_learning_folder + "keyword_subclasses.csv"
    keywords = pd.read_csv(csv_filename, sep = ',', header = None).values.squeeze()

elif (CONCEPT_TO_USE is CONCEPTS.classes_matching):
    csv_filename = zero_shot_learning_folder + "keyword_classes_matching.csv"
    keywords = pd.read_csv(csv_filename, sep = ',', header = None).values.squeeze()
    N_CLASSES = 8 + 3

elif (CONCEPT_TO_USE is CONCEPTS.subclasses_matching):
    csv_filename = zero_shot_learning_folder + "keyword_subclasses_matching.csv"
    keywords = pd.read_csv(csv_filename, sep = ',', header = None).values.squeeze()
    N_CLASSES = 15 + 3

print(keywords)

csv_filename = CSV_FLD + "/" + DATASET + "/classes_subclasses_metadata_mapping.csv"
csv_metadata = pd.read_csv(csv_filename, sep = ',', header = None).values


filename_test = CSV_FLD + "/" + DATASET + "/labels_test.csv"
test_dataset = pd.read_csv(filename_test, sep = ',', header = None).values



PRED_SIMILARITY_FLD = DATASET_FLD + DATASET + "/features_"+CNN_TO_USE+"/"+MODALITY+"/"+REPORTS_TRAINING+"/"

if (flag_KEYWORDS):
	PRED_SIMILARITY_FLD = PRED_SIMILARITY_FLD + "keywords/"
else:
	PRED_SIMILARITY_FLD = PRED_SIMILARITY_FLD + "no_keywords/"

if (REPORT_AUGMENTATION):
	PRED_SIMILARITY_FLD = PRED_SIMILARITY_FLD + "report_augmentation/"
else:
	PRED_SIMILARITY_FLD = PRED_SIMILARITY_FLD + "no_report_augmentation/"

PRED_SIMILARITY_FLD = PRED_SIMILARITY_FLD + "/N_EXP_"+N_EXP_str+"/images/"

if (CONCEPT_TO_USE is CONCEPTS.classes):
    similarity_classes_fname = PRED_SIMILARITY_FLD + "feature_similarities_keyword_classes.csv"
    similarity_classes = pd.read_csv(similarity_classes_fname, sep = ',', header = None).values#.tolist()
    idx_metadata = 4
    #labels_metadata = csv_metadata[:,idx_metadata]

elif (CONCEPT_TO_USE is CONCEPTS.subclasses):
    similarity_classes_fname = PRED_SIMILARITY_FLD + "feature_similarities_keyword_subclasses.csv"
    similarity_classes = pd.read_csv(similarity_classes_fname, sep = ',', header = None).values#.tolist()
    idx_metadata = 5
    #labels_metadata = csv_metadata[:,idx_metadata]

elif (CONCEPT_TO_USE is CONCEPTS.classes_matching):
    similarity_classes_fname = PRED_SIMILARITY_FLD + "feature_similarities_keyword_classes_matching.csv"
    similarity_classes = pd.read_csv(similarity_classes_fname, sep = ',', header = None).values#.tolist()
    idx_metadata = 6
    #labels_metadata = csv_metadata[:,idx_metadata]

elif (CONCEPT_TO_USE is CONCEPTS.subclasses_matching):
    similarity_classes_fname = PRED_SIMILARITY_FLD + "feature_similarities_keyword_subclasses_matching.csv"
    similarity_classes = pd.read_csv(similarity_classes_fname, sep = ',', header = None).values#.tolist()
    idx_metadata = 7
    #labels_metadata = csv_metadata[:,idx_metadata]

keywords_list = keywords.tolist()

def get_row_metadata(fname, metadata):

    i = 0
    b = False
    row = -1

    fname = fname.split('.')[0]

    while (i<len(metadata) and b == False):

        current_fname = metadata[i,0]
        if (current_fname in fname or fname in current_fname):
            row = metadata[i]
            b = True
        else:
            i = i + 1

    if (b == False):
        i = -1
    return i



def mapping_concept_to_idx(concept, keywords_list, CONCEPT_TO_USE):

    if CONCEPT_TO_USE is CONCEPTS.classes_matching:

        if ('actinic keratosis' in concept):
            idx = 0
        elif ('basal cell cancer' in concept or 'basal cell carcinoma' in concept ):
            idx = 1
        elif ('seborrheic keratosis' in concept or 'benign/seborrheic keratosis' in concept):
            idx = 2
        elif ('dermatofibroma' in concept):
            idx = 3
        elif ('melanocytic / benign nevus' in concept or 'melanocytic nevus' in concept):
            idx = 4
        elif ('melanoma' in concept):
            idx = 5
        elif ('squamous cell cancer' in concept or 'squamous cell carcinoma' in concept ):
            idx = 6
        elif ('vascular lesion' in concept):
            idx = 7
        

    elif CONCEPT_TO_USE is CONCEPTS.subclasses_matching:

        if ('actinic keratosis' in concept):
            idx = 0
        elif ('basal cell cancer' in concept or 'basal cell carcinoma' in concept):
            idx = 1
        elif ('benign melanocytic nevus' in concept or 'melanocytic nevus' in concept):
            idx = 2
        elif ('blue nevus' in concept):
            idx = 3
        elif ('bowen disease / SCC in situ' in concept):
            idx = 4
        elif ('congenital / special-pattern nevi' in concept):
            idx = 5
        elif ('dermatofibroma / fibrous lesions' in concept):
            idx = 6
        elif ('dysplastic / atypical nevus (clark-type)' in concept or 'dysplastic / atypical clark-type' in concept):
            idx = 7
        elif ('lentigo maligna' in concept):
            idx = 8
        elif ('lichenoid keratosis' in concept):
            idx = 9
        elif ('melanoma' in concept):
            idx = 10
        elif ('seborrheic keratosis & pigmented keratoses' in concept or 'seborrheic keratosis' in concept):
            idx = 11
        elif ('solar lentigo' in concept):
            idx = 12
        elif ('squamous cell carcinoma (invasive)' in concept or 'squamous cell carcinoma' in concept):
            idx = 13
        elif ('vascular lesion' in concept):
            idx = 14


    return idx


y_true = np.empty(len(test_dataset))

output_similarities = np.empty((len(test_dataset), N_CLASSES), dtype = "object")

for i in tqdm(range(len(test_dataset))):
    current_fname = test_dataset[i][0]
    row = get_row_metadata(current_fname, csv_metadata)
    #y_true.append(mapping_concept_to_idx(csv_metadata[row,idx_metadata], keywords_list, CONCEPT_TO_USE))
    y_true[i] = mapping_concept_to_idx(csv_metadata[row,idx_metadata], keywords_list, CONCEPT_TO_USE)
    output_similarities[i,0] = current_fname
    output_similarities[i,1] = y_true[i]

y_pred = np.empty(len(test_dataset))



for i in tqdm(range(len(test_dataset))):

    current_fname = test_dataset[i][0]
    row = get_row_metadata(current_fname, similarity_classes)

    similarities = similarity_classes[row,1:]
    #y_pred.append(np.argmax(similarities))
    y_pred[i] = np.argmax(similarities)
    output_similarities[i,2] = y_pred[i]
    output_similarities[i,3:] = similarities


f1_score = sklearn.metrics.f1_score(y_true = y_true, y_pred = y_pred, average = 'weighted')

print(f1_score)

new_filename = OUTPUT_FLD + "f1_score_"+DATASET+"_"+CONCEPT_TO_USE_str+".csv"
File = {'val' : [f1_score]}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(new_filename, df.values, fmt='%s',delimiter=',')

new_filename = OUTPUT_FLD + "f1_score_"+DATASET+"_"+CONCEPT_TO_USE_str+"_similarity.csv"
np.savetxt(new_filename, output_similarities, delimiter=",", fmt='%s',)