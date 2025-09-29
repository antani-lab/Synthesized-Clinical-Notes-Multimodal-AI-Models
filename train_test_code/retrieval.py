import pandas as pd
import numpy as np
import os, sys
from enum_multi import MOD, PHASE, TYPE_REPORT, REPORTS
from tqdm import tqdm
import argparse
import utils_retrieval
from sklearn.metrics.pairwise import cosine_similarity


#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-m', '--MODALITY', help='modalities to use: multimodal, CLIP, NT_Xent',type=str, default='multimodal')
parser.add_argument('-t', '--TYPE', help='type of data',type=str, default='img')
parser.add_argument('-r', '--REPORT', help='REPORT TYPE: abcd, short, char, doc',type=str, default='abcd')

parser.add_argument('-c', '--CNN', help='CNN TO USE',type=str, default='densenet121')
parser.add_argument('-w', '--multiclass', help='CNN TO USE',type=str, default='multiclass')
parser.add_argument('-k', '--KEYWORDS', help='train on keywords',type=str, default='False')
parser.add_argument('-d', '--DOC', help='reports training: all, abcd, char, short, doc, meta, random',type=str, default='all')
parser.add_argument('-a', '--AUGMENTATION', help='report augmentation',type=str, default='False')

args = parser.parse_args()

REPORTS_TRAINING = args.DOC
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

FLD = 'PLACEHOLDER'
MAIN_FLD = FLD + '/datasets/'

CSV_FOLDER = FLD + '/csv_folder/'

MODALITY = args.MODALITY
N_EXP = str(args.N_EXP)
CNN_TO_USE = args.CNN

print("MODALITY: " + str(MODALITY))
print("N_EXP: " + str(N_EXP))
print("TYPE_DOC: " + str(REPORTS_TRAINING))



POOL_DATASETS_TEST = ['HAM10000', 'BCN20000', 'Derm7pt', 'DermNet',
			'Fitzpatrick17k', 'Hospital_Italiano_Buenos_Aires',
			'PAD_UFES_20', 'SD198', 'derm12345', 'SKINL2']

REPORT_TYPE = args.REPORT
REPORT_style = TYPE_REPORT[REPORT_TYPE]

if ('o4' in REPORTS_TRAINING):
	flag_o4 = True
else:
	flag_o4 = False

MODEL_FLD = '/PLACEHOLDER_MODEL_FLD/' + MODALITY + '/'
os.makedirs(MODEL_FLD, exist_ok=True)

MODEL_FLD = MODEL_FLD + REPORTS_TRAINING + '/'
os.makedirs(MODEL_FLD, exist_ok=True)

if (flag_KEYWORDS):
	MODEL_FLD = MODEL_FLD + 'keywords/'
	os.makedirs(MODEL_FLD, exist_ok=True)
else:
	MODEL_FLD = MODEL_FLD + 'no_keywords/'
	os.makedirs(MODEL_FLD, exist_ok=True)

if (REPORT_AUGMENTATION):
	MODEL_FLD = MODEL_FLD + 'report_augmentation/'
	os.makedirs(MODEL_FLD, exist_ok=True)
else:
	MODEL_FLD = MODEL_FLD + 'no_report_augmentation/'
	os.makedirs(MODEL_FLD, exist_ok=True)

MODEL_FLD = MODEL_FLD + '/multiclass/'+CNN_TO_USE+'/N_EXP_'+N_EXP+'/checkpoints/test/retrieval/'
os.makedirs(MODEL_FLD, exist_ok=True)



#decide modality
MOD_INPUT_str = args.TYPE
#MOD_INPUT = 'txt'

print("TYPE: " + str(MOD_INPUT_str))
print("REPORT_TYPE: " + str(REPORT_TYPE))#

if (MOD_INPUT_str == 'img'):
	MOD_INPUT = MOD.img
	MOD_OUTPUT = MOD.txt
else:
	MOD_INPUT = MOD.txt
	MOD_OUTPUT = MOD.img

NUM_CLASSES = 5

fold_test = MAIN_FLD+'DATASET/features_'+CNN_TO_USE+'/'+MODALITY+'/'+REPORTS_TRAINING+'/'

if (flag_KEYWORDS):
	fold_test = fold_test + 'keywords/'
	os.makedirs(fold_test, exist_ok=True)
else:
	fold_test = fold_test + 'no_keywords/'
	os.makedirs(fold_test, exist_ok=True)

if (REPORT_AUGMENTATION):
	fold_test = fold_test + 'report_augmentation/'
	os.makedirs(fold_test, exist_ok=True)
else:
	fold_test = fold_test + 'no_report_augmentation/'
	os.makedirs(fold_test, exist_ok=True)

fold_test = fold_test+'/N_EXP_'+N_EXP+'/'

print("getting pool data")
feature_pool, label_pool = utils_retrieval.get_pool_data(POOL_DATASETS_TEST, MOD_INPUT, CSV_FOLDER, fold_test, flag_o4 = flag_o4)
print(feature_pool.shape)
print("pool data loaded")

pool_features = feature_pool
pool_labels = label_pool
pool_labels = np.array(pool_labels)

for DATASET in POOL_DATASETS_TEST:

	print("DATASET: " + str(DATASET))
	feature_input, label_input = utils_retrieval.get_test_data(DATASET, MOD_INPUT, CSV_FOLDER, fold_test, REPORT_type = REPORT_style)

	query_features = feature_input
	query_labels = label_input
	query_labels = np.array(query_labels)
	

	sim_matrix = cosine_similarity(query_features, pool_features)  # shape: (num_queries, num_pool)
	sorted_indices = np.argsort(-sim_matrix, axis=1)  # descending order


	k = 5
	p, _ = utils_retrieval.eval_precision_recall(sorted_indices, label_input, pool_labels, k)
	METRIC = "precision@"+str(k)
	utils_retrieval.save_metric(MODEL_FLD, METRIC, DATASET, MOD_INPUT_str, p, REPORT_TYPE)
	print("precision@"+str(k) + ": " + str(p))
	
	k = 10
	p, _ = utils_retrieval.eval_precision_recall(sorted_indices, label_input, pool_labels, k)
	METRIC = "precision@"+str(k)
	utils_retrieval.save_metric(MODEL_FLD, METRIC, DATASET, MOD_INPUT_str, p, REPORT_TYPE)
	print("precision@"+str(k) + ": " + str(p))
	
	k = 20
	p, _ = utils_retrieval.eval_precision_recall(sorted_indices, label_input, pool_labels, k)
	METRIC = "precision@"+str(k)
	utils_retrieval.save_metric(MODEL_FLD, METRIC, DATASET, MOD_INPUT_str, p, REPORT_TYPE)
	print("precision@"+str(k) + ": " + str(p))
	
	k = 50
	p, _ = utils_retrieval.eval_precision_recall(sorted_indices, label_input, pool_labels, k)
	METRIC = "precision@"+str(k)
	utils_retrieval.save_metric(MODEL_FLD, METRIC, DATASET, MOD_INPUT_str, p, REPORT_TYPE)
	print("precision@"+str(k) + ": " + str(p))
	
	k = 100
	p, _ = utils_retrieval.eval_precision_recall(sorted_indices, label_input, pool_labels, k)
	METRIC = "precision@"+str(k)
	utils_retrieval.save_metric(MODEL_FLD, METRIC, DATASET, MOD_INPUT_str, p, REPORT_TYPE)
	print("precision@"+str(k) + ": " + str(p))
	
	mAP = utils_retrieval.eval_mAP(sorted_indices, label_input, pool_labels)
	METRIC = "mAP"
	utils_retrieval.save_metric(MODEL_FLD, METRIC, DATASET, MOD_INPUT_str, mAP, REPORT_TYPE)
	print("mAP: " + str(mAP))

	print()