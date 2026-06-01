import pandas as pd
import numpy as np
import os, sys
sys.path.append("../utils/")
sys.path.append("../models/")

from enum_multi import MOD, PHASE, TYPE_REPORT, REPORTS, PARTITION
from tqdm import tqdm
import argparse
import utils_retrieval
from sklearn.metrics.pairwise import cosine_similarity


#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-m', '--MODALITY', help='modalities to use: multimodal, CLIP, NT_Xent',type=str, default='multimodal')
parser.add_argument('-i', '--TYPE_INPUT', help='img / abcd / etc.',type=str, default='img')
parser.add_argument('-o', '--TYPE_POOL', help='img / abcd / etc. ',type=str, default='img')

parser.add_argument('-c', '--CNN', help='CNN TO USE',type=str, default='densenet121')
parser.add_argument('-w', '--multiclass', help='CNN TO USE',type=str, default='multiclass')
parser.add_argument('-k', '--KEYWORDS', help='train on keywords',type=str, default='False')
parser.add_argument('-d', '--DOC', help='reports training: all, abcd, char, short, doc, meta, random',type=str, default='all')
parser.add_argument('-a', '--AUGMENTATION', help='report augmentation',type=str, default='False')
parser.add_argument('-p', '--PARTITION', help='internal, external, dermoscopic, whole',type=str, default='external')
parser.add_argument('-x', '--DATA_FOLDER', help='path of the folder where to images are stored',type=str, default='')
parser.add_argument('-y', '--WEIGHTS', help='path for pre-trained weights',type=str, default='')

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

PARTITION_TO_USE_str = args.PARTITION
PARTITION_TO_USE = PARTITION[PARTITION_TO_USE_str]

MAIN_FLD = args.DATA_FOLDER
DATA_FLD = MAIN_FLD + '/datasets/'

CSV_FOLDER = MAIN_FLD + '/csv_folder/'

MODALITY = args.MODALITY
N_EXP = str(args.N_EXP)
CNN_TO_USE = args.CNN

print("MODALITY: " + str(MODALITY))
print("N_EXP: " + str(N_EXP))
print("TYPE_DOC: " + str(REPORTS_TRAINING))



POOL_DATASETS_TEST = ['HAM10000', 'BCN20000', 'Derm7pt', 'DermNet', 'FLUO_SC', 'MRA_MIDAS',
			'Fitzpatrick17k', 'Hospital_Italiano_Buenos_Aires',
			'PAD_UFES_20', 'SD198', 'derm12345', 'SKINL2',
			'MSK', 'Milk10k_clinic', 'Milk10k_dermo']

flag_o4 = False

MODEL_FLD = args.WEIGHTS + MODALITY + '/'
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

DATA_INPUT_str = args.TYPE_INPUT
DATA_POOL_str = args.TYPE_POOL

print(DATA_INPUT_str, DATA_POOL_str)

DATA_INPUT = REPORTS[DATA_INPUT_str]
DATA_POOL = REPORTS[DATA_POOL_str]

NUM_CLASSES = 8

fold_test = DATA_FLD+'DATASET/features_'+CNN_TO_USE+'/'+MODALITY+'/'+REPORTS_TRAINING+'/'

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



def get_input_features(list_folder, csv_dataset, N_ELEMENTS = 1000000):

	
	features = np.empty((N_ELEMENTS, 128))
	labels = np.empty((N_ELEMENTS))
	filenames = np.empty((N_ELEMENTS), dtype = "object")
	idx = 0

	for feat_fold_test in list_folder:

		for i in tqdm(range(len(csv_dataset))):
			fname = feat_fold_test + csv_dataset[i,0].split('.')[0] + '.npy'
			label = csv_dataset[i,1]

			try:
				with open(fname, 'rb') as f:
					cls_img = np.load(f)
				
				features[idx] = cls_img
				labels[idx] = int(label)
				filenames[idx] = csv_dataset[i,0].split('.')[0]
				idx = idx + 1

			except Exception as e:
				print(e)

	return features, labels, idx, filenames


def get_pool_features(list_folder, csv_dataset, features, labels, idx, seen, DATA_TYPE = REPORTS.images):

	idx_to_use = idx
	
	for feat_fold_test in list_folder:

		for i in tqdm(range(len(csv_dataset))):
			fname = feat_fold_test + csv_dataset[i,0].split('.')[0] + '.npy'
			label = csv_dataset[i,1]

			try:
				with open(fname, 'rb') as f:
					cls_img = np.load(f)


				if (DATA_TYPE is REPORTS.images):

					features[idx_to_use] = cls_img
					labels[idx_to_use] = int(label)
					idx_to_use = idx_to_use + 1

				else:
					key = cls_img.tobytes()
					#if not np.any(np.all(elements_to_check == cls_img, axis=1)):
					if (key not in seen):
						features[idx_to_use] = cls_img
						labels[idx_to_use] = int(label)
						idx_to_use = idx_to_use + 1
						seen.add(key)

			except Exception as e:
				print(e, 'aaa')
				pass
	
	print(idx_to_use)
	return features, labels, idx_to_use, seen

def get_feature_dataset(main_fld, dataset):

	return main_fld.replace('DATASET', dataset)

def get_pool_data(POOL_DATASETS, DATA_TYPE, CSV_FOLDER, MAIN_FLD, DIM_RESIZE = 128, MAX_ELEMENT = 1000000, flag_o4 = False):

	#POOL: evaluate the number of size needed 

	feature_pool = np.empty((MAX_ELEMENT, 128))
	label_pool = np.empty((MAX_ELEMENT))
	
	seen = set()

	idx = 0

	print("pool data")
	for d in POOL_DATASETS:
		csv_test_file = CSV_FOLDER + d + '/labels.csv'
		csv_dataset = pd.read_csv(csv_test_file, sep = ',', header = None).values
		
		#set_to_filter = []
		#csv_dataset = utils_data.filter_labels(csv_dataset, set_to_filter) 

		fold_test = get_feature_dataset(MAIN_FLD, d)

		if (DATA_TYPE is REPORTS.images):
			list_folder = [fold_test + 'images/']

			#gpt
		elif (DATA_TYPE is REPORTS.abcd):
			list_folder = [fold_test + 'reports_abcd/']
		elif (DATA_TYPE is REPORTS.short):
			list_folder = [fold_test + 'reports_shorts/']
		elif (DATA_TYPE is REPORTS.doc):
			list_folder = [fold_test + 'reports_doc/']
		elif (DATA_TYPE is REPORTS.char):
			list_folder = [fold_test + 'reports_char/']
		elif (DATA_TYPE is REPORTS.meta):
			list_folder = [fold_test + 'reports_abcd/', fold_test + 'reports_char/']
		elif (DATA_TYPE is REPORTS.all):
			list_folder = [fold_test + 'reports_abcd/', fold_test + 'reports_shorts/', fold_test + 'reports_char/']

		
		#skingpt4
		elif (DATA_TYPE is REPORTS.skingpt4_meta):
			list_folder = [fold_test + 'reports_skingpt4_abcd/', fold_test + 'reports_skingpt4_char/']

		elif (DATA_TYPE is REPORTS.skingpt4_all):
			list_folder = [fold_test + 'reports_skingpt4_abcd/', fold_test + 'reports_shorts/', fold_test + 'reports_skingpt4_char/']

		elif (DATA_TYPE is REPORTS.skingpt4_p1):
			list_folder = [fold_test + 'reports_skingpt4_p1/']

		elif (DATA_TYPE is REPORTS.skingpt4_p1_all):
			list_folder = [fold_test + 'reports_skingpt4_p1/', fold_test + 'reports_shorts/']

		elif (DATA_TYPE is REPORTS.skingpt4_p2):
			list_folder = [fold_test + 'reports_skingpt4_p2/']

		#dermlip
		elif (DATA_TYPE is REPORTS.dermlip_meta):
			list_folder = [fold_test + 'reports_dermlip_abcd/', fold_test + 'reports_dermlip_char/']

		elif (DATA_TYPE is REPORTS.dermlip_all):
			list_folder = [fold_test + 'reports_dermlip_abcd/', fold_test + 'reports_shorts/', fold_test + 'reports_dermlip_char/']

		elif (DATA_TYPE is REPORTS.dermlip_p1):
			list_folder = [fold_test + 'reports_dermlip_p1/']

		elif (DATA_TYPE is REPORTS.dermlip_p1_all):
			list_folder = [fold_test + 'reports_dermlip_p1/', fold_test + 'reports_shorts/']

		elif (DATA_TYPE is REPORTS.dermlip_p2):
			list_folder = [fold_test + 'reports_dermlip_p2/']

		#medgemma
		elif (DATA_TYPE is REPORTS.medgemma_meta):
			list_folder = [fold_test + 'reports_medgemma_abcd/', fold_test + 'reports_medgemma_char/']

		elif (DATA_TYPE is REPORTS.medgemma_all):
			list_folder = [fold_test + 'reports_medgemma_abcd/', fold_test + 'reports_shorts/', fold_test + 'reports_medgemma_char/']
	

		feature_pool, label_pool, idx, seen = get_pool_features(list_folder, csv_dataset, feature_pool, label_pool, idx, seen, DATA_TYPE)

	
	feature_pool = feature_pool[:idx]
	label_pool = label_pool[:idx]

	N_ELEM = idx
	feature_pool = np.reshape(feature_pool, (N_ELEM, DIM_RESIZE))

	return feature_pool, label_pool

def get_test_data(DATASET, CSV_FOLDER, MAIN_FLD, DIM_RESIZE = 128, DATA_TYPE = REPORTS.abcd, MAX_ELEMENT = 500000):

	#print(REPORT_type)
	feature_input = np.empty((MAX_ELEMENT, 128))
	label_input = np.empty((MAX_ELEMENT))

	idx = 0

	#INPUT
	#print("input data")
	csv_test_file = CSV_FOLDER + DATASET + '/labels_test.csv'
	test_dataset = pd.read_csv(csv_test_file, sep = ',', header = None).values

	#set_to_filter = []
	#test_dataset = utils_data.filter_labels(test_dataset, set_to_filter) 
	
	fold_test = get_feature_dataset(MAIN_FLD, DATASET)

	if (DATA_TYPE is REPORTS.images):
			list_folder = [fold_test + 'images/']
		#gpt
	elif (DATA_TYPE is REPORTS.abcd):
		list_folder = [fold_test + 'reports_abcd/']
	elif (DATA_TYPE is REPORTS.short):
		list_folder = [fold_test + 'reports_shorts/']
	elif (DATA_TYPE is REPORTS.doc):
		list_folder = [fold_test + 'reports_doc/']
	elif (DATA_TYPE is REPORTS.char):
		list_folder = [fold_test + 'reports_char/']
	elif (DATA_TYPE is REPORTS.meta):
		list_folder = [fold_test + 'reports_abcd/', fold_test + 'reports_char/']
	elif (DATA_TYPE is REPORTS.all):
		list_folder = [fold_test + 'reports_abcd/', fold_test + 'reports_shorts/', fold_test + 'reports_char/']

	
	#skingpt4
	elif (DATA_TYPE is REPORTS.skingpt4_meta):
		list_folder = [fold_test + 'reports_skingpt4_abcd/', fold_test + 'reports_skingpt4_char/']

	elif (DATA_TYPE is REPORTS.skingpt4_all):
		list_folder = [fold_test + 'reports_skingpt4_abcd/', fold_test + 'reports_shorts/', fold_test + 'reports_skingpt4_char/']

	elif (DATA_TYPE is REPORTS.skingpt4_p1):
		list_folder = [fold_test + 'reports_skingpt4_p1/']

	elif (DATA_TYPE is REPORTS.skingpt4_p1_all):
		list_folder = [fold_test + 'reports_skingpt4_p1/', fold_test + 'reports_shorts/']

	elif (DATA_TYPE is REPORTS.skingpt4_p2):
		list_folder = [fold_test + 'reports_skingpt4_p2/']

	#dermlip
	elif (DATA_TYPE is REPORTS.dermlip_meta):
		list_folder = [fold_test + 'reports_dermlip_abcd/', fold_test + 'reports_dermlip_char/']

	elif (DATA_TYPE is REPORTS.dermlip_all):
		list_folder = [fold_test + 'reports_dermlip_abcd/', fold_test + 'reports_shorts/', fold_test + 'reports_dermlip_char/']

	elif (DATA_TYPE is REPORTS.dermlip_p1):
		list_folder = [fold_test + 'reports_dermlip_p1/']

	elif (DATA_TYPE is REPORTS.dermlip_p1_all):
		list_folder = [fold_test + 'reports_dermlip_p1/', fold_test + 'reports_shorts/']

	elif (DATA_TYPE is REPORTS.dermlip_p2):
		list_folder = [fold_test + 'reports_dermlip_p2/']

	#medgemma
	elif (DATA_TYPE is REPORTS.medgemma_meta):
		list_folder = [fold_test + 'reports_medgemma_abcd/', fold_test + 'reports_medgemma_char/']

	elif (DATA_TYPE is REPORTS.medgemma_all):
		list_folder = [fold_test + 'reports_medgemma_abcd/', fold_test + 'reports_shorts/', fold_test + 'reports_medgemma_char/']


	feature_input, label_input, idx, filenames = get_input_features(list_folder, test_dataset)

	feature_input = feature_input[:idx]
	label_input = label_input[:idx]
	filenames = filenames[:idx]

	N_ELEM = idx
	feature_input = np.reshape(feature_input, (N_ELEM, DIM_RESIZE))

	return feature_input, label_input, filenames

if (DATA_INPUT_str == "images"):
	MOD_INPUT = MOD.img
else:
	MOD_INPUT = MOD.txt


def save_prediction(MODEL_PATH, K, INPUT_DATA, OUTPUT_DATA, VALUE_lab, PARTITION_TO_USE_str):

	fname_file = MODEL_PATH + 'input_' + INPUT_DATA + '_pool_' +OUTPUT_DATA+ '_K_' + str(K) + '_similarities_label_'+PARTITION_TO_USE_str+'.csv'
	print(fname_file)
	np.savetxt(fname_file, VALUE_lab, delimiter=",", fmt='%s',)


def save_metric(MODEL_PATH, METRIC, INPUT_DATA, OUTPUT_DATA, VALUE, PARTITION_TO_USE_str):


	fname_file = MODEL_PATH + METRIC + '_input_' + INPUT_DATA + '_pool_' +OUTPUT_DATA+ '_'+PARTITION_TO_USE_str+'.csv'

	print(fname_file)

	File = {'val' : [VALUE]}
	df = pd.DataFrame(File,columns=['val'])
	np.savetxt(fname_file, df.values, fmt='%s',delimiter=',')

if (PARTITION_TO_USE is PARTITION.internal):

	DATASETS = ['BCN20000', 'derm12345', 
			'Derm7pt', 'DermNet', 
			'FLUO_SC', 'MRA_MIDAS', 
			]

elif (PARTITION_TO_USE is PARTITION.external):
	DATASETS = [ 
			'HAM10000', 'SKINL2',
			'Fitzpatrick17k', 'Hospital_Italiano_Buenos_Aires',
			'PAD_UFES_20', 'SD198', 
			'MSK', 'Milk10k_clinic', 'Milk10k_dermo']

elif (PARTITION_TO_USE is PARTITION.dermoscopic):
	DATASETS = ['BCN20000', 'derm12345', 
			'HAM10000', 'SKINL2',
			'Hospital_Italiano_Buenos_Aires', 
			'MSK', 'Milk10k_dermo']

elif (PARTITION_TO_USE is PARTITION.clinical):
	DATASETS = [ 
			'Derm7pt', 'DermNet', 
			'FLUO_SC', 'MRA_MIDAS', 
			
			'Fitzpatrick17k', 
			'PAD_UFES_20', 'SD198', 
			'MSK', 'Milk10k_clinic']

elif (PARTITION_TO_USE is PARTITION.whole):
	DATASETS = ['BCN20000', 'derm12345', 
			'Derm7pt', 'DermNet', 
			'FLUO_SC', 'MRA_MIDAS', 
			'HAM10000', 'SKINL2',
			'Fitzpatrick17k', 'Hospital_Italiano_Buenos_Aires',
			'PAD_UFES_20', 'SD198', 
			'MSK', 'Milk10k_clinic', 'Milk10k_dermo']


print("getting pool data")
feature_pool, label_pool = get_pool_data(POOL_DATASETS_TEST, DATA_POOL, CSV_FOLDER, fold_test, flag_o4 = flag_o4)
print(feature_pool.shape)
print("pool data loaded")

pool_features = feature_pool
pool_labels = label_pool
pool_labels = np.array(pool_labels)

max_rows = 1000000

query_features_final = np.empty((max_rows, 128))
query_labels_final = np.empty((max_rows))
filenames_final = np.empty((max_rows), dtype = "object")

i = 0  # write pointer

KK = 100

def get_similarities(filenames, query_labels, pool_labels, sorted_indices, sim_matrix, k = 10):

	whole_array = np.empty((len(filenames), k + 2), dtype = "object")
	whole_array_sim = np.empty((len(filenames), k + 2), dtype = "object")
	#print(whole_array.shape)
	for i in range(len(filenames)):
		whole_array[i,0] = filenames[i]

		whole_array[i,1] = query_labels[i]
		
		retrieved_labels = pool_labels[sorted_indices[i]]

		k_elements = retrieved_labels[:k]

		#print(retrieved_labels.shape, k_elements.shape)
		whole_array[i,2:] = k_elements

	return whole_array

for DATASET in DATASETS:

	print("DATASET: " + str(DATASET))
	feature_input, label_input, filenames = get_test_data(DATASET, CSV_FOLDER, fold_test, DATA_TYPE = DATA_INPUT)
	#get_test_data(DATASET, MOD_INPUT, MOD_OUTPUT, CSV_FOLDER, MAIN_FLD, MODALITY, N_EXP, DIM_RESIZE = 128)
	Y = feature_input.shape[0]

	query_features_final[i:i + Y] = feature_input
	query_labels_final[i:i + Y] = label_input
	filenames_final[i:i + Y] = filenames
	i = i + Y

filenames_final = filenames_final[:i]
query_features_final = query_features_final[:i]
query_labels_final = query_labels_final[:i]

query_features = query_features_final
query_labels = query_labels_final
query_labels = np.array(query_labels)

print(query_features.shape)
sim_matrix = cosine_similarity(query_features_final, pool_features)  # shape: (num_queries, num_pool)
sorted_indices = np.argsort(-sim_matrix, axis=1)  # descending order


mAP = utils_retrieval.eval_mAP(sorted_indices, query_labels_final, pool_labels)
METRIC = "mAP"
save_metric(MODEL_FLD, METRIC, DATA_INPUT_str, DATA_POOL_str, mAP, PARTITION_TO_USE_str)
print("mAP: " + str(mAP))

print()

whole_array = get_similarities(filenames_final, query_labels, pool_labels, sorted_indices, sim_matrix, k = KK)

save_prediction(MODEL_FLD, KK, DATA_INPUT_str, DATA_POOL_str, whole_array, PARTITION_TO_USE_str)