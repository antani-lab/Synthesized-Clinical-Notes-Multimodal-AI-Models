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
from enum_multi import ALG, PHASE, TYPE_DATA, MOD, TYPE_REPORT
import utils_data
import random
from dataloader import ImbalancedDatasetSampler, Dataset_instance_reports_generated, filter_labels
from model import MultimodalArchitecture
from tqdm import tqdm
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
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-c', '--CNN', help='cnn_to_use',type=str, default='densenet121')
parser.add_argument('-e', '--EPOCHS', help='epochs to train',type=int, default=10)
parser.add_argument('-z', '--hidden_space', help='hidden_space_size',type=int, default=128)
parser.add_argument('-b', '--batch_size', help='batch size bag level',type=int, default=128)
parser.add_argument('-f', '--CSV_FOLDER', help='folder where csv including IDs and classes are stored',type=str, default='True')
parser.add_argument('-d', '--DATASET', help='dataset to use',type=str, default='HAM10000')
parser.add_argument('-w', '--PROBLEM', help='type of PROBLEM:binary (nevus vs melanoma), multiclass (sebo vs nevus vs basal vs melanoma)',type=str, default='multiclass')
parser.add_argument('-m', '--MODALITY', help='modalities to use: img, txt, multimodal',type=str, default='img')
parser.add_argument('-t', '--TYPE', help='reports training: all, abcd, char, short, doc, meta, random',type=str, default='all')
parser.add_argument('-k', '--KEYWORDS', help='train on keywords',type=str, default='False')
parser.add_argument('-a', '--AUGMENTATION', help='report augmentation',type=str, default='False')
parser.add_argument('-i', '--INPUT', help='report type to evaluate',type=str, default='abcd')
parser.add_argument('-o', '--OVERWRITE', help='overwrite file',type=str, default='False')

args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
CNN_TO_USE = args.CNN
BATCH_SIZE = args.batch_size
EPOCHS = args.EPOCHS
EPOCHS_str = EPOCHS
PROBLEM = args.PROBLEM
MODALITY = args.MODALITY

INPUT_TYPE_str = args.INPUT
INPUT_TYPE = TYPE_REPORT[INPUT_TYPE_str]

EMBEDDING_bool = True
DATASET = args.DATASET

hidden_space_len = args.hidden_space

OVERWRITE = args.OVERWRITE

if (OVERWRITE == 'True'):
	OVERWRITE = True
else:
	OVERWRITE = False

REPORTS_TRAINING = args.TYPE
REPORT_AUGMENTATION = args.AUGMENTATION

flag_KEYWORDS = args.KEYWORDS

if (flag_KEYWORDS == 'True'):
	flag_KEYWORDS = True
else:
	flag_KEYWORDS = False

if (REPORT_AUGMENTATION == 'True'):
	REPORT_AUGMENTATION = True
else:
	REPORT_AUGMENTATION = False


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

####PATH where find patches
instance_dir = '/PLACEHOLDER/DATASETS/'

#OUTPUT_folder = args.output_folder
print("CREATE DIRECTORY WHERE MODELS WILL BE STORED")


OUTPUT_folder = '/PLACEHOLDER/MODELS/'
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
checkpoint_path = models_path+'checkpoints/'
os.makedirs(checkpoint_path, exist_ok=True)

storing_dir = checkpoint_path + '/test/prediction_txt/'
os.makedirs(storing_dir, exist_ok = True)

filename_val_txts = storing_dir+DATASET+'_predictions_txt_'+INPUT_TYPE_str+'.csv'

if (OVERWRITE or os.path.exists(filename_val_txts)==False):

	#path model file
	model_weights_filename = models_path+'model.pt'

	clinical_note_fld = "/PLACEHOLDER/CLINICAL/NOTES/"+DATASET+"/clinical_notes/decoder_reports/"+MODALITY+"/"+INPUT_TYPE_str+"/"
	
	if (flag_KEYWORDS):
		clinical_note_fld = clinical_note_fld + 'keywords/'
	else:
		clinical_note_fld = clinical_note_fld + 'no_keywords/'

	if (REPORT_AUGMENTATION):
		clinical_note_fld = clinical_note_fld + 'report_augmentation/'
	else:
		clinical_note_fld = clinical_note_fld + 'no_report_augmentation/'

	clinical_note_fld = clinical_note_fld+PROBLEM+"/"+CNN_TO_USE+"/N_EXP_"+N_EXP_str+"/"

	MODALITY = MOD.txt

	#LOAD DATA
	#TODO ADD PARAMETER
	list_DATASETS = ['HAM10000']


	flag_test = PHASE.test
	
	MAIN_FOLDER = 'PLACEHOLDER'
	MAIN_FOLDER = MAIN_FOLDER + '/datasets/'
	CSV_FOLDER = MAIN_FOLDER + '/csv_folder/'
	#all_data = valid_dataset
	target_img = None

	#all_data = valid_dataset
	target_img = None

	test_dataset_filename = CSV_FOLDER+DATASET+'/labels_test.csv'
	test_dataset = pd.read_csv(test_dataset_filename, sep = ',', header = None).values
	#test_dataset = utils_data.get_specific_dataset(MAIN_FOLDER, DATASET, PHASE.test, CSV_FOLDER)

	print(test_dataset.shape)

	#aggregate classes
	N_CLASSES = 1
	#N_CLASSES = 3
	N_CLASSES = 7

	if (PROBLEM == 'binary'):
		#"""
		TOT_CLASSES = 7
		set_to_filter = [0,1,3,4,5]
		test_dataset = filter_labels(test_dataset, set_to_filter, TOT_CLASSES) 
		unique, counts = np.unique(test_dataset[:,1], return_counts=True)
		N_CLASSES = TOT_CLASSES - len(set_to_filter)
		#"""

	elif (PROBLEM == 'multiclass'):
		#"""
		TOT_CLASSES = 7
		#set_to_filter = [1,3,4]
		set_to_filter = [1, 3]
		test_dataset = filter_labels(test_dataset, set_to_filter, TOT_CLASSES) 
		unique, counts = np.unique(test_dataset[:,1], return_counts=True)
		N_CLASSES = TOT_CLASSES - len(set_to_filter)
		#"""

	if(N_CLASSES == 2):
		N_CLASSES = 1
	#"""
	print(test_dataset.shape)

	#"""
	#MODEL DEFINITION
	#CNN BACKBONE
	if (CNN_TO_USE == 'HIPT'):
		fc_input_features = 512

	else:

		pre_trained_network = torch.hub.load('pytorch/vision:v0.10.0', CNN_TO_USE, pretrained=True)
		if (('resnet' in CNN_TO_USE) or ('resnext' in CNN_TO_USE)):
			fc_input_features = pre_trained_network.fc.in_features
		elif (('densenet' in CNN_TO_USE)):
			fc_input_features = pre_trained_network.classifier.in_features
		elif ('mobilenet' in CNN_TO_USE):
			fc_input_features = pre_trained_network.classifier[1].in_features

		del pre_trained_network
		
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)

	AUGMENT_PROB_THRESHOLD = 0.5
	prob = AUGMENT_PROB_THRESHOLD

	sampler = ImbalancedDatasetSampler

	params_valid_bag = {'batch_size': BATCH_SIZE,
			'shuffle': False,
			'drop_last': False,
			'num_workers': 1}


	testing_set_bag = Dataset_instance_reports_generated(test_dataset, N_CLASSES, clinical_note_fld)
	testing_generator_bag = data.DataLoader(testing_set_bag, **params_valid_bag)

	print("initialize CNN")

	input_dim = fc_input_features
	hidden_dim = hidden_space_len
	output_dim = N_CLASSES

	if ('HAM10000' in list_DATASETS):
		centers = len(list_DATASETS) + 1
	else:
		centers = len(list_DATASETS)

	centers = 3

	intermediate_layers = True
	TEMPERATURE = 0.07

	model = MultimodalArchitecture(device, CNN_TO_USE, in_dim = input_dim, 
						out_dim = output_dim, 
						intermediate_dim = hidden_dim,
						TEMPERATURE = TEMPERATURE)

	#load weights
	model.load_state_dict(torch.load(model_weights_filename), strict=False)
	"""
	for name, param in model.base_encoder.named_parameters():
		param.requires_grad = False
	"""
	model.to(device)
	model.eval()


	#lr = 1e-4
	num_epochs = EPOCHS

	lambda_val = 0.5 
	#start loop

	phase_str = 'test'
	dataloader_iterator = iter(testing_generator_bag)
	iterations = int(len(test_dataset) / BATCH_SIZE) + 1
			

	#TODO change classes

	cumulative_preds = np.empty((0, N_CLASSES))
	filenames = []

	y_pred = []
	y_true = []

	progress_bar = tqdm(range(iterations), desc='samples', position=0) 
	kappa_log = tqdm(total=0, position=1, bar_format='{desc}')
	f1_score_log = tqdm(total=0, position=2, bar_format='{desc}')

	for i in progress_bar:
		
		with torch.autocast(device_type='cuda', dtype=torch.float16):

			#print(', %d / %d ' % (i, iterations))
			try:
				tokens, labels, batch_filenames = next(dataloader_iterator)
			except StopIteration:
				dataloader_iterator = iter(testing_generator_bag)
				tokens, labels, batch_filenames = next(dataloader_iterator)

			labels_np = labels.data.numpy()


			batch_filenames = list(batch_filenames)

			
			labels = torch.as_tensor(labels).to(device)

			with torch.no_grad():
				tokens = tokens.to(device)
				_, _, logits_txt, _ = model(None, tokens)


			if (MODALITY is MOD.txt):
				if (N_CLASSES > 1):
					softmax_output_txt = F.softmax(logits_txt)
				else:
					softmax_output_txt = F.sigmoid(logits_txt.view(-1))

				outputs_np_txt = softmax_output_txt.cpu().data.numpy()
			
				if (N_CLASSES > 1):
					output_norm_txt = np.argmax(outputs_np_txt, axis = 1)
				else:
					output_norm_txt = np.where(outputs_np_txt > 0.5, 1, 0)

				y_pred = np.append(y_pred,output_norm_txt)

			y_true = np.append(y_true,labels_np)

			filenames = np.append(filenames, batch_filenames)

			if (N_CLASSES > 1):
				cumulative_preds = np.vstack([cumulative_preds, outputs_np_txt])
			else:
				cumulative_preds = np.vstack([cumulative_preds, np.expand_dims(outputs_np_txt, 1)])

			preds = np.hstack((filenames.reshape(-1,1), cumulative_preds))

			kappa = kappa_score(y_true, y_pred, checkpoint_path, phase_str, None, DATASET)
			f1_score_macro, f1_score_micro, f1_score_weighted = f1_scores(y_true, y_pred, checkpoint_path, phase_str, None, DATASET)


			kappa_log.set_description_str(f'kappa: {kappa}')
			f1_score_log.set_description_str(f'f1_score: {f1_score_weighted}')


	utils_data.save_prediction(checkpoint_path, N_CLASSES, phase_str, None, None, preds, DATASET, MODALITY)

	micro_accuracy_train = accuracy_score(y_true, y_pred, checkpoint_path, phase_str, None, DATASET)
	kappa = kappa_score(y_true, y_pred, checkpoint_path, phase_str, None, DATASET)
	f1_score_macro, f1_score_micro, f1_score_weighted = f1_scores(y_true, y_pred, checkpoint_path, phase_str, None, DATASET)
	precision_score_macro, precision_score_micro = precisions(y_true, y_pred, checkpoint_path, phase_str, None, DATASET)
	recall_score_macro, recall_score_micro = recalls(y_true, y_pred, checkpoint_path, phase_str, None, DATASET)

						
	print("accuracy " + str(micro_accuracy_train)) 
	print("kappa " + str(kappa)) 
	print("f1_score_macro " + str(f1_score_macro))
	print("f1_score_micro " + str(f1_score_micro))
	print("f1_score_weighted " + str(f1_score_weighted))
	print("precision_score_macro " + str(precision_score_macro))
	print("precision_score_micro " + str(precision_score_micro))
	print("recall_score_macro " + str(recall_score_macro))
	print("recall_score_micro " + str(recall_score_micro))



	storing_dir = checkpoint_path + '/test/prediction_txt/'
	os.makedirs(storing_dir, exist_ok = True)

	filename_val_txts = storing_dir+DATASET+'_predictions_txt_'+INPUT_TYPE_str+'.csv'

	columns = ['filenames'] + [f'class_{i}' for i in range(N_CLASSES)]

	# Constructing the data dictionary
	File = {'filenames': preds[:, 0]}
	for i in range(1, N_CLASSES + 1):
		File[f'class_{i - 1}'] = preds[:, i]

	fmt = ['%s'] + ['%.4f'] * (preds.shape[1] - 1)
	# Creating the DataFrame
	df = pd.DataFrame(File, columns=columns)

	np.savetxt(filename_val_txts, df.values, fmt='%s',delimiter=',')