import sys, getopt
import torch
from torch.utils import data
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
import argparse
import warnings
#warnings.filterwarnings("ignore")
from enum_multi import ALG, PHASE, TYPE_DATA, MOD, COMPONENTS, REPORTS
import utils_data
import random
from dataloader import ImbalancedDatasetSampler, Dataset_reports_only
from model import MultimodalArchitecture
from tqdm import tqdm
from metrics_multiclass import accuracy_score, kappa_score, f1_scores, precisions, recalls
from transformers import BertModel

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
parser.add_argument('-c', '--CNN', help='cnn_to_use',type=str, default='PanDerm')
parser.add_argument('-e', '--EPOCHS', help='epochs to train',type=int, default=10)
parser.add_argument('-z', '--hidden_space', help='hidden_space_size',type=int, default=128)
parser.add_argument('-b', '--batch_size', help='batch size bag level',type=int, default=128)
parser.add_argument('-o', '--output_folder', help='path folder where to store output model',type=str, default='')
parser.add_argument('-i', '--INPUT_FOLDER', help='path of the folder where to images are stored',type=str, default='')
parser.add_argument('-f', '--CSV_FOLDER', help='folder where csv including IDs and classes are stored',type=str, default='True')
parser.add_argument('-d', '--DATASET', help='dataset to use',type=str, default='HAM10000')
parser.add_argument('-t', '--TYPE', help='abcd, char, short, doc',type=str, default='abcd')

args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
CNN_TO_USE = args.CNN
BATCH_SIZE = args.batch_size
EPOCHS = args.EPOCHS
EPOCHS_str = EPOCHS
TYPE_REPORT = args.TYPE
possible_reports = REPORTS[TYPE_REPORT]

EMBEDDING_bool = True
DATASET = args.DATASET

hidden_space_len = args.hidden_space

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

#OUTPUT_folder = args.output_folder
print("CREATE DIRECTORY WHERE MODELS WILL BE STORED")

#LOAD DATA
#TODO ADD PARAMETER
list_DATASETS = ['HAM10000']


flag_test = PHASE.test

MAIN_FOLDER = args.INPUT_FOLDER

DATA_FOLDER = MAIN_FOLDER + '/datasets/'
CSV_FOLDER = MAIN_FOLDER + '/csv_folder/'
#all_data = valid_dataset
target_img = None

test_dataset = utils_data.get_specific_dataset(DATA_FOLDER, DATASET, PHASE.all, CSV_FOLDER)
print(test_dataset.shape)


OUTPUT_FLD = args.output_folder + '/' +  DATASET + '/pubmed_embeddings/' 
os.makedirs(OUTPUT_FLD, exist_ok = True)

if (possible_reports is REPORTS.abcd):
	feat_imgs = OUTPUT_FLD + 'abcd/'
elif (possible_reports is REPORTS.char):
	feat_imgs = OUTPUT_FLD + 'char/'
elif (possible_reports is REPORTS.short):
	feat_imgs = OUTPUT_FLD + 'shorts/'
elif (possible_reports is REPORTS.doc):
	feat_imgs = OUTPUT_FLD + 'doc/'

elif (possible_reports is REPORTS.skingpt4_abcd):
	feat_imgs = OUTPUT_FLD + 'skingpt4_abcd/'
elif (possible_reports is REPORTS.skingpt4_char):
	feat_imgs = OUTPUT_FLD + 'skingpt4_char/'
elif (possible_reports is REPORTS.skingpt4_doc):
	feat_imgs = OUTPUT_FLD + 'skingpt4_doc/'
elif (possible_reports is REPORTS.skingpt4_p1):
	feat_imgs = OUTPUT_FLD + 'skingpt4_p1/'
elif (possible_reports is REPORTS.skingpt4_p2):
	feat_imgs = OUTPUT_FLD + 'skingpt4_p2/'

elif (possible_reports is REPORTS.dermlip_abcd):
	feat_imgs = OUTPUT_FLD + 'derm_1M_abcd/'
elif (possible_reports is REPORTS.dermlip_char):
	feat_imgs = OUTPUT_FLD + 'derm_1M_char/'
elif (possible_reports is REPORTS.dermlip_doc):
	feat_imgs = OUTPUT_FLD + 'derm_1M_doc/'
elif (possible_reports is REPORTS.dermlip_p1):
	feat_imgs = OUTPUT_FLD + 'derm_1M_p1/'
elif (possible_reports is REPORTS.dermlip_p2):
	feat_imgs = OUTPUT_FLD + 'derm_1M_p2/'

elif (possible_reports is REPORTS.medgemma_abcd):
	feat_imgs = OUTPUT_FLD + 'medgemma_abcd/'
elif (possible_reports is REPORTS.medgemma_char):
	feat_imgs = OUTPUT_FLD + 'medgemma_char/'
elif (possible_reports is REPORTS.medgemma_doc):
	feat_imgs = OUTPUT_FLD + 'medgemma_doc/'

os.makedirs(feat_imgs, exist_ok = True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32

params_valid_bag = {'batch_size': BATCH_SIZE,
		  'shuffle': False,
		  'drop_last': False,
		  'num_workers': 1}


testing_set_bag = Dataset_reports_only(test_dataset, possible_reports)
testing_generator_bag = data.DataLoader(testing_set_bag, **params_valid_bag)

print("initialize CNN")

bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
			
txt_encoder = BertModel.from_pretrained(bert_chosen, 
											output_attentions=True, 
											output_hidden_states=True,
											attn_implementation="eager")

txt_encoder.to(device)
txt_encoder.eval()

for param in txt_encoder.parameters():
	param.requires_grad = False

#lr = 1e-4
num_epochs = EPOCHS

#start loop

phase_str = 'test'
dataloader_iterator = iter(testing_generator_bag)
iterations = int(len(test_dataset) / BATCH_SIZE) + 1
		

#TODO change classes
filenames = np.empty((len(test_dataset)), dtype = 'object')
features = np.empty((len(test_dataset), 768))

for i in tqdm(range(iterations)):
		
	with torch.autocast(device_type='cuda', dtype=torch.float16):

		#print(', %d / %d ' % (i, iterations))
		try:
			tokens, batch_filenames = next(dataloader_iterator)
		except StopIteration:
			dataloader_iterator = iter(testing_generator_bag)
			tokens, batch_filenames = next(dataloader_iterator)

		input_txt = tokens.to(device)

		batch_filenames = list(batch_filenames)

		with torch.no_grad():
			# forward + backward + optimize
			
			input_ids_txt = input_txt["input_ids"].squeeze(1)# (batch_size, seq_length)
			attention_mask_txt = input_txt["attention_mask"].squeeze(1)  # (batch_size, seq_length)

			outputs_txt = txt_encoder(input_ids=input_ids_txt, attention_mask=attention_mask_txt)

			pooled_output = outputs_txt.pooler_output  # (batch_size, hidden_dim)
			
			pooled_output_np = pooled_output.cpu().data.numpy()

			del input_ids_txt, attention_mask_txt, outputs_txt

		for i in range(len(batch_filenames)):

			fname = batch_filenames[i]
			fname = batch_filenames[i].split('.')[0]
			
			features_filename = feat_imgs + fname + '.npy'
			try:
				features_to_save = pooled_output_np[i]
				with open(features_filename, 'wb') as f:
					np.save(f, features_to_save)
			except Exception as e:
				#print(e)
				pass
						