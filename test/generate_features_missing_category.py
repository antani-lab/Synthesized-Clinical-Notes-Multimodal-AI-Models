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
from dataloader import ImbalancedDatasetSampler, Dataset_generate_features_specific, filter_labels
from model import MultimodalArchitecture, MultimodalArchitecture_CLIP
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
from metrics_multiclass import accuracy_score, kappa_score, f1_scores, precisions, recalls
from tqdm import tqdm
import utils_txt

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
parser.add_argument('-b', '--batch_size', help='batch size bag level',type=int, default=32)
parser.add_argument('-o', '--output_folder', help='path folder where to store output model',type=str, default='')
parser.add_argument('-d', '--DATASET', help='dataset to use',type=str, default='HAM10000')
parser.add_argument('-w', '--PROBLEM', help='type of PROBLEM:binary (nevus vs melanoma), multiclass (sebo vs nevus vs basal vs melanoma)',type=str, default='multiclass')
parser.add_argument('-m', '--MODALITY', help='modalities to use: img, txt, multimodal',type=str, default='multimodal')
parser.add_argument('-t', '--TYPE', help='reports training: all, abcd, char, short, doc, meta, random',type=str, default='all')
parser.add_argument('-k', '--KEYWORDS', help='train on keywords',type=str, default='False')
parser.add_argument('-a', '--AUGMENTATION', help='report augmentation',type=str, default='False')
parser.add_argument('-i', '--INPUT', help='data to analyze',type=str, default='abcd')
parser.add_argument('-p', '--weights', help='algorithm for pre-trained weights (PanDerm)',type=str, default='')
parser.add_argument('-f', '--DATA_FOLDER', help='path of the folder where to images are stored',type=str, default='')

args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
CNN_TO_USE = args.CNN
BATCH_SIZE = args.batch_size
PROBLEM = args.PROBLEM
MODALITY = args.MODALITY

EMBEDDING_bool = True
DATASET = args.DATASET
PANDERM_FOLDER = args.weights

hidden_space_len = args.hidden_space

REPORTS_TRAINING = args.TYPE
REPORT_AUGMENTATION = args.AUGMENTATION

if (REPORT_AUGMENTATION == 'True'):
	REPORT_AUGMENTATION = True
else:
	REPORT_AUGMENTATION = False

INPUT_DATA = args.INPUT

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
print("CNN used: " + str(CNN_TO_USE))
print("DATASET: " + str(DATASET))
print("N_EXP: " + str(N_EXP_str))
print("MODALITY: " + str(MODALITY))
####PATH where find patches
#instance_dir = args.DATA_FOLDER


#OUTPUT_folder = args.output_folder
print("CREATE DIRECTORY WHERE MODELS WILL BE STORED")


OUTPUT_folder = args.output_folder
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

#path model file
model_weights_filename = models_path+'model.pt'

N_CLASSES = 8

flag_all = PHASE.all
MAIN_FOLDER = args.DATA_FOLDER

DATA_FOLDER = MAIN_FOLDER + '/datasets/'
CSV_FOLDER = MAIN_FOLDER + '/csv_folder/'
#all_data = valid_dataset
target_img = None

features_dir = DATA_FOLDER+DATASET+'/features_'+CNN_TO_USE+'/'+MODALITY+'/'
os.makedirs(features_dir, exist_ok=True)

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

if (INPUT_DATA == "images"):
	feat_dir = features_dir + 'images/'
	os.makedirs(feat_dir, exist_ok = True)
elif (INPUT_DATA == "short"):
	feat_dir = features_dir + 'reports_shorts/'
	os.makedirs(feat_dir, exist_ok = True)
else:
	feat_dir = features_dir + 'reports_'+INPUT_DATA+'/'
	os.makedirs(feat_dir, exist_ok = True)

print(feat_dir)
features_dirs = [feat_dir]

test_dataset = utils_data.get_specific_dataset_features(DATA_FOLDER, DATASET, PHASE.all, CSV_FOLDER = CSV_FOLDER, overwrite_flag = False, feat_flds = features_dirs)
#test_dataset = utils_data.get_specific_dataset(MAIN_FOLDER, DATASET, PHASE.all, None, overwrite_flag = False, feat_flds = None)
print(test_dataset.shape)

#print(test_dataset)

flag_embeddings = True

print("flag_embeddings: " + str(flag_embeddings))

print(test_dataset.shape)

if (len(test_dataset) > 0):

	print(test_dataset.shape)

	#"""
	#MODEL DEFINITION
	#CNN BACKBONE
	model_weights_filename_pre_trained_img = None
	if (CNN_TO_USE == 'HIPT'):
		#fc_input_features = 512
		fc_input_features = 384
		
	elif (CNN_TO_USE == 'ViT'):
		fc_input_features = 192

	elif (CNN_TO_USE == 'PanDerm'):
		model_weights_filename_pre_trained_img = PANDERM_FOLDER + "panderm_bb_data6_checkpoint-499.pth"
		fc_input_features = 768

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

	print("initialize CNN")

	input_dim = fc_input_features
	hidden_dim = hidden_space_len
	output_dim = N_CLASSES
	TEMPERATURE = 0.07
	patch_size = 16

	model = MultimodalArchitecture(device, CNN_TO_USE, in_dim = input_dim, 
							out_dim = output_dim, 
							intermediate_dim = hidden_dim,
							TEMPERATURE = TEMPERATURE, patch_size=patch_size, pretrained_path=model_weights_filename_pre_trained_img)

	#load weights
	model.load_state_dict(torch.load(model_weights_filename), strict=False)
	"""
	for name, param in model.base_encoder.named_parameters():
		param.requires_grad = False
	"""
	model.to(device)
	model.eval()


	if (REPORTS[INPUT_DATA] is REPORTS.images):
		num_workers = 1
	else:
		num_workers = 0

	params_valid_bag = {'batch_size': BATCH_SIZE,
			'shuffle': False,
			'drop_last': False,
			'num_workers': num_workers}


	testing_set_bag = Dataset_generate_features_specific(test_dataset, PHASE.valid, 0.0, N_CLASSES, REPORTS[INPUT_DATA], flag_embeddings, CNN_TO_USE)
	testing_generator_bag = data.DataLoader(testing_set_bag, **params_valid_bag)

	phase_str = 'test'
	dataloader_iterator = iter(testing_generator_bag)
	iterations = int(len(test_dataset) / BATCH_SIZE) + 1

	features_imgs = []

	MODALITY = MOD[MODALITY]

	for i in tqdm(range(iterations)):
		
		with torch.autocast(device_type='cuda', dtype=torch.float16):

			#print(', %d / %d ' % (i, iterations))
			try:
				IDs, batch_filenames = next(dataloader_iterator)
			except StopIteration:
				dataloader_iterator = iter(testing_generator_bag)
				IDs, batch_filenames = next(dataloader_iterator)

			IDs = IDs.to(device, non_blocking=True)
			
			batch_filenames = list(batch_filenames)

			with torch.no_grad():
				# forward + backward + optimize
				
				if (REPORTS[INPUT_DATA] is REPORTS.images):
					try:
						_, cls_img, _, _ = model(IDs, None)
						cls_np = cls_img.cpu().data.numpy()
					except Exception as ex:
						print(ex)
						pass
				
				else:
					try:
						_, _, _, cls_txt = model(None, IDs, flag_embeddings)
						cls_np = cls_txt.cpu().data.numpy()
					except Exception as ex:
						print(ex)
						pass


			for i in range(len(batch_filenames)):

				fname = batch_filenames[i]
				fname = batch_filenames[i].split('.')[0]
				
				#image
				features_filename = feat_dir + fname + '.npy'

				#print(features_filename)
				try:
					features_to_save = cls_np[i]
					with open(features_filename, 'wb') as f:
						np.save(f, features_to_save)
				except Exception as ex:
					print(ex)
					pass

