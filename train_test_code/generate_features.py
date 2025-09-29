import sys
import torch
from torch.utils import data
import numpy as np
import pandas as pd
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
from enum_multi import ALG, PHASE, TYPE_DATA, MOD, REPORTS
import utils_data
import random
from dataloader import Dataset_generate_features
from model import MultimodalArchitecture, MultimodalArchitecture_CLIP
from tqdm import tqdm

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
parser.add_argument('-o', '--overwrite', help='path folder where to store output model',type=str, default='True')
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

OVERWRITE_flag = args.overwrite
if (OVERWRITE_flag == 'True'):
	OVERWRITE_flag = True
else:
	OVERWRITE_flag = False

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
print("MODALITY: " + str(MODALITY))
print("TYPE_DOC: " + str(REPORTS_TRAINING))
print("flag_KEYWORDS: " + str(flag_KEYWORDS))

####PATH where find patches
#instance_dir = args.DATA_FOLDER


#OUTPUT_folder = args.output_folder
print("CREATE DIRECTORY WHERE MODELS WILL BE STORED")

OUTPUT_folder = 'MODEL_PATH'
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

N_CLASSES = 5

flag_all = PHASE.all
MAIN_FLD = 'PLACEHOLDER'
MAIN_FOLDER = MAIN_FLD + '/datasets/'
CSV_FOLDER = MAIN_FLD + '/csv_folder/'
#all_data = valid_dataset
target_img = None


features_dir = MAIN_FOLDER+DATASET+'/features_'+CNN_TO_USE+'/'+MODALITY+'/'
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

feat_imgs_dir = features_dir + 'images/'
os.makedirs(feat_imgs_dir, exist_ok = True)

feat_report_short_dir = features_dir + 'reports_shorts/'
os.makedirs(feat_report_short_dir, exist_ok = True)

feat_report_acbd_dir = features_dir + 'reports_abcd/'
os.makedirs(feat_report_acbd_dir, exist_ok = True)

feat_report_char_dir = features_dir + 'reports_char/'
os.makedirs(feat_report_char_dir, exist_ok = True)

feat_report_doc_dir = features_dir + 'reports_doc/'
os.makedirs(feat_report_doc_dir, exist_ok = True)


list_feats_dirs = [feat_imgs_dir, feat_report_short_dir, feat_report_char_dir, feat_report_doc_dir, feat_report_short_dir]

test_dataset = utils_data.get_specific_dataset_features(MAIN_FOLDER, DATASET, PHASE.all, None, overwrite_flag = OVERWRITE_flag, feat_flds = list_feats_dirs)

flag_embeddings = utils_data.get_flag_embeddings_exist(test_dataset[:,2], REPORTS.all)

print("flag_embeddings: " + str(flag_embeddings))

#print(test_dataset)

if (len(test_dataset) > 0):

	print(test_dataset.shape)

	#"""
	#MODEL DEFINITION
	#CNN BACKBONE
	if (CNN_TO_USE == 'HIPT'):
		fc_input_features = 512
	elif (CNN_TO_USE == 'ViT'):
		fc_input_features = 192
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
							TEMPERATURE = TEMPERATURE, patch_size=patch_size)

	#load weights
	model.load_state_dict(torch.load(model_weights_filename), strict=False)
	"""
	for name, param in model.base_encoder.named_parameters():
		param.requires_grad = False
	"""
	model.to(device)
	model.eval()




	params_valid_bag = {'batch_size': BATCH_SIZE,
			'shuffle': False,
			'num_workers': 1}


	testing_set_bag = Dataset_generate_features(test_dataset, PHASE.valid, 0.0, N_CLASSES, flag_embeddings, CNN_TO_USE)
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
				IDs, tokens_short, tokens_abcd, tokens_char, tokens_doc, labels, batch_filenames = next(dataloader_iterator)
			except StopIteration:
				dataloader_iterator = iter(testing_generator_bag)
				IDs, tokens_short, tokens_abcd, tokens_char, tokens_doc, labels, batch_filenames = next(dataloader_iterator)

			IDs = IDs.to(device, non_blocking=True)
			tokens_short = tokens_short.to(device, non_blocking=True)
			tokens_abcd = tokens_abcd.to(device, non_blocking=True)
			tokens_char = tokens_char.to(device, non_blocking=True)
			tokens_doc = tokens_doc.to(device, non_blocking=True)

			batch_filenames = list(batch_filenames)

			with torch.no_grad():
				# forward + backward + optimize
				
				try:
					_, cls_img, _, _ = model(IDs, None)
					cls_img_np = cls_img.cpu().data.numpy()
				except:
					pass

				try:
					_, _, _, cls_txt_short = model(None, tokens_short, flag_embeddings)
					cls_txt_short_np = cls_txt_short.cpu().data.numpy()
				except:
					pass

				try:
					_, _, _, cls_txt_abcd = model(None, tokens_abcd, flag_embeddings)
					cls_txt_abcd_np = cls_txt_abcd.cpu().data.numpy()
				except:
					pass

				try:
					_, _, _, cls_txt_char = model(None, tokens_char, flag_embeddings)
					cls_txt_char_np = cls_txt_char.cpu().data.numpy()
				except:
					pass

				try:
					_, _, _, cls_txt_doc = model(None, tokens_doc, flag_embeddings)
					cls_txt_doc_np = cls_txt_doc.cpu().data.numpy()
				except:
					pass

			for i in range(len(batch_filenames)):

				fname = batch_filenames[i]
				fname = batch_filenames[i].split('.')[0]
				

				if (MODALITY is not MOD.txt):
					#image
					features_filename = feat_imgs_dir + fname + '.npy'
					try:
						features_to_save = cls_img_np[i]
						with open(features_filename, 'wb') as f:
							np.save(f, features_to_save)
					except Exception as e:
						#print(e)
						pass

				#report short
				
				if (MODALITY is not MOD.img):
					#report short
					features_filename = feat_report_short_dir + fname + '.npy'
					try:
						features_to_save = cls_txt_short_np[i]
						with open(features_filename, 'wb') as f:
							np.save(f, features_to_save)
					except Exception as e:
						#print(e)
						pass

					#report abcd
					features_filename = feat_report_acbd_dir + fname + '.npy'
					try:
						features_to_save = cls_txt_abcd_np[i]
						with open(features_filename, 'wb') as f:
							np.save(f, features_to_save)
					except Exception as e:
						#print(e)
						pass
					
					#report char
					features_filename = feat_report_char_dir + fname + '.npy'
					try:
						features_to_save = cls_txt_char_np[i]
						with open(features_filename, 'wb') as f:
							np.save(f, features_to_save)
					except Exception as e:
						#print(e)
						pass
					
					#report doc
					features_filename = feat_report_doc_dir + fname + '.npy'
					try:
						features_to_save = cls_txt_doc_np[i]
						with open(features_filename, 'wb') as f:
							np.save(f, features_to_save)
					except Exception as e:
						#print(e)
						pass

