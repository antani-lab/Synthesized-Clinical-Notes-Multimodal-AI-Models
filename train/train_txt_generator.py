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
from enum_multi import ALG, PHASE, TYPE_DATA, MOD, REPORTS
import utils_data
import random
from dataloader import ImbalancedDatasetSampler, Dataset_instance, filter_labels, Dataset_instance_txt_generation
from model import MultimodalArchitecture, MultimodalArchitecture_CLIP, FeatureToTextDecoder
from transformers import AutoTokenizer
from tqdm import tqdm
from metrics_multiclass import accuracy_score, kappa_score, f1_scores, precisions, recalls
import loss_functions

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
parser.add_argument('-e', '--EPOCHS', help='epochs to train',type=int, default=20)
parser.add_argument('-z', '--hidden_space', help='hidden_space_size',type=int, default=128)
parser.add_argument('-b', '--batch_size', help='batch size bag level',type=int, default=32)
parser.add_argument('-o', '--output_folder', help='path folder where to store output model',type=str, default='')
parser.add_argument('-s', '--FEATURE_DIR', help='path folder with pretrained network',type=str, default='True')
parser.add_argument('-w', '--weights', help='algorithm for pre-trained weights',type=str, default='simCLR')
parser.add_argument('-i', '--DATA_FOLDER', help='path of the folder where to images are stored',type=str, default='')
parser.add_argument('-m', '--MODALITY', help='modalities to use: img, txt, multimodal',type=str, default='multimodal')
parser.add_argument('-p', '--PROBLEM', help='type of PROBLEM:binary (nevus vs melanoma), multiclass (sebo vs nevus vs basal vs melanoma)',type=str, default='multiclass')
parser.add_argument('-t', '--TYPE', help='reports training: all, abcd, char, short, doc, meta, random',type=str, default='all')
parser.add_argument('-k', '--KEYWORDS', help='train on keywords',type=str, default='False')
parser.add_argument('-a', '--AUGMENTATION', help='report augmentation',type=str, default='False')

args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
CNN_TO_USE = args.CNN
BATCH_SIZE = args.batch_size
BATCH_SIZE_str = str(BATCH_SIZE)
EPOCHS = args.EPOCHS
EPOCHS_str = EPOCHS
PANDERM_FOLDER = args.weights

MODALITY = args.MODALITY

PROBLEM = args.PROBLEM
EMBEDDING_bool = True

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


seed = N_EXP % 10
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
print("BATCH_SIZE: " + str(BATCH_SIZE_str))

####PATH where find patches


#OUTPUT_folder = args.output_folder
print("CREATE DIRECTORY WHERE MODELS WILL BE STORED")

OUTPUT_folder = args.output_folder
models_path = OUTPUT_folder
os.makedirs(models_path, exist_ok=True)
models_path = models_path + 'multimodal/'
os.makedirs(models_path, exist_ok=True)
models_path = models_path + 'txt_generator/'
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



pre_trained_model = args.FEATURE_DIR
pre_trained_model = pre_trained_model + REPORTS_TRAINING + '/'

if (flag_KEYWORDS):
	pre_trained_model = pre_trained_model + 'keywords/'
else:
	pre_trained_model = pre_trained_model + 'no_keywords/'

if (REPORT_AUGMENTATION):
	pre_trained_model = pre_trained_model + 'report_augmentation/'
else:
	pre_trained_model = pre_trained_model + 'no_report_augmentation/'

pre_trained_model = pre_trained_model + PROBLEM + '/'
pre_trained_model = pre_trained_model + CNN_TO_USE+'/'
pre_trained_model = pre_trained_model + 'N_EXP_'+N_EXP_str+'/model.pt'


tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

# Special tokens
PAD_ID = tokenizer.pad_token_id
BOS_ID = tokenizer.cls_token_id  # or tokenizer.bos_token_id if defined
EOS_ID = tokenizer.sep_token_id  # or tokenizer.eos_token_id if defined


#LOAD DATA
#TODO ADD PARAMETER
#list_DATASETS_train = ['HAM10000','BCN20000']
#list_DATASETS_valid = ['HAM10000','BCN20000']

list_DATASETS_train = ['BCN20000','derm12345', 'Derm7pt', 'DermNet','MRA_MIDAS','FLUO_SC']
list_DATASETS_valid = ['BCN20000','derm12345', 'Derm7pt', 'DermNet','MRA_MIDAS','FLUO_SC']
#list_DATASETS_valid = ['BCN20000']

#list_DATASETS = ['HAM10000']

flag_train = PHASE.train
flag_valid = PHASE.valid

MAIN_FOLDER = args.DATA_FOLDER

DATA_FOLDER = MAIN_FOLDER + 'datasets/'
CSV_FOLDER = MAIN_FOLDER + '/csv_folder/'

#all_data = valid_dataset
train_dataset = utils_data.get_instances_txt_generation(DATA_FOLDER, list_DATASETS_train, flag_train, CNN_TO_USE, MODALITY, N_EXP_str, CSV_FOLDER, REPORTS_TRAINING = REPORTS_TRAINING, flag_KEYWORDS = flag_KEYWORDS)
valid_dataset = utils_data.get_instances_txt_generation(DATA_FOLDER, list_DATASETS_valid, flag_valid, CNN_TO_USE, MODALITY, N_EXP_str, CSV_FOLDER, REPORTS_TRAINING = REPORTS_TRAINING, flag_KEYWORDS = flag_KEYWORDS)

#"""
list_DATASETS_out = ['HAM10000', 'SKINL2',
            'Fitzpatrick17k', 'Hospital_Italiano_Buenos_Aires', 'PAD_UFES_20',
            'SD198']

list_DATASETS_out = []

flag_all = PHASE.all
rest_dataset = utils_data.get_instances_txt_generation(MAIN_FOLDER, list_DATASETS_out, flag_all, CNN_TO_USE, MODALITY, N_EXP_str, CSV_FOLDER, REPORTS_TRAINING = REPORTS_TRAINING, flag_KEYWORDS = flag_KEYWORDS)

train_dataset = np.append(train_dataset, rest_dataset, axis = 0)
#"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_LEN_SEQ = 256

AUGMENT_PROB_THRESHOLD = 0.5
prob = AUGMENT_PROB_THRESHOLD

sampler = ImbalancedDatasetSampler

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
	
input_dim = fc_input_features
hidden_dim = hidden_space_len
output_dim = 8
intermediate_layers = True
if (MODALITY is MOD.InfoNCE_supervised or MOD.InfoNCE):
	TEMPERATURE = 0.07 #best infonce
else:
	TEMPERATURE = 0.5

pre_trained_encoder = MultimodalArchitecture(device, CNN_TO_USE, in_dim = input_dim, 
						out_dim = output_dim, 
						intermediate_dim = hidden_dim,
						TEMPERATURE = TEMPERATURE, pretrained_path=model_weights_filename_pre_trained_img, patch_size=16)


pre_trained_encoder.load_state_dict(torch.load(pre_trained_model), strict = False) #best infonce (temp 0.07)
pre_trained_encoder.eval()
pre_trained_encoder = pre_trained_encoder.to(device)

pin_memory = True
pin_memory = False
if (REPORT_AUGMENTATION):
	n_workers = 0
else:
	n_workers = 1

n_workers = 1
#"""
params_train_bag = {'batch_size': BATCH_SIZE,
		'pin_memory': pin_memory,
		'sampler': sampler(train_dataset),
		'num_workers': n_workers,
		'drop_last':True} #for contrastive loss
#"""
"""
params_train_bag = {'batch_size': BATCH_SIZE,
		'pin_memory': pin_memory,
		'shuffle': True,
		'num_workers': n_workers}
"""

params_valid_bag = {'batch_size': BATCH_SIZE,
					 'drop_last':True,
		  			'shuffle': False}

possible_reports = REPORTS[REPORTS_TRAINING]

#training_set_bag = Dataset_instance_txt_generation(train_dataset, PAD_ID, MAX_LEN_SEQ)
training_set_bag = Dataset_instance_txt_generation(train_dataset, possible_reports, PAD_ID, pre_trained_encoder, 0.75, PHASE.train, device, MAX_LEN_SEQ, only_imgs = False, flag_augment_reports = REPORT_AUGMENTATION, flag_augment_images = False)
training_generator_bag = data.DataLoader(training_set_bag, **params_train_bag)

#validation_set_bag = Dataset_instance_txt_generation(valid_dataset, PAD_ID, MAX_LEN_SEQ)
validation_set_bag = Dataset_instance_txt_generation(valid_dataset, possible_reports, PAD_ID, pre_trained_encoder, 1.0, PHASE.valid, device, MAX_LEN_SEQ)
validation_generator_bag = data.DataLoader(validation_set_bag, **params_valid_bag)

print("initialize CNN")


model = FeatureToTextDecoder(feature_dim=128, hidden_dim=128, num_layers=4, nhead=4, dropout=0.1, max_len=MAX_LEN_SEQ)

model.to(device)
#model.eval()

#print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters CNN.')

num_epochs = EPOCHS

print("initialize hyperparameters")
import torch.optim as optim


#previous setup
lr = 1e-4
wt_decay = 1e-5

optimizer_CNN = optim.Adam(model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wt_decay, amsgrad=True)

scaler = torch.amp.GradScaler()

criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)


def model_forward(epoch, phase = 'train'):
	
	if (phase is PHASE.train):
		phase_str = 'train'
		generator = training_generator_bag
		dataloader_iterator = iter(generator)
		iterations = iterations = int(len(train_dataset) / BATCH_SIZE) + 1
		model.train()
	

	elif (phase is PHASE.valid):
		phase_str = 'valid'
		generator = validation_generator_bag
		dataloader_iterator = iter(generator)
		iterations = iterations = int(len(valid_dataset) / BATCH_SIZE) + 1
		model.eval()
	
	
	img_loss = 0.0
	progress_bar = tqdm(range(iterations), desc='samples', position=0) 

	img_log = tqdm(total=0, position=1, bar_format='{desc}')

	for i in progress_bar:

		with torch.autocast(device_type='cuda', dtype=torch.float16):

			try:
				features, input_ids, target_ids, _ = next(dataloader_iterator)
			except StopIteration:
				dataloader_iterator = iter(generator)
				features, input_ids, target_ids, _ = next(dataloader_iterator)

			input_ids = input_ids.squeeze(1)
			target_ids = target_ids.squeeze(1)

			input_ids = input_ids[:, :-1]     # shape (B, S-1)
			target_ids = target_ids[:, 1:]    # shape (B, S-1)

			features = features.to(device)
			input_ids = input_ids.to(device)
			target_ids = target_ids.to(device)
			
			#print(features.shape, input_ids.shape, target_ids.shape)
			logits = model(features, input_ids)
			loss = criterion(logits.reshape(-1, model.vocab_size), target_ids.reshape(-1))

			if (phase is PHASE.train):
				#"""
				scaler.scale(loss).backward()
				scaler.step(optimizer_CNN)
				scaler.update()
				optimizer_CNN.zero_grad(set_to_none=True)
				model.zero_grad(set_to_none=True)
				#"""
				"""
				loss.backward()
				optimizer_CNN.step()
				optimizer_CNN.zero_grad(set_to_none=True)
				model.zero_grad(set_to_none=True)
				"""
				
				
			
			img_loss = img_loss + ((1 / (i+1)) * (loss.item() - img_loss))

			img_log.set_description_str(f'img: {img_loss}')

	print("end phase " + phase_str + ", loss: " + str(img_loss))
	return img_loss


epoch = 0

best_loss = 100000.0

EARLY_STOP_NUM = 5
early_stop_cont = 0
epoch = 0

batch_size_instance = int(BATCH_SIZE_str)

while (epoch<num_epochs and early_stop_cont<EARLY_STOP_NUM):
		
	#train
	phase = PHASE['train']
	train_loss = model_forward(epoch, phase = phase)

	phase = PHASE['valid']
	valid_loss = model_forward(epoch, phase = phase)
	
	if (best_loss>valid_loss):
		early_stop_cont = 0
		print ("=> Saving a new best model")
		print("previous loss : " + str(best_loss) + ", new loss function: " + str(valid_loss))
		best_loss = valid_loss
	
		try:
			torch.save(model.state_dict(), model_weights_filename,_use_new_zipfile_serialization=False)
		except:
			try:
				torch.save(model.state_dict(), model_weights_filename)
			except:
				torch.save(model, model_weights_filename)
		
		print(model_weights_filename)
	else:
		early_stop_cont = early_stop_cont+1
	
	
	epoch = epoch + 1

torch.cuda.empty_cache()