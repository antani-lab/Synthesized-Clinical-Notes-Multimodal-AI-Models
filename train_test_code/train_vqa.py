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
from enum_multi import ALG, PHASE, TYPE_DATA, MOD
import utils_data
import random
from dataloader import ImbalancedDatasetSampler, Dataset_instance, filter_labels, Dataset_instance_VQA_json
from model import MultiCategoryVQAModel
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
parser.add_argument('-c', '--CNN', help='cnn_to_use',type=str, default='densenet121')
parser.add_argument('-e', '--EPOCHS', help='epochs to train',type=int, default=10)
parser.add_argument('-z', '--hidden_space', help='hidden_space_size',type=int, default=128)
parser.add_argument('-b', '--batch_size', help='batch size bag level',type=int, default=128)
parser.add_argument('-o', '--output_folder', help='path folder where to store output model',type=str, default='')
parser.add_argument('-s', '--self_supervised', help='path folder with pretrained network',type=str, default='True')
parser.add_argument('-w', '--weights', help='algorithm for pre-trained weights',type=str, default='simCLR')
parser.add_argument('-i', '--DATA_FOLDER', help='path of the folder where to images are stored',type=str, default='')
parser.add_argument('-m', '--MODALITY', help='modalities to use: img, txt, multimodal',type=str, default='img')
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
WEIGHTS = args.weights
WEIGHTS_str = WEIGHTS
MODALITY = args.MODALITY

PROBLEM = args.PROBLEM
EMBEDDING_bool = True

hidden_space_len = args.hidden_space

flag_ssl = args.self_supervised
if (flag_ssl == 'True'):
	flag_ssl = True
else:
	flag_ssl = False

flag_ssl = True

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


#model_weights_filename_pre_trained = args.self_supervised

#OUTPUT_folder = args.output_folder
print("CREATE DIRECTORY WHERE MODELS WILL BE STORED")

OUTPUT_folder = '/PLACEHOLDER/model_weights/'
models_path = OUTPUT_folder
os.makedirs(models_path, exist_ok=True)
models_path = models_path + 'multimodal/'
os.makedirs(models_path, exist_ok=True)
models_path = models_path + 'VQA/'
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


MAIN_FOLDER = 'PLACEHOLDER'
MAIN_FOLDER = MAIN_FOLDER + '/datasets/'
CSV_FOLDER = MAIN_FOLDER + '/csv_folder/'

questions_dir = models_path + 'questions_VQA/'
questions_dir = questions_dir.replace('/VQA/','/')

os.makedirs(questions_dir, exist_ok = True)


####load questions
fname_questions = questions_dir + 'cls_questions.npy'
with open(fname_questions, 'rb') as f:
	cls_questions = np.load(f)

fname_answers = questions_dir + 'cls_answers.npy'
with open(fname_answers, 'rb') as f:
	cls_answers = np.load(f)


#LOAD DATA
#TODO ADD PARAMETER
#list_DATASETS_train = ['HAM10000','BCN20000']
#list_DATASETS_valid = ['HAM10000','BCN20000']

list_DATASETS_train = ['BCN20000','derm12345', 'Derm7pt', 'DermNet']

list_DATASETS_valid = ['BCN20000','derm12345', 'Derm7pt', 'DermNet']



#list_DATASETS = ['HAM10000']

flag_train = PHASE.train
flag_valid = PHASE.valid


train_dataset, train_questions, train_answers = utils_data.get_instances_vqa_json(MAIN_FOLDER, list_DATASETS_train, flag_train, CNN_TO_USE, MODALITY, N_EXP_str, CSV_FOLDER, REPORTS_TRAINING = REPORTS_TRAINING, flag_KEYWORDS = False, REPORT_AUGMENTATION = REPORT_AUGMENTATION)
valid_dataset, valid_questions, valid_answers = utils_data.get_instances_vqa_json(MAIN_FOLDER, list_DATASETS_valid, flag_valid, CNN_TO_USE, MODALITY, N_EXP_str, CSV_FOLDER, REPORTS_TRAINING = REPORTS_TRAINING, flag_KEYWORDS = False, REPORT_AUGMENTATION = REPORT_AUGMENTATION)

#print(train_dataset)
#print(train_dataset)
#print(train_dataset)
#train_dataset = train_dataset[:128]
#valid_dataset = valid_dataset[:128]

MODALITY = MOD[MODALITY]





N_CLASSES = len(cls_answers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = 128
intermediate_dim = 128

model = MultiCategoryVQAModel(input_dim, intermediate_dim, N_CLASSES)
model.eval()
model.to(device)

print(model)

pin_memory = True
pin_memory = False
n_workers = 1

#"""
params_train_bag = {'batch_size': BATCH_SIZE,
		'pin_memory': pin_memory,
		'shuffle': True,
		'drop_last':True,
		}
#"""

params_valid_bag = {'batch_size': BATCH_SIZE,
					 'drop_last':True,
		  			'shuffle': False}

#training_set_bag = Dataset_instance_txt_generation(train_dataset, PAD_ID, MAX_LEN_SEQ)
training_set_bag = Dataset_instance_VQA_json(train_dataset, train_questions, train_answers, cls_questions, cls_answers)
training_generator_bag = data.DataLoader(training_set_bag, **params_train_bag)

#validation_set_bag = Dataset_instance_txt_generation(valid_dataset, PAD_ID, MAX_LEN_SEQ)
validation_set_bag = Dataset_instance_VQA_json(valid_dataset, valid_questions, valid_answers, cls_questions, cls_answers)
validation_generator_bag = data.DataLoader(validation_set_bag, **params_valid_bag)


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

criterion_representation_pairwise = loss_functions.ContrastiveLoss()
criterion_representation_cosine = torch.nn.CosineEmbeddingLoss()
criterion_representation_rmse = torch.nn.L1Loss()
#criterion_contrastive = loss_functions.NT_Xent(batch_size = BATCH_SIZE, temperature = TEMPERATURE)
#criterion_contrastive = loss_functions.NT_Xent(batch_size = BATCH_SIZE, temperature = TEMPERATURE)
criterion_contrastive_infonce = loss_functions.InfoNCE(temperature = 0.07)
criterion_contrastive_ntxent = loss_functions.NT_Xent(batch_size = BATCH_SIZE, temperature = 0.5)
criterion_classification = torch.nn.CrossEntropyLoss()

criterion_contrastive_infonce = loss_functions.InfoNCE_supervised(temperature = 0.07)
criterion_contrastive_ntxent = loss_functions.NT_Xent_supervised(batch_size = BATCH_SIZE, temperature = 0.5)


label_embedding_pos = torch.tensor([1]).to(device)

def model_forward(epoch, phase = 'train'):

	if (phase is PHASE.train):
		phase_str = 'train'
		dataloader_iterator = iter(training_generator_bag)
		iterations = iterations = int(len(train_dataset) / BATCH_SIZE) + 1
		model.train()
	

	elif (phase is PHASE.valid):
		phase_str = 'valid'
		dataloader_iterator = iter(validation_generator_bag)
		iterations = iterations = int(len(valid_dataset) / BATCH_SIZE) + 1
		model.eval()
	
	
	total_loss = 0.0

	representation_loss_rmse = 0.0
	representation_loss_cosine = 0.0
	representation_pairwise = 0.0
	representation_ntxent_loss = 0.0
	representation_infonce_loss = 0.0
	classification_loss = 0.0

	progress_bar = tqdm(range(iterations), desc='samples', position=0) 

	img_log = tqdm(total=0, position=1, bar_format='{desc}')
	rmse_log = tqdm(total=0, position=2, bar_format='{desc}')
	cosine_log = tqdm(total=0, position=3, bar_format='{desc}')
	pairwise_log = tqdm(total=0, position=4, bar_format='{desc}')
	ntxent_log = tqdm(total=0, position=5, bar_format='{desc}')
	infonce_log = tqdm(total=0, position=6, bar_format='{desc}')
	tot_log = tqdm(total=0, position=7, bar_format='{desc}')

	for i in progress_bar:

		with torch.autocast(device_type='cuda', dtype=torch.float16):
			try:
				X, Q, A, y, _ = next(dataloader_iterator)
			except StopIteration:
				dataloader_iterator = iter(training_generator_bag)
				X, Q, A, y, _ = next(dataloader_iterator)

			X = X.to(device, non_blocking=True)
			Q = Q.to(device, non_blocking=True)
			A = A.to(device, non_blocking=True)
			y = y.to(device, non_blocking=True)

			model.zero_grad(set_to_none=True)
			optimizer_CNN.zero_grad(set_to_none=True)

			logits, output_class = model(X, Q)
						
			#print(logits_0.shape, logits_1.shape, logits_2.shape, logits_3.shape)

			loss_representation_cosine = criterion_representation_cosine(logits, A, label_embedding_pos)
			representation_loss_cosine = representation_loss_cosine + ((1 / (i+1)) * (loss_representation_cosine.item() - representation_loss_cosine))
			
			loss_representation_rmse = criterion_representation_rmse(logits, A)
			representation_loss_rmse = representation_loss_rmse + ((1 / (i+1)) * (loss_representation_rmse.item() - representation_loss_rmse))

			loss_representation_pairwise = criterion_representation_pairwise(logits, A)
			representation_pairwise = representation_pairwise + ((1 / (i+1)) * (loss_representation_pairwise.item() - representation_pairwise))

			try:
				loss_infonce = criterion_contrastive_infonce(logits, A)
			except:
				loss_infonce = criterion_contrastive_infonce(logits, A, y)
			representation_infonce_loss = representation_infonce_loss + ((1 / (i+1)) * (loss_infonce.item() - representation_infonce_loss))
			
			try:
				loss_ntxent = criterion_contrastive_ntxent(logits, A)
			except:
				loss_ntxent = criterion_contrastive_ntxent(logits, A, y)

			representation_ntxent_loss = representation_ntxent_loss + ((1 / (i+1)) * (loss_ntxent.item() - representation_ntxent_loss))

			loss_classification = criterion_classification(output_class, y)			
			classification_loss = classification_loss + ((1 / (i+1)) * (loss_classification.item() - classification_loss))

			#loss = loss_representation_cosine + loss_representation_rmse + loss_classification + loss_infonce #+ loss_ntxent #+ loss_representation_pairwise
			loss = loss_representation_cosine + loss_ntxent + loss_representation_rmse + loss_classification  #+ loss_infonce #+ loss_representation_pairwise
			total_loss = total_loss + ((1 / (i+1)) * (loss.item() - total_loss))#
			
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

			img_log.set_description_str(f'img: {classification_loss}')
			rmse_log.set_description_str(f'rmse: {representation_loss_rmse}')
			cosine_log.set_description_str(f'cosine: {representation_loss_cosine}')
			pairwise_log.set_description_str(f'pairwise: {representation_pairwise}')
			ntxent_log.set_description_str(f'nt_xent: {representation_ntxent_loss}')
			infonce_log.set_description_str(f'infonce: {representation_infonce_loss}')
			tot_log.set_description_str(f'f1_score: {total_loss}')

	return total_loss


epoch = 0

best_loss = 100000.0

EARLY_STOP_NUM = 10
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
		
	else:
		early_stop_cont = early_stop_cont+1
	
	
	epoch = epoch + 1

torch.cuda.empty_cache()