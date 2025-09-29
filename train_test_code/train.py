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
from dataloader import ImbalancedDatasetSampler, Dataset_instance, filter_labels, Dataset_instance_concept
from model import MultimodalArchitecture, MultimodalArchitecture_CLIP
from tqdm import tqdm
from metrics_multiclass import kappa_score, f1_scores, precisions, recalls
import loss_functions
import utils_concept_extraction

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
REPORTS_TRAINING = args.TYPE
REPORT_AUGMENTATION = args.AUGMENTATION

if (REPORT_AUGMENTATION == 'True'):
	REPORT_AUGMENTATION = True
else:
	REPORT_AUGMENTATION = False

PROBLEM = args.PROBLEM
EMBEDDING_bool = True

hidden_space_len = args.hidden_space

flag_ssl = args.self_supervised
if (flag_ssl == 'True'):
	flag_ssl = True
else:
	flag_ssl = False

flag_ssl = True
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

OUTPUT_folder = 'PLACEHOLDERPATH'
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

MODALITY = MOD[MODALITY]

#LOAD DATA
#TODO ADD PARAMETER

list_DATASETS_train = ['BCN20000','derm12345', 'Derm7pt', 'DermNet']
list_DATASETS_valid = ['BCN20000','derm12345', 'Derm7pt', 'DermNet']

flag_train = PHASE.train
flag_valid = PHASE.valid

MAIN_FLD = 'PLACEHOLDER'
MAIN_FOLDER = MAIN_FLD + '/datasets/'
CSV_FOLDER = MAIN_FLD + '/csv_folder/'

#all_data = valid_dataset
train_dataset = utils_data.get_instances_paths_from_bags(MAIN_FOLDER, list_DATASETS_train, flag_train, CSV_FOLDER)
valid_dataset = utils_data.get_instances_paths_from_bags(MAIN_FOLDER, list_DATASETS_valid, flag_valid, CSV_FOLDER)

flag_embeddings_train = False
flag_embeddings_valid = False

#print(train_dataset)
#train_dataset = train_dataset[:1001]
#valid_dataset = valid_dataset[:1001]

#aggregate classes
N_CLASSES = 1
#N_CLASSES = 3
N_CLASSES = 7

if (PROBLEM == 'binary'):
	#"""
	TOT_CLASSES = 7
	set_to_filter = [0,1,3,4,5]
	train_dataset = filter_labels(train_dataset, set_to_filter) 
	valid_dataset = filter_labels(valid_dataset, set_to_filter) 

	N_CLASSES = TOT_CLASSES - len(set_to_filter)
	#"""

elif (PROBLEM == 'multiclass'):
	#"""
	TOT_CLASSES = 7
	#set_to_filter = [1,3,4]
	set_to_filter = [1,3]
	train_dataset = filter_labels(train_dataset, set_to_filter) 
	valid_dataset = filter_labels(valid_dataset, set_to_filter) 

	N_CLASSES = TOT_CLASSES - len(set_to_filter)
	#"""

unique, counts = np.unique(train_dataset[:,1], return_counts=True)
print("train_dataset: " + str(len(train_dataset[:,1])) + ", " + str(dict(zip(unique, counts))))

unique, counts = np.unique(valid_dataset[:,1], return_counts=True)
print("valid_dataset: " + str(len(valid_dataset[:,1])) + ", " + str(dict(zip(unique, counts))))

if(N_CLASSES == 2):
	N_CLASSES = 1
	
print(N_CLASSES)
#match classes



#train_dataset = train_dataset[:128]
#valid_dataset = valid_dataset[:128]

print(train_dataset.shape)
print(valid_dataset.shape)

print("train")
unique, counts = np.unique(train_dataset[:,1], return_counts=True)
print(dict(zip(unique, counts)))

print("valid")
unique, counts = np.unique(valid_dataset[:,1], return_counts=True)
print(dict(zip(unique, counts)))

target_img = None

#MODEL DEFINITION
#CNN BACKBONE

if (CNN_TO_USE == 'HIPT'):
	#fc_input_features = 512
	fc_input_features = 384
	
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


AUGMENT_PROB_THRESHOLD = 0.5
prob = AUGMENT_PROB_THRESHOLD

sampler = ImbalancedDatasetSampler


pin_memory = True
pin_memory = False
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

if (MODALITY is MOD.img):
	COMPONENTS_TO_USE = [COMPONENTS.images]
elif (MODALITY is MOD.txt):
	COMPONENTS_TO_USE = [COMPONENTS.reports]
else:
	COMPONENTS_TO_USE = [COMPONENTS.images, COMPONENTS.reports]

if (flag_KEYWORDS):
	COMPONENTS_TO_USE.append(COMPONENTS.keywords)

print(COMPONENTS_TO_USE)

training_set_bag = Dataset_instance_concept(train_dataset, PHASE.train, prob, N_CLASSES, possible_reports, components = COMPONENTS_TO_USE, CNN_TO_USE = CNN_TO_USE)
training_generator_bag = data.DataLoader(training_set_bag, **params_train_bag)

validation_set_bag = Dataset_instance_concept(valid_dataset, PHASE.valid, 0.0, N_CLASSES, possible_reports, components = COMPONENTS_TO_USE, CNN_TO_USE = CNN_TO_USE)
validation_generator_bag = data.DataLoader(validation_set_bag, **params_valid_bag)

print("initialize CNN")

input_dim = fc_input_features
hidden_dim = hidden_space_len
output_dim = N_CLASSES


intermediate_layers = True
if (MODALITY is MOD.InfoNCE_supervised or MOD.InfoNCE):
	TEMPERATURE = 0.07 #best infonce
else:
	TEMPERATURE = 0.5

#TEMPERATURE = 0.07

model = MultimodalArchitecture(device, CNN_TO_USE, in_dim = input_dim, 
						out_dim = output_dim, 
						intermediate_dim = hidden_dim,
						TEMPERATURE = TEMPERATURE, pretrained_path=None, patch_size=16)

if (MODALITY is MOD.img):
	for param in model.txt_encoder.parameters():
		param.requires_grad = False
	for param in model.embedding_output_txt.parameters():
		param.requires_grad = False
else:

	#if (MODALITY is not MOD.unimodal):
	for param in model.txt_encoder.embeddings.parameters():
		param.requires_grad = False

	for name, param in model.txt_encoder.encoder.named_parameters():
		#if '10' in name or '11' in name: 
		if '12' in name: 
			param.requires_grad = True
		else:
			param.requires_grad = False
	#"""
	for param in model.txt_encoder.pooler.parameters():
		param.requires_grad = False
	#"""


model.to(device)
#model.eval()


if (flag_KEYWORDS):

	keywords_pubmed_embeddings = utils_concept_extraction.get_keywords_embeddings_PubMED(model.txt_encoder, device)

	print(utils_concept_extraction.concept_to_index)	


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

if (N_CLASSES > 1):
	criterion_classification = torch.nn.CrossEntropyLoss()
else:
	criterion_classification = torch.nn.BCEWithLogitsLoss()

criterion_representation_pairwise = loss_functions.ContrastiveLoss()
criterion_representation_cosine = torch.nn.CosineEmbeddingLoss()
criterion_representation_rmse = torch.nn.L1Loss()

if (MODALITY is MOD.multimodal):

	criterion_contrastive = loss_functions.NT_Xent_supervised(batch_size = BATCH_SIZE, temperature = TEMPERATURE)


elif (MODALITY is MOD.InfoNCE_supervised):

	criterion_contrastive = loss_functions.InfoNCE_supervised(temperature = TEMPERATURE)

elif (MODALITY is MOD.InfoNCE or MODALITY is MOD.InfoNCE_only):

	criterion_contrastive = loss_functions.InfoNCE(temperature = TEMPERATURE)


elif (MODALITY is MOD.CLIP or MODALITY is MOD.CLIP_only):

	criterion_contrastive = loss_functions.CLIP_Loss(batch_size = BATCH_SIZE, temperature = TEMPERATURE)

elif (MODALITY is MOD.NT_Xent or MODALITY is MOD.NT_Xent_only):

	criterion_contrastive = loss_functions.NT_Xent(batch_size = BATCH_SIZE, temperature = TEMPERATURE)

if (flag_KEYWORDS):

	criterion_keywords = loss_functions.MultiPositiveInfoNCELoss(temperature = TEMPERATURE)


# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

label_embedding_pos = torch.tensor([1]).to(device)

WARMUP_LOSS_EPOCHS = 0
WARMUP_WEIGHTS_EPOCHS = 1
if (MODALITY is MOD.InfoNCE_supervised or MODALITY is MOD.InfoNCE):
	lambda_loss = 0.25 #best infonce 
else:
	lambda_loss = 0.5 #best infonce 

def model_forward(epoch, phase = 'train'):
	
	if (phase is PHASE.train):
		phase_str = 'train'
		dataloader_iterator = iter(training_generator_bag)
		iterations = int(len(train_dataset) / BATCH_SIZE) #+ 1
		model.train()
		
		filenames = np.empty((len(train_dataset)), dtype = 'object')

		y_pred_img = np.empty((len(train_dataset)), dtype=np.uint8)
		y_pred_txt = np.empty((len(train_dataset)), dtype=np.uint8)

		y_true = np.empty((len(train_dataset)), dtype=np.uint8)
		
		cumulative_preds_imgs = np.empty((len(train_dataset), N_CLASSES))
		cumulative_preds_txts = np.empty((len(train_dataset), N_CLASSES))

		flag_embeddings_pubmed = flag_embeddings_train

	elif (phase is PHASE.valid):
		phase_str = 'valid'
		dataloader_iterator = iter(validation_generator_bag)
		iterations = int(len(valid_dataset) / BATCH_SIZE) #+ 1
		model.eval()

		filenames = np.empty((len(valid_dataset)), dtype = 'object')

		y_pred_img = np.empty((len(valid_dataset)))
		y_pred_txt = np.empty((len(valid_dataset)))
		
		y_true = np.empty((len(valid_dataset)))

		cumulative_preds_imgs = np.empty((len(valid_dataset), N_CLASSES))
		cumulative_preds_txts = np.empty((len(valid_dataset), N_CLASSES))
		flag_embeddings_pubmed = flag_embeddings_valid

	#label_embedding_pos = torch.ones(BATCH_SIZE).to(device)
	

	img_loss = 0.0
	txt_loss = 0.0
	
	representation_loss_rmse = 0.0
	representation_loss_cosine = 0.0
	representation_pairwise = 0.0
	representation_contrastive = 0.0
	keyword_loss = 0.0
	
	kappa_val = 0.0
	f1_score_val = 0.0
	
	i = 0
	progress_bar = tqdm(range(iterations), desc='samples', position=0) 

	img_log = tqdm(total=0, position=1, bar_format='{desc}')
	txt_log = tqdm(total=0, position=2, bar_format='{desc}')
	#pairwise_log = tqdm(total=0, position=3, bar_format='{desc}')
	cosine_log = tqdm(total=0, position=3, bar_format='{desc}')
	contrastive_log = tqdm(total=0, position=4, bar_format='{desc}')
	rmse_log = tqdm(total=0, position=5, bar_format='{desc}')
	keyword_log = tqdm(total=0, position=6, bar_format='{desc}')
	kappa_log = tqdm(total=0, position=7, bar_format='{desc}')
	f1_score_log = tqdm(total=0, position=8, bar_format='{desc}')

	for i in progress_bar:

		with torch.autocast(device_type='cuda', dtype=torch.float16):

			try:
				IDs, tokens, labels, keywords, batch_filenames = next(dataloader_iterator)
			except StopIteration:
				dataloader_iterator = iter(training_generator_bag)
				IDs, tokens, labels, keywords, batch_filenames = next(dataloader_iterator)

			labels_np = labels.data.numpy()
			labels = labels.to(device, non_blocking  =True)
			
			batch_filenames = list(batch_filenames)

			model.zero_grad(set_to_none=True)
			optimizer_CNN.zero_grad(set_to_none=True)	

			if (phase is PHASE.train):
				if (MODALITY is MOD.img):
					IDs = IDs.to(device, non_blocking = True)
					logits_img, _, _, _ = model(IDs, None)

				elif (MODALITY is MOD.txt):
					tokens = tokens.to(device, non_blocking = True)
					_, _, logits_txt, _ = model(None, tokens, flag_embeddings_pubmed)

				elif (MODALITY is MOD.multimodal or MODALITY is MOD.InfoNCE_supervised):
					IDs = IDs.to(device, non_blocking = True)
					tokens = tokens.to(device, non_blocking = True)

					logits_img, cls_img, logits_txt, cls_txt = model(IDs, tokens, flag_embeddings_pubmed)

				elif (MODALITY is MOD.CLIP or MODALITY is MOD.NT_Xent or MODALITY is MOD.InfoNCE or
		  			MODALITY is MOD.CLIP_only or MODALITY is MOD.NT_Xent_only or MODALITY is MOD.InfoNCE_only):
					IDs = IDs.to(device, non_blocking = True)
					tokens = tokens.to(device, non_blocking = True)
					_, cls_img, _, cls_txt = model(IDs, tokens, flag_embeddings_pubmed)

				if (MODALITY is not MOD.img and MODALITY is not MOD.txt and flag_KEYWORDS):

					keywords = keywords.to(device, non_blocking = True)
					_, _, _, cls_keywords = model(None, keywords_pubmed_embeddings.to(device), flag_embeddings_pubmed)
					
			else:	
				with torch.no_grad():
					labels = torch.as_tensor(labels).to(device, non_blocking=True)

					if (MODALITY is MOD.img):
						IDs = IDs.to(device, non_blocking = True)
						logits_img, cls_img, logits_txt, cls_txt = model(IDs, None, flag_embeddings_pubmed)

					elif (MODALITY is MOD.txt):
						tokens = tokens.to(device, non_blocking = True)
						logits_img, cls_img, logits_txt, cls_txt = model(None, tokens, flag_embeddings_pubmed)

					elif (MODALITY is MOD.multimodal or MODALITY is MOD.InfoNCE_supervised):
						IDs = IDs.to(device, non_blocking = True)
						tokens = tokens.to(device, non_blocking = True)
						logits_img, cls_img, logits_txt, cls_txt = model(IDs, tokens, flag_embeddings_pubmed)

					elif (MODALITY is MOD.CLIP or MODALITY is MOD.NT_Xent or MODALITY is MOD.InfoNCE
						or MODALITY is MOD.CLIP_only or MODALITY is MOD.NT_Xent_only or MODALITY is MOD.InfoNCE_only):
						IDs = IDs.to(device, non_blocking = True)
						tokens = tokens.to(device, non_blocking = True)
						_, cls_img, _, cls_txt = model(IDs, tokens, flag_embeddings_pubmed)

					if (MODALITY is not MOD.img and MODALITY is not MOD.txt and flag_KEYWORDS):

						keywords = keywords.to(device, non_blocking = True)
						_, _, _, cls_keywords = model(None, keywords_pubmed_embeddings.to(device), flag_embeddings_pubmed)

			if (MODALITY is MOD.img or MODALITY is MOD.multimodal or MODALITY is MOD.InfoNCE_supervised):
				if (N_CLASSES > 1):
					softmax_output_img = F.softmax(logits_img)
				else:
					softmax_output_img = F.sigmoid(logits_img.view(-1))

				outputs_np_img = softmax_output_img.cpu().data.numpy()
			
				if (N_CLASSES > 1):
					output_norm_img = np.argmax(outputs_np_img, axis = 1)
				else:
					output_norm_img = np.where(outputs_np_img > 0.5, 1, 0)

				for j in range(len(output_norm_img)):
					cont = int(i * BATCH_SIZE + j)
					y_pred_img[cont] = output_norm_img[j]

				#y_pred_img = np.append(y_pred_img,output_norm_img)

			if (MODALITY is MOD.txt or MODALITY is MOD.multimodal or MODALITY is MOD.InfoNCE_supervised):
				if (N_CLASSES > 1):
					softmax_output_txt = F.softmax(logits_txt)
				else:
					softmax_output_txt = F.sigmoid(logits_txt.view(-1))

				outputs_np_txt = softmax_output_txt.cpu().data.numpy()
			
				if (N_CLASSES > 1):
					output_norm_txt = np.argmax(outputs_np_txt, axis = 1)
				else:
					output_norm_txt = np.where(outputs_np_txt > 0.5, 1, 0)

				#y_pred_txt = np.append(y_pred_txt,output_norm_txt)
				for j in range(len(output_norm_txt)):
					cont = int(i * BATCH_SIZE + j)
					y_pred_txt[cont] = output_norm_txt[j]
			
			#y_true = np.append(y_true,labels_np)
			if (MODALITY is MOD.img or MODALITY is MOD.txt or MODALITY is MOD.multimodal or MODALITY is MOD.InfoNCE_supervised):
				for j in range(len(labels_np)):
					cont = int(i * BATCH_SIZE + j)
					y_true[cont] = labels_np[j]

			if (i%50 == 0 and (MODALITY is MOD.img or MODALITY is MOD.multimodal or MODALITY is MOD.InfoNCE_supervised)):
				
				try:
					kappa_val = kappa_score(y_true[:cont], y_pred_img[:cont], None, None, None, None)
					_,_,f1_score_val = f1_scores(y_true[:cont], y_pred_img[:cont], None, None, None, None)
				except Exception as e:
					print(e)
					pass
   
			if (MODALITY is MOD.img):
				if (N_CLASSES > 1):
					loss_img = criterion_classification(logits_img, labels)
				else:
					loss_img = criterion_classification(logits_img.view(-1), labels)

				img_loss = img_loss + ((1 / (i+1)) * (loss_img.item() - img_loss))
				loss = loss_img

			elif (MODALITY is MOD.txt):
				if (N_CLASSES > 1):
					loss_txt = criterion_classification(logits_txt, labels)
				else:
					loss_txt = criterion_classification(logits_txt.view(-1), labels) 

				txt_loss = txt_loss + ((1 / (i+1)) * (loss_txt.item() - txt_loss))
				loss = loss_txt

			elif (MODALITY is MOD.multimodal or MODALITY is MOD.InfoNCE_supervised):
				#img
				if (N_CLASSES > 1):
					loss_img = criterion_classification(logits_img, labels)
				else:
					loss_img = criterion_classification(logits_img.view(-1), labels)

				img_loss = img_loss + ((1 / (i+1)) * (loss_img.item() - img_loss))

				#txt
				if (N_CLASSES > 1):
					loss_txt = criterion_classification(logits_txt, labels)
				else:
					loss_txt = criterion_classification(logits_txt.view(-1), labels) 

				txt_loss = txt_loss + ((1 / (i+1)) * (loss_txt.item() - txt_loss))
				
				#alignment
				loss_representation_cosine = criterion_representation_cosine(cls_img, cls_txt, label_embedding_pos)
				representation_loss_cosine = representation_loss_cosine + ((1 / (i+1)) * (loss_representation_cosine.item() - representation_loss_cosine))
				
				loss_representation_rmse = criterion_representation_rmse(cls_img, cls_txt)
				representation_loss_rmse = representation_loss_rmse + ((1 / (i+1)) * (loss_representation_rmse.item() - representation_loss_rmse))


				try:
					loss_contrastive = criterion_contrastive(cls_img, cls_txt, labels)
					representation_contrastive = representation_contrastive + ((1 / (i+1)) * (loss_contrastive.item() - representation_contrastive))
				except Exception as e:
					print(e)
					loss_contrastive = 0.0
					#pass	

				if (phase is PHASE.train):

					loss = loss_img + loss_txt + loss_representation_cosine + lambda_loss * loss_contrastive + loss_representation_rmse #+ lambda_loss * loss_representation_pairwise

				else:

					loss = loss_img
				
			elif (MODALITY is MOD.CLIP or MODALITY is MOD.NT_Xent or MODALITY is MOD.InfoNCE):
				
				loss_representation_cosine = criterion_representation_cosine(cls_img, cls_txt, label_embedding_pos)
				representation_loss_cosine = representation_loss_cosine + ((1 / (i+1)) * (loss_representation_cosine.item() - representation_loss_cosine))
				
				loss_representation_rmse = criterion_representation_rmse(cls_img, cls_txt)
				representation_loss_rmse = representation_loss_rmse + ((1 / (i+1)) * (loss_representation_rmse.item() - representation_loss_rmse))

				try:
					loss_contrastive = criterion_contrastive(cls_img, cls_txt)
					representation_contrastive = representation_contrastive + ((1 / (i+1)) * (loss_contrastive.item() - representation_contrastive))
				except Exception as e:
					#print(e)
					loss_contrastive = 0.0
					#pass	
				
				loss = loss_representation_cosine + loss_contrastive + loss_representation_rmse #+ lambda_loss * loss_representation_pairwise

				#"""
				img_loss = img_loss + ((1 / (i+1)) * (loss.item() - img_loss))

			elif (MODALITY is MOD.CLIP_only or MODALITY is MOD.NT_Xent_only or MODALITY is MOD.InfoNCE_only):
				
				
				try:
					loss_contrastive = criterion_contrastive(cls_img, cls_txt)
					representation_contrastive = representation_contrastive + ((1 / (i+1)) * (loss_contrastive.item() - representation_contrastive))
				except Exception as e:
					#print(e)
					loss_contrastive = 0.0
					#pass	
				
				loss = loss_contrastive #+ lambda_loss * loss_representation_pairwise
				#"""
				try:
					img_loss = img_loss + ((1 / (i+1)) * (loss.item() - img_loss))
				except:
					img_loss = img_loss + ((1 / (i+1)) * (loss - img_loss))


			if (flag_KEYWORDS):
				
				loss_keywords = criterion_keywords(cls_img, cls_keywords.detach(), keywords)
				keyword_loss = keyword_loss + ((1 / (i+1)) * (loss_keywords.item() - keyword_loss))
				loss = loss + lambda_loss * loss_keywords

			if (phase is PHASE.train):
				#"""
				scaler.scale(loss).backward()
				scaler.step(optimizer_CNN)
				scaler.update()
				optimizer_CNN.zero_grad(set_to_none=True)
				model.zero_grad(set_to_none=True)
				
			
			img_log.set_description_str(f'img: {img_loss}')
			txt_log.set_description_str(f'txt: {txt_loss}')
			#pairwise_log.set_description_str(f'pairwise: {representation_pairwise}')
			cosine_log.set_description_str(f'cosine: {representation_loss_cosine}')
			contrastive_log.set_description_str(f'contrastive: {representation_contrastive}')
			rmse_log.set_description_str(f'rmse: {representation_loss_rmse}')
			keyword_log.set_description_str(f'keyword: {keyword_loss}')
			kappa_log.set_description_str(f'kappa: {kappa_val}')
			f1_score_log.set_description_str(f'f1_score: {f1_score_val}')
   
			#filenames = np.append(filenames, batch_filenames)
			
			for j in range(len(batch_filenames)):
				cont = int(i * BATCH_SIZE + j)
				filenames[cont] = batch_filenames[j]
   
			if (MODALITY is MOD.img):

				for j in range(len(outputs_np_img)):
					cont = int(i * BATCH_SIZE + j)
					cumulative_preds_imgs[cont] = outputs_np_img[j]

			elif (MODALITY is MOD.txt):
				
				for j in range(len(outputs_np_txt)):
					cont = int(i * BATCH_SIZE + j)
					cumulative_preds_txts[cont] = outputs_np_txt[j]

			elif (MODALITY is MOD.multimodal or MODALITY is MOD.InfoNCE_supervised):

				for j in range(len(outputs_np_img)):
					cont = int(i * BATCH_SIZE + j)
					cumulative_preds_imgs[cont] = outputs_np_img[j]
					cumulative_preds_txts[cont] = outputs_np_txt[j]


			if (i == (iterations - 1)):

				if (MODALITY is not MOD.CLIP or MODALITY is not MOD.NT_Xent or MODALITY is MOD.InfoNCE or
					MODALITY is not MOD.CLIP_only or MODALITY is not MOD.NT_Xent_only or MODALITY is MOD.InfoNCE_only):
					preds_img = np.hstack((filenames.reshape(-1,1), cumulative_preds_imgs))
					preds_txt = np.hstack((filenames.reshape(-1,1), cumulative_preds_txts))
					
					utils_data.save_prediction(checkpoint_path, N_CLASSES, phase_str, epoch, preds_img, preds_txt, None, MODALITY)
					#save loss
					utils_data.save_loss_function(checkpoint_path, phase_str, epoch, img_loss)
			
				print()


	return img_loss


epoch = 0

best_loss = 100000.0

EARLY_STOP_NUM = 10
early_stop_cont = 0
epoch = 0

batch_size_instance = int(BATCH_SIZE_str)

while (epoch<num_epochs and early_stop_cont<EARLY_STOP_NUM):
		
	#train
	phase = PHASE['train']
	print("epoch: " + str(epoch) + "/" + str(num_epochs))
	print("train")
	train_loss = model_forward(epoch, phase = phase)
	print("valid")
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