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
from enum_multi import ALG, PHASE, TYPE_DATA, MOD
import utils_data
import random
from dataloader import ImbalancedDatasetSampler, Dataset_instance, filter_labels
from model import MultimodalArchitecture, MultimodalArchitecture_CLIP
from transformers import AutoTokenizer
import utils_zero_shot_learning
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
parser.add_argument('-o', '--output_folder', help='path folder where to store output model',type=str, default='')
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
PANDERM_FOLDER = args.weights

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
features_dir = models_path+'prompts/'
os.makedirs(features_dir, exist_ok=True)


#path model file
model_weights_filename = models_path+'model.pt'

MAIN_FOLDER = args.DATA_FOLDER
DATA_FOLDER = MAIN_FOLDER + '/datasets/'
CSV_FOLDER = MAIN_FOLDER + 'csv_folder/'
zero_shot_learning_fld = CSV_FOLDER + 'zero_shot_keywords/'

filename_keyword_classes = zero_shot_learning_fld + "keyword_classes.csv"
filename_keyword_subclasses = zero_shot_learning_fld + "keyword_subclasses.csv"
filename_keyword_classes_matching = zero_shot_learning_fld + "keyword_classes_matching.csv"
filename_keyword_subclasses_matching = zero_shot_learning_fld + "keyword_subclasses_matching.csv"

keyword_classes = pd.read_csv(filename_keyword_classes, sep = ',', header = None).values.tolist()
keyword_subclasses = pd.read_csv(filename_keyword_subclasses, sep = ',', header = None).values.tolist()
keyword_classes_matching = pd.read_csv(filename_keyword_classes_matching, sep = ',', header = None).values.tolist()
keyword_subclasses_matching = pd.read_csv(filename_keyword_subclasses_matching, sep = ',', header = None).values.tolist()

N_CLASSES = 8

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
	model_weights_filename_pre_trained_img = PANDERM_FOLDER + "/panderm_bb_data6_checkpoint-499.pth"
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


#lr = 1e-4
num_epochs = EPOCHS

lambda_val = 0.5 

#TODO change classes

cumulative_preds = np.empty((0, N_CLASSES))


bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'

try:
	tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  
except:
	try:
		tokenizer = AutoTokenizer.from_pretrained(bert_chosen, local_files_only = True, force_download = True)  
	except:
		tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  
		

"""
keyword_classes = pd.read_csv(filename_keyword_classes, sep = ',', header = None).values.tolist()

filenames = []
features = []

for i in range(len(keyword_classes)):
	
	with torch.autocast(device_type='cuda', dtype=torch.float16):


		#keyword = 'the image suggests for melanoma, possibly malignat'
		keyword = keyword_classes[i][0]

		print(i, keyword)

		encoded_keyword = tokenizer(
						keyword,
						add_special_tokens=True,
						return_token_type_ids=True,
						return_attention_mask=True,
						padding="max_length",  # Pads all sequences to the max_length in the batch
						truncation=True,  # Ensures no sequence exceeds max length
						max_length=512,  # Adjust based on model constraints
						return_tensors="pt"  # Directly returns PyTorch tensors
					)

		encoded_keyword_GPU = encoded_keyword.to(device)

		with torch.no_grad():
			_, _, _, cls_txt = model(None, encoded_keyword_GPU)

		cls_txt_np = cls_txt.cpu().data.numpy()
		filenames.append(keyword)

	features.append(cls_txt_np)

features = np.reshape(features, (len(keyword_classes), hidden_dim))

#save features
features_filename = features_dir + 'cls_reports_classes.npy'
with open(features_filename, 'wb') as f:
	np.save(f, features)

#save list
features_csv_filename = features_dir + 'list_keywords_classes.csv'

File = {'keywords':filenames}
df = pd.DataFrame(File,columns=['keywords'])

df.to_csv(features_csv_filename, index=False, header=False)

keyword_subclasses = pd.read_csv(filename_keyword_subclasses, sep = ',', header = None).values.tolist()


filenames = []
features = []

for i in range(len(keyword_subclasses)):
	
	with torch.autocast(device_type='cuda', dtype=torch.float16):


		#keyword = 'the image suggests for melanoma, possibly malignat'
		keyword = keyword_subclasses[i][0]

		print(i, keyword)

		encoded_keyword = tokenizer(
						keyword,
						add_special_tokens=True,
						return_token_type_ids=True,
						return_attention_mask=True,
						padding="max_length",  # Pads all sequences to the max_length in the batch
						truncation=True,  # Ensures no sequence exceeds max length
						max_length=512,  # Adjust based on model constraints
						return_tensors="pt"  # Directly returns PyTorch tensors
					)

		encoded_keyword_GPU = encoded_keyword.to(device)

		with torch.no_grad():
			_, _, _, cls_txt = model(None, encoded_keyword_GPU)

		cls_txt_np = cls_txt.cpu().data.numpy()
		filenames.append(keyword)

	features.append(cls_txt_np)

features = np.reshape(features, (len(keyword_subclasses), hidden_dim))


#save features
features_filename = features_dir + 'cls_reports_subclasses.npy'
with open(features_filename, 'wb') as f:
	np.save(f, features)

#save list
features_csv_filename = features_dir + 'list_keywords_subclasses.csv'

File = {'keywords':filenames}
df = pd.DataFrame(File,columns=['keywords'])

df.to_csv(features_csv_filename, index=False, header=False)
"""


keyword_classes_matching = pd.read_csv(filename_keyword_classes_matching, sep = ',', header = None).values.tolist()

filenames = []
features = []

for i in range(len(keyword_classes_matching)):
	
	with torch.autocast(device_type='cuda', dtype=torch.float16):


		#keyword = 'the image suggests for melanoma, possibly malignat'
		keyword = keyword_classes_matching[i][0]

		print(i, keyword)

		encoded_keyword = tokenizer(
						keyword,
						add_special_tokens=True,
						return_token_type_ids=True,
						return_attention_mask=True,
						padding="max_length",  # Pads all sequences to the max_length in the batch
						truncation=True,  # Ensures no sequence exceeds max length
						max_length=512,  # Adjust based on model constraints
						return_tensors="pt"  # Directly returns PyTorch tensors
					)

		encoded_keyword_GPU = encoded_keyword.to(device)

		with torch.no_grad():
			_, _, _, cls_txt = model(None, encoded_keyword_GPU)

		cls_txt_np = cls_txt.cpu().data.numpy()
		filenames.append(keyword)

	features.append(cls_txt_np)

features = np.reshape(features, (len(keyword_classes_matching), hidden_dim))


#save features
features_filename = features_dir + 'cls_reports_classes_matching.npy'
with open(features_filename, 'wb') as f:
	np.save(f, features)

#save list
features_csv_filename = features_dir + 'list_keywords_classes_matching.csv'

File = {'keywords':filenames}
df = pd.DataFrame(File,columns=['keywords'])

df.to_csv(features_csv_filename, index=False, header=False)


keyword_subclasses_matching = pd.read_csv(filename_keyword_subclasses_matching, sep = ',', header = None).values.tolist()

filenames = []
features = []

for i in range(len(keyword_subclasses_matching)):
	
	with torch.autocast(device_type='cuda', dtype=torch.float16):


		#keyword = 'the image suggests for melanoma, possibly malignat'
		keyword = keyword_subclasses_matching[i][0]

		print(i, keyword)

		encoded_keyword = tokenizer(
						keyword,
						add_special_tokens=True,
						return_token_type_ids=True,
						return_attention_mask=True,
						padding="max_length",  # Pads all sequences to the max_length in the batch
						truncation=True,  # Ensures no sequence exceeds max length
						max_length=512,  # Adjust based on model constraints
						return_tensors="pt"  # Directly returns PyTorch tensors
					)

		encoded_keyword_GPU = encoded_keyword.to(device)

		with torch.no_grad():
			_, _, _, cls_txt = model(None, encoded_keyword_GPU)

		cls_txt_np = cls_txt.cpu().data.numpy()
		filenames.append(keyword)

	features.append(cls_txt_np)

features = np.reshape(features, (len(keyword_subclasses_matching), hidden_dim))


#save features
features_filename = features_dir + 'cls_reports_subclasses_matching.npy'
with open(features_filename, 'wb') as f:
	np.save(f, features)

#save list
features_csv_filename = features_dir + 'list_keywords_subclasses_matching.csv'

File = {'keywords':filenames}
df = pd.DataFrame(File,columns=['keywords'])

df.to_csv(features_csv_filename, index=False, header=False)
