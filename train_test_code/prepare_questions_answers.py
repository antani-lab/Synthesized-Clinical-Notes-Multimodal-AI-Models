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

####PATH where find patches
#instance_dir = args.DATA_FOLDER
instance_dir = '/PLACEHOLDER/DATASET_FLD/'

#OUTPUT_folder = args.output_folder
print("CREATE DIRECTORY WHERE MODELS WILL BE STORED")

OUTPUT_folder = '/PLACEHOLDER/MODEL/WEIGHTS/'
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


MAIN_FOLDER = 'PLACEHOLDER'
MAIN_FOLDER = MAIN_FOLDER + '/datasets/'
CSV_FOLDER = MAIN_FOLDER + '/csv_folder/'
#all_data = valid_dataset


feat_prompts_dir = models_path + 'questions_VQA/'
os.makedirs(feat_prompts_dir, exist_ok = True)

list_answers = [
    'yes',
    'no',
    'cannot answer',
    #3-7 8-14
    'benign',
    'non-malignant',
    'pre-cancerous',
    'malignant',
    'cancerous',
    'benign keratosis',
    'dermatofibroma',
    'benign nevus',
    'vascular lesion',
    'actinic keratosis',
    'basal cell cancer',
    'melanoma', 
    #15-16
    'symmetric',
    'asymmetric',
    #17-19
    'indistinct',
    'sharp',
    'mixed',
    #20-27
    'white',
    'pink',
    'red',
    'blue',
    'grey',
    'brown',
    'black',
    'dark',
    #28-36
    'structureless',
    'pigment',
    'streaks',
    'dots',
    'globules',
    'macule',
    'plaque',
    'papule',
    'nodul',
    
]

list_questions = [
    "Is the lesion symmetric?",
    "Does the lesion have a symmetrical shape?",
    "Is there symmetry in the lesion appearance?",
    "Can the lesion be considered symmetrical?",
    "Is the lesion evenly shaped on all sides?",
    "Does the lesion show balanced or mirrored features?",

    "Is the lesion border regular?",
    "Are the edges of the lesion smooth and well-defined?",
    "Does the lesion have a regular or irregular border?",
    "Is the outline of the lesion even and consistent?",
    "Are the margins of the lesion clearly and regularly shaped?",
    "Does the lesion show uniformity along its border?",

    "What dermoscopic characteristics of the lesion?",
    "What features are visible under dermoscopic examination of the lesion?",
    "Which dermoscopic patterns are present in the lesion?",
    "What can be observed in the lesion through dermoscopy?",
    "What are the visual structures seen in the lesion under magnification?",
    "What specific dermoscopic traits does the lesion exhibit?",

    "What is the color of the lesion?",
    "Which colors are present in the lesion?",
    "What is the predominant color of the lesion?",
    "What color characteristics does the lesion show?",
    "How would you describe the lesion coloration?",
    "What hues or shades can be observed in the lesion?",
    
    "What type of lesion do you identify?",
    "How would you classify the lesion shown?",
    "What category does this lesion fall into?",
    "What kind of lesion is visible in the image?",
    "Can you determine the lesion type?",
    "Which lesion type do you recognize here?",
]

N_CLASSES = 5

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

if (MODALITY is MOD.CLIP):
	model = MultimodalArchitecture_CLIP(device, CNN_TO_USE, in_dim = input_dim, 
						out_dim = output_dim, 
						intermediate_dim = hidden_dim,
						TEMPERATURE = TEMPERATURE, patch_size=patch_size)
else:
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


#questions
filenames = []
features = []

for i in range(len(list_questions)):
	
	with torch.autocast(device_type='cuda', dtype=torch.float16):


		#keyword = 'the image suggests for melanoma, possibly malignat'
		keyword = list_questions[i]

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

		encoded_keyword_GPU = encoded_keyword.to(device, non_blocking=True)

		with torch.no_grad():
			_, _, _, cls_txt = model(None, encoded_keyword_GPU)

		cls_txt_np = cls_txt.cpu().data.numpy()
		

	features.append(cls_txt_np)

features = np.reshape(features, (len(list_questions), hidden_dim))



#save features
features_filename = feat_prompts_dir + 'cls_questions.npy'
with open(features_filename, 'wb') as f:
	np.save(f, features)

#save list
features_csv_filename = feat_prompts_dir + 'list_questions.csv'

File = {'keywords':list_questions}
df = pd.DataFrame(File,columns=['keywords'])

np.savetxt(features_csv_filename, df.values, fmt='%.s',delimiter=',')



#answers
filenames = []
features = []

for i in range(len(list_answers)):
	
	with torch.autocast(device_type='cuda', dtype=torch.float16):


		#keyword = 'the image suggests for melanoma, possibly malignat'
		keyword = list_answers[i]

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

		encoded_keyword_GPU = encoded_keyword.to(device, non_blocking=True)

		with torch.no_grad():
			_, _, _, cls_txt = model(None, encoded_keyword_GPU)

		cls_txt_np = cls_txt.cpu().data.numpy()
		

	features.append(cls_txt_np)

features = np.reshape(features, (len(list_answers), hidden_dim))


#save features
features_filename = feat_prompts_dir + 'cls_answers.npy'
with open(features_filename, 'wb') as f:
	np.save(f, features)

#save list
features_csv_filename = feat_prompts_dir + 'list_answers.csv'

File = {'keywords':list_answers}
df = pd.DataFrame(File,columns=['keywords'])

np.savetxt(features_csv_filename, df.values, fmt='%.s',delimiter=',')