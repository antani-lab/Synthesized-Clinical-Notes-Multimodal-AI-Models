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
sys.path.append("../utils/")
sys.path.append("../models/")

from enum_multi import ALG, PHASE, TYPE_DATA, MOD, REPORTS
import utils_data
import random
from dataloader import ImbalancedDatasetSampler, Dataset_generate_features, filter_labels, Dataset_instance_txt_generation
from model import MultimodalArchitecture, MultimodalArchitecture_CLIP, FeatureToTextDecoder
from transformers import AutoTokenizer
import utils_txt
from metrics_multiclass import accuracy_score, kappa_score, f1_scores, precisions, recalls
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
torch.backends.cudnn.benchmark = True
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
#algorithm parameters

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-c', '--CNN', help='cnn_to_use',type=str, default='PanDerm')
parser.add_argument('-e', '--EPOCHS', help='epochs to train',type=int, default=10)
parser.add_argument('-z', '--hidden_space', help='hidden_space_size',type=int, default=128)
parser.add_argument('-b', '--batch_size', help='batch size bag level',type=int, default=256)
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


#OUTPUT_folder = args.output_folder
print("CREATE DIRECTORY WHERE MODELS WILL BE STORED")

print("CREATE DIRECTORY WHERE MODELS WILL BE STORED")

OUTPUT_folder = args.output_folder
models_path = OUTPUT_folder
os.makedirs(models_path, exist_ok=True)
models_path = models_path + 'multimodal/txt_generator/'
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
CSV_FOLDER = MAIN_FOLDER + 'csv_folder/'
#all_data = valid_dataset
target_img = None

test_dataset = utils_data.get_instances_txt_generation(DATA_FOLDER, [DATASET], flag_all, CNN_TO_USE, MODALITY, N_EXP_str, CSV_FOLDER, REPORTS_TRAINING = REPORTS_TRAINING)

print(test_dataset.shape)

REPORT_DIR = DATA_FOLDER+DATASET+'/clinical_notes/decoder_reports/'+MODALITY+'/'

REPORT_DIR = REPORT_DIR + REPORTS_TRAINING + '/'

if (flag_KEYWORDS):
	REPORT_DIR = REPORT_DIR + 'keywords/'
else:
	REPORT_DIR = REPORT_DIR + 'no_keywords/'

if (REPORT_AUGMENTATION):
	REPORT_DIR = REPORT_DIR + 'report_augmentation/'
else:
	REPORT_DIR = REPORT_DIR + 'no_report_augmentation/'

REPORT_DIR = REPORT_DIR + PROBLEM + '/'
REPORT_DIR = REPORT_DIR + CNN_TO_USE+'/'
REPORT_DIR = REPORT_DIR + 'N_EXP_'+str(N_EXP)+'/'
os.makedirs(REPORT_DIR, exist_ok = True)

tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

# Special tokens
PAD_ID = tokenizer.pad_token_id
BOS_ID = tokenizer.cls_token_id  # or tokenizer.bos_token_id if defined
EOS_ID = tokenizer.sep_token_id  # or tokenizer.eos_token_id if defined

MAX_LEN_SEQ = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = FeatureToTextDecoder(feature_dim=128, hidden_dim=128, num_layers=4, nhead=4, dropout=0.1, max_len=MAX_LEN_SEQ)
model.load_state_dict(torch.load(model_weights_filename), strict = False) #best infonce (temp 0.07)

model = torch.compile(model, mode="reduce-overhead")

model.to(device)
model.eval()

n_workers = 0

params_valid_bag = {'batch_size': BATCH_SIZE,
		  'shuffle': False,
		  'num_workers': n_workers}

possible_reports = REPORTS[REPORTS_TRAINING]

testing_set_bag = Dataset_instance_txt_generation(test_dataset, possible_reports, None, None, 1.0, PHASE.valid, device, MAX_LEN_SEQ, only_imgs = True)
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
            cls_sample, _, _, batch_filenames = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(testing_generator_bag)
            cls_sample, _, _ = next(dataloader_iterator)

        cls_sample = cls_sample.to(device, non_blocking=True)

        batch_filenames = list(batch_filenames)

        with torch.no_grad():
            
            for f in range(len(cls_sample)):
                
                feat = cls_sample[f].unsqueeze(0)				
                generated_text = utils_txt.generate_text(model, feat, tokenizer, max_len=256)[0]

                #print(generated_text, batch_filenames[f])
                
                new_fname = REPORT_DIR + batch_filenames[f] + '.txt'
                
                f = open(new_fname, "w")
                f.write(generated_text)
                f.close() 