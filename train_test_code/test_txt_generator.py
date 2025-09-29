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
from dataloader import ImbalancedDatasetSampler, Dataset_eval_txt_generation, filter_labels
from model import MultimodalArchitecture
from transformers import AutoModel, AutoTokenizer
from bert_score import BERTScorer
import utils_txt
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
torch.backends.cudnn.benchmark = True
#algorithm parameters

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-c', '--CNN', help='cnn_to_use',type=str, default='densenet121')
parser.add_argument('-e', '--EPOCHS', help='epochs to train',type=int, default=10)
parser.add_argument('-z', '--hidden_space', help='hidden_space_size',type=int, default=128)
parser.add_argument('-b', '--batch_size', help='batch size bag level',type=int, default=512)
parser.add_argument('-o', '--output_folder', help='path folder where to store output model',type=str, default='')
parser.add_argument('-i', '--DATA_FOLDER', help='path of the folder where to images are stored',type=str, default='')
parser.add_argument('-f', '--CSV_FOLDER', help='folder where csv including IDs and classes are stored',type=str, default='True')
parser.add_argument('-d', '--DATASET', help='dataset to use',type=str, default='HAM10000')
parser.add_argument('-w', '--PROBLEM', help='type of PROBLEM:binary (nevus vs melanoma), multiclass (sebo vs nevus vs basal vs melanoma)',type=str, default='multiclass')
parser.add_argument('-m', '--MODALITY', help='modalities to use: img, txt, multimodal',type=str, default='img')
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
instance_dir = '//PLACEHOLDER/DATASETS/'

#OUTPUT_folder = args.output_folder
print("CREATE DIRECTORY WHERE MODELS WILL BE STORED")

OUTPUT_folder = '/PLACEHOLDER/DATASETS/model_weights/'
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

#LOAD DATA
#TODO ADD PARAMETER
list_DATASETS = ['HAM10000']


flag_test = PHASE.test

MAIN_FOLDER = 'PLACEHOLDER'
MAIN_FOLDER = MAIN_FOLDER + '/datasets/'
CSV_FOLDER = MAIN_FOLDER + '/csv_folder/'
#all_data = valid_dataset
target_img = None

fname = CSV_FOLDER + 'labels_test.csv'

test_dataset = pd.read_csv(fname, sep = ',', header = None, dtype = 'object').values

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

	
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


FLD_REPORTS = '//PLACEHOLDER/DATASETS/' + DATASET + '/clinical_notes/'

abcd_fold = FLD_REPORTS + 'gpt_4o_mini_fld/'
char_fold = FLD_REPORTS + 'gpt_4o_mini_fld_abcd/'
doc_fold = FLD_REPORTS + 'gpt_4_mini_as_doctor/'
short_fold = FLD_REPORTS + 'short_reports/'

REPORT_DIR = MAIN_FOLDER+DATASET+'/clinical_notes/decoder_reports/'+MODALITY+'/'

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


params_train_bag = {'batch_size': BATCH_SIZE,
		'pin_memory': False,
		'shuffle': False,
		#'drop_last':True,
		}

FLDs = [abcd_fold, 
		char_fold, 
		doc_fold, 
		short_fold, 
		REPORT_DIR]

testing_set_bag = Dataset_eval_txt_generation(test_dataset[:,0], FLDs)
validation_generator_bag = data.DataLoader(testing_set_bag, **params_train_bag)


iterations = len(test_dataset)

filenames = np.empty(iterations,dtype='object')
F1s_abcd = np.empty(iterations)
Recalls_abcd = np.empty(iterations)
Precisions_abcd = np.empty(iterations)

F1s_char = np.empty(iterations)
Recalls_char = np.empty(iterations)
Precisions_char = np.empty(iterations)

F1s_doc = np.empty(iterations)
Recalls_doc = np.empty(iterations)
Precisions_doc = np.empty(iterations)

F1s_short = np.empty(iterations)
Recalls_short = np.empty(iterations)
Precisions_short = np.empty(iterations)

#scorer = BERTScorer(model_type='bert-base-uncased', device = device, nthreads = 1, batch_size=BATCH_SIZE)

def load_model_and_tokenizer(model_name, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer

def get_embeddings(texts, model, tokenizer, device="cpu"):
    """
    Returns token-level embeddings (batch, seq_len, hidden_dim)
    and attention mask.
    """
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
        last_hidden = outputs.last_hidden_state    # (batch, seq_len, hidden)
    mask = encoded["attention_mask"]              # (batch, seq_len)
    return last_hidden, mask

def cosine_sim_matrix(x, y):
    """
    Compute cosine similarity between all pairs of tokens for
    two sequences: x: (seq_x, hidden), y: (seq_y, hidden)
    return matrix: (seq_x, seq_y)
    """
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    return torch.mm(x_norm, y_norm.transpose(0, 1))

def bertscore_single_pair(cand_emb, ref_emb, cand_mask, ref_mask):
    """
    Computes precision, recall, F1 for a single pair of sequences.
    """
    # Remove padding tokens
    cand_tokens = cand_emb[cand_mask.bool()]
    ref_tokens = ref_emb[ref_mask.bool()]

    sim_matrix = cosine_sim_matrix(cand_tokens, ref_tokens)

    # Precision: for each cand token, max similarity to any ref token
    precision_vals = sim_matrix.max(dim=1).values

    # Recall: for each ref token, max similarity from any cand token
    recall_vals = sim_matrix.max(dim=0).values

    precision = precision_vals.mean()
    recall = recall_vals.mean()

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = torch.tensor(0.0)

    return precision, recall, f1

def bertscore_module(candidates, references, model, tokenizer, device="cpu"):
    """
    Full pipeline: loads model, computes token-level BERTScore.
    """
    # 1) Load model and tokenizer
    

    # 2) Get embeddings for all
    cand_embs, cand_masks = get_embeddings(candidates, model, tokenizer, device)
    ref_embs, ref_masks = get_embeddings(references, model, tokenizer, device)

    # 3) Compute scores pair-wise
    results = np.empty((len(candidates),3))
    for i in range(len(candidates)):
        P, R, F1 = bertscore_single_pair(
            cand_embs[i],
            ref_embs[i],
            cand_masks[i],
            ref_masks[i]
        )
        results[i] = [P.item(), R.item(), F1.item()]

    results = np.array(results)

    return results


c = 0

dataloader_iterator = iter(validation_generator_bag)
iterations = iterations = int(len(test_dataset) / BATCH_SIZE) + 1

model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
model, tokenizer = load_model_and_tokenizer(model_name, device)
model.to(device)
model.eval

for i in tqdm(range(iterations)):

	try:
		fnames, input_abcd, input_char, input_doc, input_short, input_generated = next(dataloader_iterator)
	except StopIteration:
		dataloader_iterator = iter(dataloader_iterator)
		fnames, input_abcd, input_char, input_doc, input_short, input_generated = next(dataloader_iterator)

	with torch.no_grad():
		scores = bertscore_module(
			input_generated,
			input_abcd,
			model,
			tokenizer,
			device=device  # or "cuda"
		)
		P_abcd, R_abcd, F1_abcd = scores[:,0], scores[:,1], scores[:,2]
		
	with torch.no_grad():
		scores = bertscore_module(
			input_generated,
			input_char,
			model,
			tokenizer,
			device=device  # or "cuda"
		)
		P_char, R_char, F1_char = scores[:,0], scores[:,1], scores[:,2]
	
	with torch.no_grad():
		scores = bertscore_module(
			input_generated,
			input_doc,
			model,
			tokenizer,
			device=device  # or "cuda"
		)
		P_doc, R_doc, F1_doc = scores[:,0], scores[:,1], scores[:,2]

	with torch.no_grad():
		scores = bertscore_module(
			input_generated,
			input_short,
			model,
			tokenizer,
			device=device  # or "cuda"
		)
		P_short, R_short, F1_short = scores[:,0], scores[:,1], scores[:,2]

	#P_abcd, R_abcd, F1_abcd = scorer.score(input_abcd, input_generated)
	#P_char, R_char, F1_char = scorer.score(input_char, input_generated)
	#P_doc, R_doc, F1_doc = scorer.score(input_doc, input_generated)
	#P_short, R_short, F1_short = scorer.score(input_short, input_generated)

	for j in range(len(P_abcd)):
		
	
		F1s_abcd[c] = F1_abcd[j]
		Recalls_abcd[c] = R_abcd[j]
		Precisions_abcd[c] = P_abcd[j]


		
		F1s_char[c] = F1_char[j]
		Recalls_char[c] = R_char[j]
		Precisions_char[c] = P_char[j]


		
		F1s_doc[c] = F1_doc[j]
		Recalls_doc[c] = R_doc[j]
		Precisions_doc[c] = P_doc[j]

		
		F1s_short[c] = F1_short[j]
		Recalls_short[c] = R_short[j]
		Precisions_short[c] = P_short[j]

		filenames[c] = fnames[j]

		c = c + 1



FLD_OUTPUT = checkpoint_path + '/test/txt_generation/'
os.makedirs(FLD_OUTPUT, exist_ok=True)

print(FLD_OUTPUT)

fname_abcd = FLD_OUTPUT + '/eval_abcd_' + DATASET + '.csv'
File = {'filenames':filenames, 'precisions':Precisions_abcd,'recalls':Recalls_abcd,'f1_score':F1s_abcd}
df = pd.DataFrame(File,columns=['filenames','precisions','recalls','f1_score'])
np.savetxt(fname_abcd, df.values, fmt='%s',delimiter=',')


fname_char = FLD_OUTPUT + '/eval_char_' + DATASET + '.csv'
File = {'filenames':filenames, 'precisions':Precisions_char,'recalls':Recalls_char,'f1_score':F1s_char}
df = pd.DataFrame(File,columns=['filenames','precisions','recalls','f1_score'])
np.savetxt(fname_char, df.values, fmt='%s',delimiter=',')


fname_doc = FLD_OUTPUT + '/eval_doc_' + DATASET + '.csv'
File = {'filenames':filenames, 'precisions':Precisions_doc,'recalls':Recalls_doc,'f1_score':F1s_doc}
df = pd.DataFrame(File,columns=['filenames','precisions','recalls','f1_score'])
np.savetxt(fname_doc, df.values, fmt='%s',delimiter=',')


fname_short = FLD_OUTPUT + '/eval_short_' + DATASET + '.csv'
File = {'filenames':filenames, 'precisions':Precisions_short,'recalls':Recalls_short,'f1_score':F1s_short}
df = pd.DataFrame(File,columns=['filenames','precisions','recalls','f1_score'])
np.savetxt(fname_short, df.values, fmt='%s',delimiter=',')
