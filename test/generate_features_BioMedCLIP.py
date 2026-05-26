import torch
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
import os
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
import sys
import argparse
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
parser.add_argument('-b', '--batch_size', help='batch size bag level',type=int, default=32)
parser.add_argument('-d', '--DATASET', help='dataset to use',type=str, default='HAM10000')
parser.add_argument('-i', '--INPUT', help='data to analyze',type=str, default='abcd')
parser.add_argument('-m', '--MAIN_FOLDER', help='path to main folder including image and csv folders',type=str, default='')

args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
CNN_TO_USE = args.CNN
BATCH_SIZE = args.batch_size

EMBEDDING_bool = True
DATASET = args.DATASET

INPUT_DATA = args.INPUT

seed = N_EXP
torch.manual_seed(seed)
#torch.use_deterministic_algorithms(mode=True)
if torch.cuda.is_available():
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
#np.random.seed(seed)
#random.seed(seed)

print("PARAMETERS")
print("CNN used: " + str(CNN_TO_USE))
print("DATASET: " + str(DATASET))
print("N_EXP: " + str(N_EXP_str))
####PATH where find patches
#instance_dir = args.DATA_FOLDER


#OUTPUT_folder = args.output_folder
print("CREATE DIRECTORY WHERE MODELS WILL BE STORED")



N_CLASSES = 5

MAIN_FOLDER = args.MAIN_FOLDER

DATA_FOLDER = MAIN_FOLDER + '/datasets/'
CSV_FOLDER = MAIN_FOLDER + '/csv_folder/'
#all_data = valid_dataset
target_img = None

features_dir = DATA_FOLDER+DATASET+'/features_BioMedClip/'
os.makedirs(features_dir, exist_ok=True)
flag_img = False

if (INPUT_DATA == "images"):
	feat_dir = features_dir + 'images/'
	os.makedirs(feat_dir, exist_ok = True)
	flag_img = True
elif (INPUT_DATA == "short"):
	feat_dir = features_dir + 'reports_shorts/'
	os.makedirs(feat_dir, exist_ok = True)
else:
	feat_dir = features_dir + 'reports_'+INPUT_DATA+'/'
	os.makedirs(feat_dir, exist_ok = True)



if (INPUT_DATA == 'images'):
	DATA_FOLDER = MAIN_FOLDER + DATASET + "/resized_images/"

elif (INPUT_DATA == 'abcd'):
	DATA_FOLDER = MAIN_FOLDER + DATASET + "/clinical_notes/abcd/"

elif (INPUT_DATA == 'char'):
	DATA_FOLDER = MAIN_FOLDER + DATASET + "/clinical_notes/char/"

elif (INPUT_DATA == 'short'):
	DATA_FOLDER = MAIN_FOLDER + DATASET + "/clinical_notes/shorts/"

elif (INPUT_DATA == 'doc'):
	DATA_FOLDER = MAIN_FOLDER + DATASET + "/clinical_notes/doc/"

elif (INPUT_DATA == 'medgemma_abcd'):
	DATA_FOLDER = MAIN_FOLDER + DATASET + "/clinical_notes/medgemma_abcd/"

elif (INPUT_DATA == 'medgemma_char'):
	DATA_FOLDER = MAIN_FOLDER + DATASET + "/clinical_notes/medgemma_char/"


features_dirs = [feat_dir]

test_dataset_filename = CSV_FOLDER + DATASET + "/labels.csv"
test_dataset = pd.read_csv(test_dataset_filename, sep = ',', header = None).values

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

feat_size = 512
context_length = 512

def encode_images(img_tensor):
    """
    img_tensor: [N, 3, H, W], preprocessed
    returns: image_features [N, D] (L2-normalized)
    """
    with torch.no_grad():
        image_features = model.encode_image(img_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features


def encode_texts(text_list):
    """
    text_list: list of strings
    returns: text_features [M, D] (L2-normalized)
    """
    tokens = tokenizer(text_list, context_length=context_length).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

model.eval()
model.to(device)
	
hidden_dim = 512

np.random.shuffle(test_dataset)
test_dataset = np.random.permutation(test_dataset)

#print(test_dataset.shape)
#print(test_dataset[:10])

for i in tqdm(range(len(test_dataset))):

	sample = test_dataset[i]
	sample_fname = sample[0]
	features_filename = feat_dir + sample_fname.split('.')[0] +  '.npy'

	if (os.path.exists(features_filename) == False):

		if (flag_img):
			sample_fname = DATA_FOLDER + sample_fname
			img = Image.open(sample_fname)
		
			img = preprocess(img)
			img = torch.unsqueeze(img, 0)
			img_tensor = img.to(device)

			with torch.no_grad():
				image_feat = model.encode_image(img_tensor)
				image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

			cls_np = image_feat.cpu().numpy()
		
		else:
			sample_fname = DATA_FOLDER + sample_fname.split('.')[0] + ".txt"

			with open(sample_fname, 'r', encoding='utf-8', errors='ignore') as file:
				#with open(ID_txt, 'r') as file:
				input_txt = file.read()
				file.close()

			tokens = tokenizer([input_txt], context_length=context_length).to(device)
			with torch.no_grad():
				features_classes = model.encode_text(tokens)
				features_classes = features_classes / features_classes.norm(dim=-1, keepdim=True)

			cls_np = features_classes.cpu().numpy()

		cls_np = np.squeeze(cls_np)
		#print(cls_np.shape)
		

		#image
		

		#print(features_filename)
		try:
			features_to_save = cls_np
			with open(features_filename, 'wb') as f:
				np.save(f, features_to_save)
		except Exception as ex:
			print(ex)
			pass

			
