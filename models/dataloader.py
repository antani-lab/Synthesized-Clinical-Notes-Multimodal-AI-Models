import sys, getopt
import torch
from torch.utils import data
import numpy as np 
import pandas as pd
from PIL import Image
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
from torchvision import transforms
from enum_multi import PHASE, MOD, COMPONENTS, REPORTS
import random
import data_augmentation
import utils_data
from scipy.spatial import KDTree
import skimage
from skimage.color import rgb2hsv, hsv2rgb, rgb2lab, lab2rgb
import colorsys
import utils_mask
import color_transformation
from transformers import AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "False"
import utils_txt
import json
import albumentations as A
import cv2
import utils_concept_extraction
import utils_SD_text

#sampler
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
	
	#https://github.com/ufoym/imbalanced-dataset-sampler

	def __init__(self, dataset, indices=None, num_samples=None):
				
		# if indices is not provided, 
		# all elements in the dataset will be considered
		self.indices = list(range(len(dataset)))             if indices is None else indices
			
		# if num_samples is not provided, 
		# draw `len(indices)` samples in each iteration
		self.num_samples = len(self.indices)             if num_samples is None else num_samples
			
		# distribution of classes in the dataset 
		label_to_count = {}
		for idx in self.indices:
			label = self._get_label(dataset, idx)
			if label in label_to_count:
				label_to_count[label] += 1
			else:
				label_to_count[label] = 1
				
		# weight for each sample
		weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
				   for idx in self.indices]
		self.weights = torch.DoubleTensor(weights)

	def _get_label(self, dataset, idx):
		return dataset[idx,1]
				
	def __iter__(self):
		return (self.indices[i] for i in torch.multinomial(
			self.weights, self.num_samples, replacement=True))

	def __len__(self):
		return self.num_samples

def map_classes(dataset, N_CLASSES):

	filenames = []
	labels = []
	centers = []

	flag_center = False
	for i in range(len(dataset)):

		fname = dataset[i,0]
		label = int(dataset[i,1])
		
		if (N_CLASSES == 3):
			if (label <= 3):
				new_label = 0
			elif (label == 4):
				new_label = 1
			elif (label > 4):
				new_label = 2
		elif (N_CLASSES == 2 or N_CLASSES == 1):
			if (label <= 4):
				new_label = 0
			else:
				new_label = 1
		else:
			new_label = label

		filenames.append(fname)
		labels.append(new_label)

		try:
			centers.append(dataset[i,2])
			flag_center = True
		except:
			pass
		
	if (flag_center):
		data_to_use = np.column_stack((filenames, labels, centers))
	else:
		data_to_use = np.column_stack((filenames, labels))
	return data_to_use


def filter_labels(arr, labels_to_remove, n_classes = 8):
	labels_to_remove = set(labels_to_remove)

	if (labels_to_remove == []):
		adjusted_arr = arr
	else:

		filenames = []
		labels = []
		centers = []

		flag_center = False
		for r in arr:
			try:
				label = int(r[1])

				if label not in labels_to_remove:
					filenames.append(r[0])
					labels.append(label)
					try:
						centers.append(r[2])
						flag_center = True
					except IndexError:
						pass
			except Exception as ex:
				pass
				#print(r)
			

		if flag_center:
			data_to_use = np.column_stack((filenames, labels, centers))
		else:
			data_to_use = np.column_stack((filenames, labels))

		# Build the new label mapping
		# -> Consider all labels from 0 to n_classes-1, excluding the labels_to_remove
		remaining_labels = [label for label in range(n_classes) if label not in labels_to_remove]

		# Now map: old label -> new label (even if some have no samples)
		label_mapping = {old_label: new_idx for new_idx, old_label in enumerate(remaining_labels)}

		# Apply remapping only to the available data
		if flag_center:
			adjusted_arr = np.array([[filename, label_mapping[int(label)], center] 
										for filename, label, center in data_to_use])
		else:
			adjusted_arr = np.array([[filename, label_mapping[int(label)]] 
										for filename, label in data_to_use])

	return adjusted_arr


def get_augmented_reports(ID_txt):

	augmented_txt = None

	# Split the path
	parent, child = os.path.split(ID_txt)           # parent = "/XXX/YYY/TTT", child = "ZZZ"
	grandparent, ttt = os.path.split(parent)      # grandparent = "/XXX/YYY", ttt = "TTT"

	# Modify the TTT folder name
	new_ttt = ttt + "_aug"

	# Reconstruct the new path
	new_path = os.path.join(grandparent, new_ttt, child)

	new_path = new_path.split('.')[0] + '.json'

	with open(new_path, 'r') as f:
		data_aug = json.load(f)

	data_aug = data_aug["variations"]

	idx = np.random.randint(0,len(data_aug))
	augmented_txt = data_aug[idx]

	return augmented_txt

#dataloaders
#data loader at patch-level
class Dataset_instance(data.Dataset):

	def __init__(self, list_IDs, phase, prob, classes, modality, doctor_type = False):

		self.list_IDs = utils_data.labels2int(list_IDs)

		self.modality = modality
		self.geometric_pipeline = color_transformation.get_pipeline_geometric(prob)
		self.color_pipeline = color_transformation.get_pipeline_color(prob)
		self.mode = phase

		self.prob = prob
		self.preprocess = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

		self.doctor_type = doctor_type

		self.N_CLASSES = classes

		self.patch_size = 224

		if (self.modality is not MOD.img):

			try:
				bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
				self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  
			except:
				try:
					bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
					self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen, local_files_only = True, force_download = True)  
				except:
					bert_chosen = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
					self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  


	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID_img = self.list_IDs[index,0]
		y = int(self.list_IDs[index,1])
		try:
			ID_txt = self.list_IDs[index,2]
		except:
			ID_txt = None
			
		input_tensor = torch.as_tensor(float(-1) , dtype=torch.long)
		encoded_text = torch.as_tensor(float(-1) , dtype=torch.long)

		if (self.modality is not MOD.txt):
			X = Image.open(ID_img)
			X = np.asarray(X)


			if (self.mode is PHASE.train):
				X = self.geometric_pipeline(image=X)['image']
				X = self.color_pipeline(image=X)['image']

			input_tensor = self.preprocess(X).type(torch.FloatTensor)

		if (self.modality is not MOD.img):
			
			if (self.doctor_type == True):

				new_fname = ID_txt.replace('short_reports','gpt_4_mini_as_doctor')
				flag_change = True
				if (os.path.exists(new_fname) == True):
					ID_txt = new_fname
					#print('a')


			if (self.mode is PHASE.train and self.doctor_type == False):

				
				prob_pre = np.random.rand(1)[0]

				flag_change = False
				#if (prob_pre >= self.prob):
				

				if (prob_pre >= 0.33 and prob_pre <= 0.66):
					ID_txt = ID_txt.replace('short_reports','gpt_4o_mini_fld_abcd')
					flag_change = True
				elif (prob_pre >= 0.66):
					ID_txt = ID_txt.replace('short_reports','gpt_4o_mini_fld')
					flag_change = True

				

			if (self.mode is PHASE.train and flag_change == True):
				prob_pre = np.random.rand(1)[0]
				
				if (prob_pre >= 0.5):
					try:
						input_txt = get_augmented_reports(ID_txt)
					except:
						input_txt = utils_txt.load_txt(ID_txt)
				else:
					input_txt = utils_txt.load_txt(ID_txt)
					
			else:
				input_txt = utils_txt.load_txt(ID_txt)
			#print(input_txt)
			
			encoded_text = self.tokenizer(
				input_txt,
				add_special_tokens=True,
				return_token_type_ids=True,
				return_attention_mask=True,
				padding="max_length",  # Pads all sequences to the max_length in the batch
				truncation=True,  # Ensures no sequence exceeds max length
				max_length=512,  # Adjust based on model constraints
				return_tensors="pt"  # Directly returns PyTorch tensors
			)

		if (self.N_CLASSES > 1):
			y = torch.as_tensor(float(y) , dtype=torch.long)
		else:
			y = torch.as_tensor(float(y) , dtype=torch.float32)

		#return input_tensor
		#print(len(encoded_text))
		ID_img = ID_img.split('/')[-1]

		return input_tensor, encoded_text, y, ID_img



class Dataset_generate_features(data.Dataset):

	def __init__(self, list_IDs, phase, prob, classes, flag_embeddings, CNN_TO_USE = "densenet121"):

		self.list_IDs = utils_data.labels2int(list_IDs)
		self.flag_embeddings = flag_embeddings
		self.mode = phase

		self.prob = prob
		if (CNN_TO_USE == "ViT"):
			self.preprocess = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
			])
		else:
			self.preprocess = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			])


		self.N_CLASSES = classes

		self.patch_size = 224

		try:
			bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
			self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  
		except:
			try:
				bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
				self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen, local_files_only = True, force_download = True)  
			except:
				bert_chosen = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
				self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  


	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID_img = self.list_IDs[index,0]
		y = int(self.list_IDs[index,1])
		ID_txt = self.list_IDs[index,2]
		
		input_tensor = torch.as_tensor(float(-1) , dtype=torch.long)
		encoded_text_short = torch.as_tensor(float(-1) , dtype=torch.long)
		encoded_text_abcd = torch.as_tensor(float(-1) , dtype=torch.long)
		encoded_text_doc = torch.as_tensor(float(-1) , dtype=torch.long)
		encoded_text_char = torch.as_tensor(float(-1) , dtype=torch.long)

		try:
			X = Image.open(ID_img)
			X = np.asarray(X)
			input_tensor = self.preprocess(X).type(torch.FloatTensor)
		except:
			pass

		try:
			if (self.flag_embeddings):
				ID_txt = ID_txt.replace('clinical_notes','pubmed_embeddings').split('.')[0]
				ID_txt = ID_txt + '.npy'

				encoded_text_short = np.load(ID_txt)#.reshape(1, -1)
				encoded_text_short = torch.from_numpy(encoded_text_short).float()

			else:

				input_txt = utils_txt.load_txt(ID_txt)
					
				encoded_text_short = self.tokenizer(
						input_txt,
						add_special_tokens=True,
						return_token_type_ids=True,
						return_attention_mask=True,
						padding="max_length",  # Pads all sequences to the max_length in the batch
						truncation=True,  # Ensures no sequence exceeds max length
						max_length=512,  # Adjust based on model constraints
						return_tensors="pt"  # Directly returns PyTorch tensors
					)
		except Exception as ex:
			print(ex)
			pass

		try:
			ID_txt = ID_txt.replace('short_reports','gpt_4o_mini_fld_abcd')

			if (self.flag_embeddings):
				ID_txt = ID_txt.replace('clinical_notes','pubmed_embeddings').split('.')[0]
				ID_txt = ID_txt + '.npy'

				encoded_text_abcd = np.load(ID_txt)#.reshape(1, -1)
				encoded_text_abcd = torch.from_numpy(encoded_text_abcd).float()

			else:
				input_txt = utils_txt.load_txt(ID_txt)
					
				encoded_text_abcd = self.tokenizer(
						input_txt,
						add_special_tokens=True,
						return_token_type_ids=True,
						return_attention_mask=True,
						padding="max_length",  # Pads all sequences to the max_length in the batch
						truncation=True,  # Ensures no sequence exceeds max length
						max_length=512,  # Adjust based on model constraints
						return_tensors="pt"  # Directly returns PyTorch tensors
					)
		except Exception as ex:
			print(ex)
			pass
		
		
		try:
			ID_txt = ID_txt.replace('gpt_4o_mini_fld_abcd','gpt_4o_mini_fld')

			if (self.flag_embeddings):
				ID_txt = ID_txt.replace('clinical_notes','pubmed_embeddings').split('.')[0]
				ID_txt = ID_txt + '.npy'

				encoded_text_char = np.load(ID_txt)#.reshape(1, -1)
				encoded_text_char = torch.from_numpy(encoded_text_char).float()

			else:
				input_txt = utils_txt.load_txt(ID_txt)
					
				encoded_text_char = self.tokenizer(
						input_txt,
						add_special_tokens=True,
						return_token_type_ids=True,
						return_attention_mask=True,
						padding="max_length",  # Pads all sequences to the max_length in the batch
						truncation=True,  # Ensures no sequence exceeds max length
						max_length=512,  # Adjust based on model constraints
						return_tensors="pt"  # Directly returns PyTorch tensors
					)
		except Exception as ex:
			print(ex)
			pass
		
		try:
			ID_txt = ID_txt.replace('gpt_4o_mini_fld','gpt_4_mini_as_doctor')

			if (self.flag_embeddings):
				ID_txt = ID_txt.replace('clinical_notes','pubmed_embeddings').split('.')[0]
				ID_txt = ID_txt + '.npy'

				encoded_text_doc = np.load(ID_txt)#.reshape(1, -1)
				encoded_text_doc = torch.from_numpy(encoded_text_doc).float()

			else:
				input_txt = utils_txt.load_txt(ID_txt)
					
				encoded_text_doc = self.tokenizer(
						input_txt,
						add_special_tokens=True,
						return_token_type_ids=True,
						return_attention_mask=True,
						padding="max_length",  # Pads all sequences to the max_length in the batch
						truncation=True,  # Ensures no sequence exceeds max length
						max_length=512,  # Adjust based on model constraints
						return_tensors="pt"  # Directly returns PyTorch tensors
					)
		except Exception as ex:
			print(ex)
			pass

		if (self.N_CLASSES > 1):
			y = torch.as_tensor(float(y) , dtype=torch.long)
		else:
			y = torch.as_tensor(float(y) , dtype=torch.float32)

		#return input_tensor
		#print(len(encoded_text))
		ID_img = ID_img.split('/')[-1]

		return input_tensor, encoded_text_short, encoded_text_abcd, encoded_text_char, encoded_text_doc, y, ID_img





class Dataset_generate_features_specific(data.Dataset):

	def __init__(self, list_IDs, phase, prob, classes, TYPE_DATA, flag_embeddings, CNN_TO_USE = "densenet121"):

		self.list_IDs = utils_data.labels2int(list_IDs)
		self.flag_embeddings = flag_embeddings
		self.mode = phase
		self.TYPE = TYPE_DATA

		self.prob = prob
		if (CNN_TO_USE == "ViT"):
			self.preprocess = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
			])
		else:
			self.preprocess = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			])


		self.N_CLASSES = classes

		self.patch_size = 224

		try:
			bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
			self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  
		except:
			try:
				bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
				self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen, local_files_only = True, force_download = True)  
			except:
				bert_chosen = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
				self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  


	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID_img = self.list_IDs[index,0]
		y = int(self.list_IDs[index,1])
		ID_txt = self.list_IDs[index,2]
		
		input_tensor = torch.as_tensor(float(-1) , dtype=torch.long)
		if (self.TYPE is REPORTS.images):
			try:
				X = Image.open(ID_img)
				X = np.asarray(X)
				input_tensor = self.preprocess(X).type(torch.FloatTensor)
			except:
				pass
		
		else:
			#print(ID_txt, "1")

			if (self.TYPE is REPORTS.abcd):
				ID_txt = ID_txt.replace('shorts','abcd')

			
			elif(self.TYPE is REPORTS.char):
				ID_txt = ID_txt.replace('shorts','char')

					
			elif(self.TYPE is REPORTS.doc):
				ID_txt = ID_txt.replace('shorts','doc')

			elif(self.TYPE is REPORTS.short):
				ID_txt = ID_txt.replace('shorts','shorts')

			####skingpt4
			elif (self.TYPE is REPORTS.skingpt4_abcd):
				ID_txt = ID_txt.replace('shorts','skingpt4_abcd')

			
			elif(self.TYPE is REPORTS.skingpt4_char):
				ID_txt = ID_txt.replace('shorts','skingpt4_char')

					
			elif(self.TYPE is REPORTS.skingpt4_doc):
				ID_txt = ID_txt.replace('shorts','skingpt4_doc')

			elif(self.TYPE is REPORTS.skingpt4_p1):
				ID_txt = ID_txt.replace('shorts','skingpt4_p1')

			elif(self.TYPE is REPORTS.skingpt4_p2):
				ID_txt = ID_txt.replace('shorts','skingpt4_p2')
			####dermlip
			elif (self.TYPE is REPORTS.dermlip_abcd):
				ID_txt = ID_txt.replace('shorts','derm_1M_abcd')

			
			elif(self.TYPE is REPORTS.dermlip_char):
				ID_txt = ID_txt.replace('shorts','derm_1M_char')

					
			elif(self.TYPE is REPORTS.dermlip_doc):
				ID_txt = ID_txt.replace('shorts','derm_1M_doc')

			elif(self.TYPE is REPORTS.dermlip_p1):
				ID_txt = ID_txt.replace('shorts','derm_1M_p1')

			elif(self.TYPE is REPORTS.dermlip_p2):
				ID_txt = ID_txt.replace('shorts','derm_1M_p2')

			####medgemma
			elif (self.TYPE is REPORTS.medgemma_abcd):
				ID_txt = ID_txt.replace('shorts','medgemma_abcd')

			
			elif(self.TYPE is REPORTS.medgemma_char):
				ID_txt = ID_txt.replace('shorts','medgemma_char')

					
			elif(self.TYPE is REPORTS.medgemma_doc):
				ID_txt = ID_txt.replace('shorts','medgemma_doc')

			else:

				raise ValueError("wrong type")

			#print(ID_txt, "2")

			try:

				if (self.flag_embeddings):
					ID_txt = ID_txt.replace('clinical_notes','pubmed_embeddings').split('.')[0]
					ID_txt = ID_txt + '.npy'

					#print(ID_txt)
					arr = np.load(ID_txt)
					arr = np.asarray(arr, dtype=np.float32).reshape(-1)   # force (768,) float32
					input_tensor = torch.tensor(arr, dtype=torch.float32) # independent storage
					
				else:
					input_txt = utils_txt.load_txt(ID_txt)
						
					input_tensor = self.tokenizer(
							input_txt,
							add_special_tokens=True,
							return_token_type_ids=True,
							return_attention_mask=True,
							padding="max_length",  # Pads all sequences to the max_length in the batch
							truncation=True,  # Ensures no sequence exceeds max length
							max_length=512,  # Adjust based on model constraints
							return_tensors="pt"  # Directly returns PyTorch tensors
						)
			except Exception as e:
				print(e)
				pass
				

		if (self.N_CLASSES > 1):
			y = torch.as_tensor(float(y) , dtype=torch.long)
		else:
			y = torch.as_tensor(float(y) , dtype=torch.float32)

		#return input_tensor
		#print(len(encoded_text))
		ID_img = ID_img.split('/')[-1]

		#print(input_tensor.shape)
		#print(y)
		#print(ID_img)
		
		return input_tensor, ID_img



class Dataset_instance_reports(data.Dataset):

	def __init__(self, list_IDs, classes):

		self.list_IDs = utils_data.labels2int(list_IDs)

		self.N_CLASSES = classes
		
		try:
			bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
			self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  
		except:
			try:
				bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
				self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen, local_files_only = True, force_download = True)  
			except:
				bert_chosen = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
				self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  


	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID_txt = self.list_IDs[index,0]
		y = int(self.list_IDs[index,1])
		report = self.list_IDs[index,2]

		input_txt = report
			
		encoded_text = self.tokenizer(
			input_txt,
			add_special_tokens=True,
			return_token_type_ids=True,
			return_attention_mask=True,
			padding="max_length",  # Pads all sequences to the max_length in the batch
			truncation=True,  # Ensures no sequence exceeds max length
			max_length=256,  # Adjust based on model constraints
			return_tensors="pt"  # Directly returns PyTorch tensors
		)

		if (self.N_CLASSES > 1):
			y = torch.as_tensor(float(y) , dtype=torch.long)
		else:
			y = torch.as_tensor(float(y) , dtype=torch.float32)

		#return input_tensor
		#print(len(encoded_text))
		return encoded_text, y, ID_txt



class Dataset_instance_reports_generated(data.Dataset):

	def __init__(self, list_IDs, classes, report_fld, flag_embeddings_pubmed = False):

		self.list_IDs = utils_data.labels2int(list_IDs)
		self.report_fld = report_fld
		self.N_CLASSES = classes
		self.flag_embeddings_pubmed = flag_embeddings_pubmed
		
		try:
			bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
			self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  
		except:
			try:
				bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
				self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen, local_files_only = True, force_download = True)  
			except:
				bert_chosen = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
				self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  


	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID_txt = self.list_IDs[index,0]
		y = int(self.list_IDs[index,1])
		
		report_path = self.report_fld + ID_txt.split(".")[0] + ".txt"

		if (self.flag_embeddings_pubmed):

			ID_features = report_path.replace('clinical_notes','pubmed_embeddings').split('.')[0]

			if ('short' in report_path):
				ID_features = ID_features.replace('short_reports','shorts')
				
			ID_features = ID_features + '.npy'

			encoded_text = np.load(ID_features)#.reshape(1, -1)
			encoded_text = torch.from_numpy(encoded_text).float()
			#encoded_text = torch.as_tensor(float(encoded_text) , dtype=torch.float32)

		else:

			input_txt = utils_txt.load_txt(report_path)

			encoded_text = self.tokenizer(
				input_txt,
				add_special_tokens=True,
				return_token_type_ids=True,
				return_attention_mask=True,
				padding="max_length",  # Pads all sequences to the max_length in the batch
				truncation=True,  # Ensures no sequence exceeds max length
				max_length=256,  # Adjust based on model constraints
				return_tensors="pt"  # Directly returns PyTorch tensors
			)

			if (self.N_CLASSES > 1):
				y = torch.as_tensor(float(y) , dtype=torch.long)
			else:
				y = torch.as_tensor(float(y) , dtype=torch.float32)

			#return input_tensor
			#print(len(encoded_text))
		return encoded_text, y, ID_txt


class Dataset_instance_txt_generation(data.Dataset):

	def __init__(self, list_IDs, possible_reports, PAD_ID, model, prob, phase, device, 
	MAX_SEQ_LEN = 64, only_imgs = False, flag_augment_reports = False, flag_augment_images = False,
	CNN_TO_USE = "densenet121"):


		self.list_IDs = list_IDs
		self.PAD_ID = PAD_ID
		self.MAX_SEQ_LEN = MAX_SEQ_LEN
		self.encoder = model
		self.prob = prob
		self.flag_augment_reports = flag_augment_reports
		self.flag_augment_images = flag_augment_images

		self.geometric_pipeline = color_transformation.get_pipeline_geometric(prob)
		self.color_pipeline = color_transformation.get_pipeline_color(prob)
		
		resize_transform = A.Resize(224, 224)

		self.possible_reports = possible_reports
		self.acceptable_reports = self.get_possible_reports()

		#"""
		self.augmentation_pipeline = A.Compose([
			# Geometric augmentations
			A.HorizontalFlip(p=self.prob),
			A.VerticalFlip(p=self.prob),
			A.RandomRotate90(p=self.prob),

			A.Affine(
				scale=(0.95, 1.10),           # zoom in/out
				translate_percent=(0.0, 0.1), # small shift
				shear=(-5, 5),             # moderate shearing
				rotate=(-5, 5),
				border_mode=cv2.BORDER_REFLECT,                   # reflect border to avoid artifacts
				fit_output=True,
				keep_ratio = True,
				p=self.prob / 2,
			),
			
			# Color augmentation (safe ranges)
			A.RGBShift(r_shift_limit=(-50,10), g_shift_limit=(-50,10), b_shift_limit=(-50,10), p=prob),
			
			#A.ColorJitter(
			#	brightness=0.4,
			#	contrast=0.4,
			#	saturation=0.4,
			#	hue=0.2,
			#	p=self.prob
			#),

			
			A.Lambda(
				image=lambda img, **kwargs: resize_transform.apply(img) if img.shape[:2] != (224, 224) else img,
				mask=lambda msk, **kwargs: resize_transform.apply_to_mask(msk) if msk.shape[:2] != (224, 224) else msk,
				p=1.0
			),
		])
		#"""

		self.mode = phase
		self.device = device
		self.only_imgs = only_imgs

		if (CNN_TO_USE == "ViT"):
			self.preprocess_img = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5, 0.5, 0.5],
										std=[0.5, 0.5, 0.5]),
			])
		else:
			self.preprocess_img = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406],
										std=[0.229, 0.224, 0.225]),
			])

		self.preprocess = transforms.Compose([
		transforms.ToTensor(),
		])

		try:
			bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
			self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  
		except:
			try:
				bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
				self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen, local_files_only = True, force_download = True)  
			except:
				bert_chosen = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
				self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  

	def get_possible_reports(self):

		list_possible = []

		if (self.possible_reports is REPORTS.all or self.possible_reports is REPORTS.random):
			list_possible = ['abcd', 'char', 'shorts']
			#list_possible = ['gpt_4_mini_as_doctor', 'gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.abcd):
			list_possible = ['abcd']

		elif (self.possible_reports is REPORTS.char):
			list_possible = ['char']

		elif (self.possible_reports is REPORTS.doc):
			list_possible = ['doc']

		elif (self.possible_reports is REPORTS.short):
			list_possible = ['shorts']

		elif (self.possible_reports is REPORTS.meta):
			list_possible = ['abcd', 'char']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']


		###skingpt4
		elif (self.possible_reports is REPORTS.skingpt4_abcd):
			list_possible = ['skingpt4_abcd']

		elif (self.possible_reports is REPORTS.skingpt4_char):
			list_possible = ['skingpt4_char']

		elif (self.possible_reports is REPORTS.skingpt4_doc):
			list_possible = ['skingpt4_doc']

		elif (self.possible_reports is REPORTS.skingpt4_meta):
			list_possible = ['skingpt4_abcd', 'skingpt4_char']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.skingpt4_all):
			list_possible = ['skingpt4_abcd', 'skingpt4_char', 'shorts']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.skingpt4_p1):
			list_possible = ['skingpt4_p1']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.skingpt4_p1_all):
			list_possible = ['skingpt4_p1', 'shorts']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.skingpt4_p2):
			list_possible = ['skingpt4_p2']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		###dermlip
		elif (self.possible_reports is REPORTS.dermlip_abcd):
			list_possible = ['derm_1M_abcd']

		elif (self.possible_reports is REPORTS.dermlip_char):
			list_possible = ['derm_1M_char']

		elif (self.possible_reports is REPORTS.dermlip_doc):
			list_possible = ['derm_1M_doc']

		elif (self.possible_reports is REPORTS.dermlip_meta):
			list_possible = ['derm_1M_abcd', 'derm_1M_char']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.dermlip_all):
			list_possible = ['derm_1M_abcd', 'derm_1M_char', 'shorts']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.dermlip_p1):
			list_possible = ['derm_1M_p1']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.dermlip_p1_all):
			list_possible = ['derm_1M_p1', 'shorts']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.dermlip_p2):
			list_possible = ['derm_1M_p2']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		###medgemma
		elif (self.possible_reports is REPORTS.medgemma_abcd):
			list_possible = ['medgemma_abcd']

		elif (self.possible_reports is REPORTS.medgemma_char):
			list_possible = ['medgemma_char']

		elif (self.possible_reports is REPORTS.medgemma_doc):
			list_possible = ['medgemma_doc']

		elif (self.possible_reports is REPORTS.medgemma_meta):
			list_possible = ['medgemma_abcd', 'medgemma_char']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.medgemma_all):
			list_possible = ['medgemma_abcd', 'medgemma_char', 'shorts']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']


		elif (self.possible_reports is REPORTS.whole):
			list_possible = ['medgemma_abcd', 'medgemma_char','derm_1M_p1', 'skingpt4_p1', 'abcd', 'char']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.whole_all):
			list_possible = ['medgemma_abcd', 'medgemma_char','derm_1M_p1', 'skingpt4_p1', 'abcd', 'char', 'shorts']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		return list_possible

	def __len__(self):
		return len(self.list_IDs)

	
	def get_augmented_reports(ID_txt):

		augmented_txt = None

		# Split the path
		parent, child = os.path.split(ID_txt)           # parent = "/XXX/YYY/TTT", child = "ZZZ"
		grandparent, ttt = os.path.split(parent)      # grandparent = "/XXX/YYY", ttt = "TTT"

		# Modify the TTT folder name
		new_ttt = ttt + "_aug"

		# Reconstruct the new path
		new_path = os.path.join(grandparent, new_ttt, child)

		new_path = new_path.split('.')[0] + '.json'

		with open(new_path, 'r') as f:
			data_aug = json.load(f)

		data_aug = data_aug["variations"]

		idx = np.random.randint(0,len(data_aug))
		augmented_txt = data_aug[idx]

		return augmented_txt


	def tokenize_texts(self, texts, PAD_ID, max_length=64):
		encodings = self.tokenizer(
				texts,
				add_special_tokens=True,
				return_token_type_ids=True,
				return_attention_mask=True,
				padding="max_length",  # Pads all sequences to the max_length in the batch
				truncation=True,  # Ensures no sequence exceeds max length
				max_length=max_length,  # Adjust based on model constraints
				return_tensors="pt"  # Directly returns PyTorch tensors
			)


		input_ids = encodings["input_ids"]
		input_ids = input_ids.squeeze(1)
		target_ids = input_ids.clone()
		target_ids[target_ids == PAD_ID] = -100  # ignore index for loss
		return input_ids, target_ids

	def get_path_img(self, ID_img):
		
		suffix = ['','.jpg','.png','.jpeg']
		i = 0
		b = False

		while (i < len(suffix) and b == False):

			path_img = ID_img + suffix[i]

			if (os.path.exists(path_img)):
				b = True
			else:
				i = i + 1

		return path_img

	def get_features(self, ID_img):
		
		img_path = self.get_path_img(ID_img)

		X = Image.open(img_path)
		X = np.asarray(X)


		if (self.mode is PHASE.train and self.flag_augment_images):
			X = self.augmentation_pipeline(image=X)['image']

		input_tensor = self.preprocess(X).type(torch.FloatTensor)
		input_tensor = input_tensor.to(self.device)
		input_tensor = input_tensor.unsqueeze(0)

		with torch.autocast(device_type='cuda', dtype=torch.float16):
			_, cls_img, _, _ = self.encoder(input_tensor, None)
			cls_img = cls_img.cpu()

		cls_img = cls_img.squeeze(0)
		return cls_img

	def __getitem__(self, index):
		# Select sample
		sample_ID = self.list_IDs[index,0]
		feature_path = self.list_IDs[index,1]
		try:
			ID_txt = self.list_IDs[index,2]
		except:
			ID_txt = None
			
		encoded_text = torch.as_tensor(float(-1) , dtype=torch.long)
		target_text = torch.as_tensor(float(-1) , dtype=torch.long)
		 
		prob_pre = np.random.rand(1)[0]
		
		if (prob_pre > self.prob and self.mode is PHASE.train and self.only_imgs == False and self.flag_augment_images == True):
			cls_sample = self.get_features(sample_ID)

		else:

			with open(feature_path, 'rb') as f:
				cls_sample = np.load(f)

			cls_sample = torch.from_numpy(cls_sample)
			
		cls_sample = cls_sample.float()

		if (self.only_imgs == False):
			
			if (self.possible_reports is REPORTS.random):
				
				rand_idx = random.randint(0, len(self.list_IDs) - 1)
				ID_txt = self.list_IDs[rand_idx,2]

			new_report_type = random.choice(self.acceptable_reports)
			report_path = ID_txt.replace('short_reports',new_report_type)

			prob_pre = np.random.rand(1)[0]
				
			if (prob_pre >= self.prob and self.mode is PHASE.train and self.flag_augment_reports == True):
				try:
					input_txt = get_augmented_reports(report_path)
				except:
					input_txt = utils_txt.load_txt(report_path)
			else:
				input_txt = utils_txt.load_txt(report_path)

			encoded_text, target_text = self.tokenize_texts(input_txt, self.PAD_ID, self.MAX_SEQ_LEN)

		ID_img = sample_ID.split('/')[-1]

		return cls_sample, encoded_text, target_text, ID_img



class Dataset_eval_txt_generation(data.Dataset):

	def __init__(self, list_IDs, FLD_to_test, FLD_to_compare):

		self.list_IDs = list_IDs
		self.FLD_to_test = FLD_to_test
		self.FLD_to_compare = FLD_to_compare
		


	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID_img = self.list_IDs[index]
		
		fname = ID_img.split('.')[0]

		fname_to_test = self.FLD_to_test + fname + '.txt'
		fname_to_compare = self.FLD_to_compare + fname + '.txt'
		

		input_test = utils_txt.load_txt(fname_to_test)
		input_compared = utils_txt.load_txt(fname_to_compare)


		return fname, input_test, input_compared

def load_tokenizer():
	bert_candidates = [
		('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', {"local_files_only": False}),
		('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', {"local_files_only": True, "force_download": True}),
		('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract', {})
	]
	for name, kwargs in bert_candidates:
		try:
			return AutoTokenizer.from_pretrained(name, **kwargs)
		except Exception:
			continue
	raise RuntimeError("Failed to load a tokenizer.")


class Dataset_instance_img_generation(data.Dataset):

	def __init__(self, list_IDs, model, prob, phase, device, TYPE_DOC=False):
		self.list_IDs = list_IDs
		self.encoder = model
		self.encoder.eval()
		self.device = device
		self.mode = phase
		self.TYPE_DOC = TYPE_DOC
		self.prob = prob

		self.augmentation_pipeline = A.Compose([
			# Geometric augmentations
			A.HorizontalFlip(p=0.5),
			A.VerticalFlip(p=0.5),
			A.RandomRotate90(p=0.5),
			A.Affine(
				scale=(0.8, 1.2),           # zoom in/out
				translate_percent=(0.0, 0.2), # small shift
				shear=(-10, 10),             # moderate shearing
				rotate=(-15, 15),
				border_mode=cv2.BORDER_REFLECT,                   # reflect border to avoid artifacts
				fit_output=True,
				p=0.5,
			),
			
			# Color augmentation (safe ranges)
			A.ColorJitter(
				brightness=0.4,
				contrast=0.4,
				saturation=0.4,
				hue=0.2,
				p=0.5
			),
			
			# Slight blur or noise (optional, avoid excessive)
			A.OneOf([
				A.GaussianBlur(blur_limit=(3, 5), p=0.3),
				A.GaussNoise(std_range=(0.1, 0.2), p=0.3),
			], p=0.2),
			
			# Resize back to desired input size
			A.Resize(224, 224),
		])

		self.preprocess_img = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225]),
		])

		self.preprocess = transforms.ToTensor()

		# Load tokenizer
		bert_options = [
			'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
			'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
		]
		for bert_model in bert_options:
			try:
				self.tokenizer = AutoTokenizer.from_pretrained(
					bert_model, local_files_only=True, force_download=True)
				break
			except:
				continue
		else:
			self.tokenizer = AutoTokenizer.from_pretrained(bert_options[0])

	def __len__(self):
		return len(self.list_IDs)

	def get_path_img(self, ID_img):
		for ext in ['', '.jpg', '.png', '.jpeg']:
			path_img = ID_img + ext
			if os.path.exists(path_img):
				return path_img
		raise FileNotFoundError(f"No valid image file found for {ID_img}")

	def get_features_img(self, X_np):
		if self.mode == PHASE.train:
			X_np = self.augmentation_pipeline(image=X_np)['image']

		input_tensor = self.preprocess(X_np).float().unsqueeze(0).to(self.device)

		with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
			_, cls_img, _, _ = self.encoder(input_tensor, None)

		return input_tensor.squeeze(0).cpu(), cls_img.squeeze(0).float()

	def get_features_txt(self, ID_txt):
		# Generate new JSON path
		parent_dir, file_name = os.path.split(ID_txt)
		grandparent, subfolder = os.path.split(parent_dir)
		aug_dir = os.path.join(grandparent, subfolder + "_aug", file_name)
		aug_json_path = aug_dir.split('.')[0] + '.json'

		with open(aug_json_path, 'r') as f:
			data_aug = json.load(f)["variations"]

		random_txt = data_aug[np.random.randint(0, len(data_aug))]

		encodings = self.tokenizer(
			random_txt,
			add_special_tokens=True,
			return_token_type_ids=True,
			return_attention_mask=True,
			padding="max_length",
			truncation=True,
			max_length=512,
			return_tensors="pt"
		).to(self.device)

		with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
			_, _, _, cls_txt = self.encoder(None, encodings)

		return cls_txt

	def __getitem__(self, index):
		ID_img, feature_path, ID_txt = self.list_IDs[index]
		img_path = self.get_path_img(ID_img)

		# Load and convert image
		img = Image.open(img_path).convert("RGB")
		img_np = np.asarray(img)

		is_train = (self.mode == PHASE.train)
		do_img_aug = is_train and np.random.rand() >= 0.5

		# Handle image features
		if do_img_aug:
			tgt_tensor, cls_img = self.get_features_img(img_np)
		else:
			tgt_tensor = self.preprocess(img_np).float()
			with open(feature_path, 'rb') as f:
				cls_img = np.load(f)

		# Handle text features
		do_txt_aug = is_train and np.random.rand() >= 0.5
		if do_txt_aug:
			rand_txt = np.random.rand()
			modified = False
			if 0.33 <= rand_txt < 0.66:
				ID_txt = ID_txt.replace('short_reports', 'gpt_4o_mini_fld_abcd')
				modified = True
			elif rand_txt >= 0.66:
				ID_txt = ID_txt.replace('short_reports', 'gpt_4o_mini_fld')
				modified = True

			if modified:
				cls_txt = self.get_features_txt(ID_txt)
			else:
				report_path = feature_path.replace('images', 'reports_shorts')
				with open(report_path, 'rb') as f:
					cls_txt = np.load(f)
		else:
			if not self.TYPE_DOC:
				rand_doc = np.random.rand()
				if rand_doc < 0.33:
					report_dir = 'reports_abcd'
				elif rand_doc < 0.66:
					report_dir = 'reports_char'
				else:
					report_dir = 'reports_shorts'
			else:
				report_dir = 'reports_doc'

			report_path = feature_path.replace('images', report_dir)
			with open(report_path, 'rb') as f:
				cls_txt = np.load(f)

		cls_img = torch.from_numpy(cls_img).float().to(self.device) if isinstance(cls_img, np.ndarray) else cls_img.to(self.device)
		cls_txt = torch.from_numpy(cls_txt).float().to(self.device) if isinstance(cls_txt, np.ndarray) else cls_txt.to(self.device)

		# Squeeze if shape is (1,128)
		if cls_img.ndim == 2 and cls_img.shape[0] == 1:
			cls_img = cls_img.squeeze(0)
		if cls_txt.ndim == 2 and cls_txt.shape[0] == 1:
			cls_txt = cls_txt.squeeze(0)

		ID_img = ID_img.split('/')[-1]

		return tgt_tensor, cls_img, cls_txt, ID_img



def denormalize_image_batch(tensor_batch):
	"""
	Input: tensor of shape (B, 3, 224, 224) with values in [0, 1]
	Output: numpy array of shape (B, 224, 224, 3) with uint8 in [0, 255]
	"""
	batch = tensor_batch.clamp(0, 1).cpu()  # Ensure values are in [0,1]
	batch = batch * 255.0
	batch = batch.permute(0, 2, 3, 1)  # (B, H, W, C)
	batch = batch.detach().numpy().astype('uint8')
	return batch


class Dataset_instance_concept(data.Dataset):

	def __init__(self, list_IDs, phase, prob, 
			  classes, possible_reports, 
			  flag_augment_reports = False,
			  components = [COMPONENTS.images, COMPONENTS.reports, COMPONENTS.keywords],
			  flag_embeddings_pubmed = False, CNN_TO_USE = "densenet121"
			  ):
		self.list_IDs = list_IDs

		self.mode = phase
		self.prob = prob
		self.N_CLASSES = classes
		self.possible_reports = possible_reports
		self.acceptable_reports = self.get_possible_reports()
		
		self.flag_augment_reports = flag_augment_reports
		

		self.components = components
		self.flag_embeddings_pubmed = flag_embeddings_pubmed
		
		self.geometric_pipeline = color_transformation.get_pipeline_geometric(prob)
		self.color_pipeline = color_transformation.get_pipeline_color(prob)

		resize_transform = A.Resize(224, 224)

		#"""
		self.augmentation_pipeline = A.Compose([
			# Geometric augmentations
			A.HorizontalFlip(p=self.prob),
			A.VerticalFlip(p=self.prob),
			A.RandomRotate90(p=self.prob),

			A.Affine(
				scale=(0.95, 1.10),           # zoom in/out
				translate_percent=(0.0, 0.1), # small shift
				shear=(-5, 5),             # moderate shearing
				rotate=(-5, 5),
				border_mode=cv2.BORDER_REFLECT,                   # reflect border to avoid artifacts
				fit_output=True,
				keep_ratio = True,
				p=self.prob / 2,
			),
			
			# Color augmentation (safe ranges)
			A.RGBShift(r_shift_limit=(-50,10), g_shift_limit=(-50,10), b_shift_limit=(-50,10), p=prob),
			
			#A.ColorJitter(
			#	brightness=0.4,
			#	contrast=0.4,
			#	saturation=0.4,
			#	hue=0.2,
			#	p=self.prob
			#),

			
			A.Lambda(
				image=lambda img, **kwargs: resize_transform.apply(img) if img.shape[:2] != (224, 224) else img,
				mask=lambda msk, **kwargs: resize_transform.apply_to_mask(msk) if msk.shape[:2] != (224, 224) else msk,
				p=1.0
			),
		])
		#"""

		if (CNN_TO_USE == "ViT"):
			self.preprocess = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5, 0.5, 0.5],
										std=[0.5, 0.5, 0.5]),
			])
		else:
			self.preprocess = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406],
										std=[0.229, 0.224, 0.225]),
			])

		if (COMPONENTS.reports in self.components):
			# Load tokenizer
			bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
			try:
				
				self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  
			except:
				try:
					self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen, local_files_only = True, force_download = True)  
				except:
					self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  
			
	def get_possible_reports(self):


		list_possible = []

		if (self.possible_reports is REPORTS.all or self.possible_reports is REPORTS.random):
			list_possible = ['abcd', 'char', 'shorts']
			#list_possible = ['gpt_4_mini_as_doctor', 'gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.abcd):
			list_possible = ['abcd']

		elif (self.possible_reports is REPORTS.char):
			list_possible = ['char']

		elif (self.possible_reports is REPORTS.doc):
			list_possible = ['doc']

		elif (self.possible_reports is REPORTS.short):
			list_possible = ['shorts']

		elif (self.possible_reports is REPORTS.meta):
			list_possible = ['abcd', 'char']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']


		###skingpt4
		elif (self.possible_reports is REPORTS.skingpt4_abcd):
			list_possible = ['skingpt4_abcd']

		elif (self.possible_reports is REPORTS.skingpt4_char):
			list_possible = ['skingpt4_char']

		elif (self.possible_reports is REPORTS.skingpt4_doc):
			list_possible = ['skingpt4_doc']

		elif (self.possible_reports is REPORTS.skingpt4_meta):
			list_possible = ['skingpt4_abcd', 'skingpt4_char']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.skingpt4_all):
			list_possible = ['skingpt4_abcd', 'skingpt4_char', 'shorts']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.skingpt4_p1):
			list_possible = ['skingpt4_p1']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.skingpt4_p1_all):
			list_possible = ['skingpt4_p1', 'shorts']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.skingpt4_p2):
			list_possible = ['skingpt4_p2']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		###dermlip
		elif (self.possible_reports is REPORTS.dermlip_abcd):
			list_possible = ['derm_1M_abcd']

		elif (self.possible_reports is REPORTS.dermlip_char):
			list_possible = ['derm_1M_char']

		elif (self.possible_reports is REPORTS.dermlip_doc):
			list_possible = ['derm_1M_doc']

		elif (self.possible_reports is REPORTS.dermlip_meta):
			list_possible = ['derm_1M_abcd', 'derm_1M_char']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.dermlip_all):
			list_possible = ['derm_1M_abcd', 'derm_1M_char', 'shorts']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.dermlip_p1):
			list_possible = ['derm_1M_p1']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.dermlip_p1_all):
			list_possible = ['derm_1M_p1', 'shorts']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.dermlip_p2):
			list_possible = ['derm_1M_p2']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		###medgemma
		elif (self.possible_reports is REPORTS.medgemma_abcd):
			list_possible = ['medgemma_abcd']

		elif (self.possible_reports is REPORTS.medgemma_char):
			list_possible = ['medgemma_char']

		elif (self.possible_reports is REPORTS.medgemma_doc):
			list_possible = ['medgemma_doc']

		elif (self.possible_reports is REPORTS.medgemma_meta):
			list_possible = ['medgemma_abcd', 'medgemma_char']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.medgemma_all):
			list_possible = ['medgemma_abcd', 'medgemma_char', 'shorts']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']


		elif (self.possible_reports is REPORTS.whole):
			list_possible = ['medgemma_abcd', 'medgemma_char','derm_1M_p1', 'skingpt4_p1', 'abcd', 'char']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.whole_all):
			list_possible = ['medgemma_abcd', 'medgemma_char','derm_1M_p1', 'skingpt4_p1', 'abcd', 'char', 'shorts']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		return list_possible

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID_img = self.list_IDs[index,0]
		y = int(self.list_IDs[index,1])

		try:
			ID_txt = self.list_IDs[index,2]
		except:
			ID_txt = None
			
		input_tensor = torch.as_tensor(float(-1) , dtype=torch.long)
		encoded_text = torch.as_tensor(float(-1) , dtype=torch.long)
		one_hot_concepts = torch.zeros((1, len(utils_concept_extraction.flattened_concepts)))


		if (COMPONENTS.images in self.components):
			X = Image.open(ID_img)
			X = np.asarray(X)

			if (self.mode is PHASE.train):
				prob_pre = np.random.rand(1)[0]
				"""
				if (prob_pre >= 0.33):
					X = self.geometric_pipeline(image=X)['image']
					X = self.color_pipeline(image=X)['image']
				else:
					X = self.augmentation_pipeline(image=X)['image']
				"""
				X = self.geometric_pipeline(image=X)['image']
				X = self.color_pipeline(image=X)['image']
				
				#X = self.augmentation_pipeline(image=X)['image']

			input_tensor = self.preprocess(X).type(torch.FloatTensor)

		if (COMPONENTS.reports in self.components):

			#random
			if (self.possible_reports is REPORTS.random):
				
				rand_idx = random.randint(0, len(self.list_IDs) - 1)
				ID_txt = self.list_IDs[rand_idx,2]
			
			new_report_type = random.choice(self.acceptable_reports)
			ID_txt = ID_txt.replace('shorts',new_report_type)

			#print(ID_txt)
			#print(ID_txt)
			if (self.flag_embeddings_pubmed and self.flag_augment_reports == False):

				ID_features = ID_txt.replace('clinical_notes','pubmed_embeddings').split('.')[0]
				ID_features = ID_features + '.npy'

				encoded_text = np.load(ID_features)#.reshape(1, -1)
				encoded_text = torch.from_numpy(encoded_text).float()
				#encoded_text = torch.as_tensor(float(encoded_text) , dtype=torch.float32)


			else:		

				
				input_txt = utils_txt.load_txt(ID_txt)

				if (self.mode is PHASE.train and self.flag_augment_reports):

					prob_pre = np.random.rand(1)[0]
					
					
					if (prob_pre >= 0.5):
						try:
							input_txt = utils_SD_text.sample_lines(input_txt)
						except:
							pass


				encoded_text = self.tokenizer(
					input_txt,
					add_special_tokens=True,
					return_token_type_ids=True,
					return_attention_mask=True,
					padding="max_length",  # Pads all sequences to the max_length in the batch
					truncation=True,  # Ensures no sequence exceeds max length
					max_length=256,  # Adjust based on model constraints
					return_tensors="pt"  # Directly returns PyTorch tensors
				)

		if (COMPONENTS.keywords in self.components):
			######
			try:
				concepts = utils_concept_extraction.find_present_concepts(input_txt, utils_concept_extraction.GLOBAL_CONCEPTS, utils_concept_extraction.CONFLICTS)
			except:
				
				input_txt = utils_txt.load_txt(ID_txt)
				concepts = utils_concept_extraction.find_present_concepts(input_txt, utils_concept_extraction.GLOBAL_CONCEPTS, utils_concept_extraction.CONFLICTS)


			concepts = utils_concept_extraction.flatten_and_remove(concepts, value_to_remove = -1)
			one_hot_concepts = utils_concept_extraction.encode_flat(concepts, utils_concept_extraction.concept_to_index)
			
			one_hot_concepts = torch.tensor(one_hot_concepts, dtype=torch.long)
			
		if (self.N_CLASSES > 1):
			y = torch.as_tensor(float(y) , dtype=torch.long)
		else:
			y = torch.as_tensor(float(y) , dtype=torch.float32)

		ID_img = ID_img.split('/')[-1]

		return input_tensor, encoded_text, y, one_hot_concepts, ID_img



class Dataset_instance_concept_augmentation(data.Dataset):

	def __init__(self, list_IDs, phase, prob, 
			  classes, possible_reports, 
			  flag_augment_reports = False,
			  components = [COMPONENTS.images, COMPONENTS.reports, COMPONENTS.keywords],
			  flag_embeddings_pubmed = False
			  ):
		self.list_IDs = list_IDs

		self.mode = phase
		self.prob = prob
		self.N_CLASSES = classes
		self.possible_reports = possible_reports
		self.acceptable_reports = self.get_possible_reports()
		
		self.flag_augment_reports = flag_augment_reports
		self.geometric_pipeline = color_transformation.get_pipeline_geometric(prob)
		self.color_pipeline = color_transformation.get_pipeline_color(prob)

		self.components = components
		self.flag_embeddings_pubmed = flag_embeddings_pubmed

		resize_transform = A.Resize(224, 224)

		#"""
		self.augmentation_pipeline = A.Compose([
			# Geometric augmentations
			A.HorizontalFlip(p=self.prob),
			A.VerticalFlip(p=self.prob),
			A.RandomRotate90(p=self.prob),

			A.Affine(
				scale=(0.95, 1.10),           # zoom in/out
				translate_percent=(0.0, 0.1), # small shift
				shear=(-5, 5),             # moderate shearing
				rotate=(-5, 5),
				border_mode=cv2.BORDER_REFLECT,                   # reflect border to avoid artifacts
				fit_output=True,
				keep_ratio = True,
				p=self.prob / 2,
			),
			
			# Color augmentation (safe ranges)
			A.RGBShift(r_shift_limit=(-50,10), g_shift_limit=(-50,10), b_shift_limit=(-50,10), p=prob),
						
			A.Lambda(
				image=lambda img, **kwargs: resize_transform.apply(img) if img.shape[:2] != (224, 224) else img,
				mask=lambda msk, **kwargs: resize_transform.apply_to_mask(msk) if msk.shape[:2] != (224, 224) else msk,
				p=1.0
			),
		])
		#"""

		self.preprocess = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225]),
		])

		if (COMPONENTS.reports in self.components):
			# Load tokenizer
			bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
			try:
				
				self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  
			except:
				try:
					self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen, local_files_only = True, force_download = True)  
				except:
					self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  
			
	def get_possible_reports(self):


		list_possible = []

		if (self.possible_reports is REPORTS.all or self.possible_reports is REPORTS.random):
			list_possible = ['gpt_4_mini_as_doctor', 'gpt_4o_mini_fld_abcd', 'gpt_4o_mini_fld', 'short_reports']
			#list_possible = ['gpt_4_mini_as_doctor', 'gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.abcd):
			list_possible = ['gpt_4o_mini_fld_abcd']

		elif (self.possible_reports is REPORTS.char):
			list_possible = ['gpt_4o_mini_fld']

		elif (self.possible_reports is REPORTS.doc):
			list_possible = ['gpt_4_mini_as_doctor']

		elif (self.possible_reports is REPORTS.short):
			list_possible = ['short_reports']

		elif (self.possible_reports is REPORTS.meta):
			list_possible = ['gpt_4o_mini_fld_abcd', 'gpt_4o_mini_fld', 'short_reports']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.abcd_o4):
			list_possible = ['gpt_4o_fld_abcd']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.char_o4):
			list_possible = ['gpt_4o_fld']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.doc_o4):
			list_possible = ['gpt_4o_fld_as_doctor']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.meta_o4):
			list_possible = ['gpt_4o_fld_abcd', 'gpt_4o_fld', 'short_reports']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.all_o4):
			list_possible = ['gpt_4o_mini_fld_abcd', 'gpt_4o_mini_fld', 'gpt_4o_fld_as_doctor', 'short_reports']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.dermlip):
			list_possible = ['derm_1M_class']
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (self.possible_reports is REPORTS.skingpt4):
			list_possible = ['skingpt4_llama']

		elif (self.possible_reports is REPORTS.skingpt4_spec):
			list_possible = ['skingpt4_llama_specific']

		return list_possible

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID_img = self.list_IDs[index,0]
		y = int(self.list_IDs[index,1])

		try:
			ID_txt = self.list_IDs[index,2]
		except:
			ID_txt = None
			
		input_tensor = torch.as_tensor(float(-1) , dtype=torch.long)
		encoded_text = torch.as_tensor(float(-1) , dtype=torch.long)
		one_hot_concepts = torch.zeros((1, len(utils_concept_extraction.flattened_concepts)))


		if (COMPONENTS.images in self.components):
			X = Image.open(ID_img)
			X = np.asarray(X)

			if (self.mode is PHASE.train):
				prob_pre = np.random.rand(1)[0]
				"""
				if (prob_pre >= 0.33):
					X = self.geometric_pipeline(image=X)['image']
					X = self.color_pipeline(image=X)['image']
				else:
					X = self.augmentation_pipeline(image=X)['image']
				"""
				X = self.geometric_pipeline(image=X)['image']
				X = self.color_pipeline(image=X)['image']
				
				#X = self.augmentation_pipeline(image=X)['image']

			input_tensor = self.preprocess(X).type(torch.FloatTensor)

		if (COMPONENTS.reports in self.components):

			#random
			if (self.possible_reports is REPORTS.random):
				
				rand_idx = random.randint(0, len(self.list_IDs) - 1)
				ID_txt = self.list_IDs[rand_idx,2]

			new_report_type = random.choice(self.acceptable_reports)
			ID_txt = ID_txt.replace('short_reports',new_report_type)

			#if (self.flag_embeddings_pubmed and self.flag_augment_reports == False):
			if (self.flag_embeddings_pubmed):
				ID_features = ID_txt.replace('clinical_notes','pubmed_embeddings').split('.')[0]
				ID_features = ID_features + '.npy'

				encoded_text = np.load(ID_features)#.reshape(1, -1)
				encoded_text = torch.from_numpy(encoded_text).float()
				#encoded_text = torch.as_tensor(float(encoded_text) , dtype=torch.float32)


			else:	
				#print(ID_img, ID_txt)
				if (self.mode is PHASE.train and self.flag_augment_reports):
					prob_pre = np.random.rand(1)[0]
					
					if (prob_pre >= 0.5):
						try:
							input_txt = get_augmented_reports(ID_txt)
						except:
							input_txt = utils_txt.load_txt(ID_txt)
					else:
						input_txt = utils_txt.load_txt(ID_txt)
						
				else:
					input_txt = utils_txt.load_txt(ID_txt)
				#print(input_txt)

				encoded_text = self.tokenizer(
					input_txt,
					add_special_tokens=True,
					return_token_type_ids=True,
					return_attention_mask=True,
					padding="max_length",  # Pads all sequences to the max_length in the batch
					truncation=True,  # Ensures no sequence exceeds max length
					max_length=512,  # Adjust based on model constraints
					return_tensors="pt"  # Directly returns PyTorch tensors
				)

		if (COMPONENTS.keywords in self.components):
			######
			try:
				concepts = utils_concept_extraction.find_present_concepts(input_txt, utils_concept_extraction.GLOBAL_CONCEPTS, utils_concept_extraction.CONFLICTS)
			except:
				
				input_txt = utils_txt.load_txt(ID_txt)
				concepts = utils_concept_extraction.find_present_concepts(input_txt, utils_concept_extraction.GLOBAL_CONCEPTS, utils_concept_extraction.CONFLICTS)


			concepts = utils_concept_extraction.flatten_and_remove(concepts, value_to_remove = -1)
			one_hot_concepts = utils_concept_extraction.encode_flat(concepts, utils_concept_extraction.concept_to_index)
			
			one_hot_concepts = torch.tensor(one_hot_concepts, dtype=torch.long)
			
		if (self.N_CLASSES > 1):
			y = torch.as_tensor(float(y) , dtype=torch.long)
		else:
			y = torch.as_tensor(float(y) , dtype=torch.float32)

		ID_img = ID_img.split('/')[-1]

		return input_tensor, encoded_text, y, one_hot_concepts, ID_img


class Dataset_reports_only(data.Dataset):

	def __init__(self, list_IDs, possible_reports):

		self.list_IDs = utils_data.labels2int(list_IDs)
		self.possible_report = self.get_possible_reports(possible_reports)
		
		try:
			bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
			self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  
		except:
			try:
				bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
				self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen, local_files_only = True, force_download = True)  
			except:
				bert_chosen = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
				self.tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  

	def get_possible_reports(self, possible_reports):


		list_possible = []

		if (possible_reports is REPORTS.abcd):
			list_possible = 'abcd'

		elif (possible_reports is REPORTS.char):
			list_possible = 'char'

		elif (possible_reports is REPORTS.doc):
			list_possible = 'doc'

		elif (possible_reports is REPORTS.short):
			list_possible = 'shorts'


		elif (possible_reports is REPORTS.skingpt4_abcd):
			list_possible = 'skingpt4_abcd'

		elif (possible_reports is REPORTS.skingpt4_char):
			list_possible = 'skingpt4_char'

		elif (possible_reports is REPORTS.skingpt4_doc):
			list_possible = 'skingpt4_doc'

		elif (possible_reports is REPORTS.skingpt4_p1):
			list_possible = 'skingpt4_p1'

		elif (possible_reports is REPORTS.skingpt4_p2):
			list_possible = 'skingpt4_p2'

		elif (possible_reports is REPORTS.dermlip_abcd):
			list_possible = 'derm_1M_abcd'

		elif (possible_reports is REPORTS.dermlip_char):
			list_possible = 'derm_1M_char'

		elif (possible_reports is REPORTS.dermlip_doc):
			list_possible = 'derm_1M_doc'

		elif (possible_reports is REPORTS.dermlip_p1):
			list_possible = 'derm_1M_p1'

		elif (possible_reports is REPORTS.dermlip_p2):
			list_possible = 'derm_1M_p2'

		elif (possible_reports is REPORTS.medgemma_abcd):
			list_possible = 'medgemma_abcd'

		elif (possible_reports is REPORTS.medgemma_char):
			list_possible = 'medgemma_char'

		elif (possible_reports is REPORTS.medgemma_doc):
			list_possible = 'medgemma_doc'

		
		return list_possible

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID_txt = self.list_IDs[index,2]
		ID_img = self.list_IDs[index,0]

		encoded_text = torch.as_tensor(float(-1) , dtype=torch.long)

		ID_txt = ID_txt.replace('short_reports',self.possible_report)
			
		input_txt = utils_txt.load_txt(ID_txt)

		encoded_text = self.tokenizer(
			input_txt,
			add_special_tokens=True,
			return_token_type_ids=True,
			return_attention_mask=True,
			padding="max_length",  # Pads all sequences to the max_length in the batch
			truncation=True,  # Ensures no sequence exceeds max length
			max_length=512,  # Adjust based on model constraints
			return_tensors="pt"  # Directly returns PyTorch tensors
		)

		ID_img = ID_img.split('/')[-1]

		return encoded_text, ID_img

if __name__ == "__main__":
	pass