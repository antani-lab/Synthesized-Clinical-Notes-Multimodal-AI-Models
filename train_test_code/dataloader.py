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
import utils_data
from scipy.spatial import KDTree
from skimage.color import rgb2hsv, hsv2rgb, rgb2lab, lab2rgb
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
	"""Samples elements randomly from a given list of indices for imbalanced dataset
	Arguments:
		indices (list, optional): a list of indices
		num_samples (int, optional): number of samples to draw
	"""

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



def filter_labels(arr, labels_to_remove, n_classes = 7):
	labels_to_remove = set(labels_to_remove)

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


class Dataset_generate_features(data.Dataset):

	def __init__(self, list_IDs, phase, prob, classes, flag_embeddings, CNN_TO_USE = "densenet121"):

		self.list_IDs = utils_data.labels2int(list_IDs)
		self.flag_embeddings = flag_embeddings
		#self.geometric_pipeline = color_transformation.get_pipeline_geometric(prob)
		#self.color_pipeline = color_transformation.get_pipeline_color(prob)
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
		except:
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
		except:
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
		except:
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
		except:
			pass

		if (self.N_CLASSES > 1):
			y = torch.as_tensor(float(y) , dtype=torch.long)
		else:
			y = torch.as_tensor(float(y) , dtype=torch.float32)

		#return input_tensor
		#print(len(encoded_text))
		ID_img = ID_img.split('/')[-1]

		return input_tensor, encoded_text_short, encoded_text_abcd, encoded_text_char, encoded_text_doc, y, ID_img





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


class Dataset_instance(data.Dataset):

	def __init__(self, list_IDs, phase, prob, 
			  classes, possible_reports, 
			  components = [COMPONENTS.images, COMPONENTS.reports],
			  CNN_TO_USE = "densenet121"
			  ):
		self.list_IDs = list_IDs

		self.mode = phase
		self.prob = prob
		self.N_CLASSES = classes
		self.possible_reports = possible_reports
		self.acceptable_reports = self.get_possible_reports()
		

		self.components = components
		
		self.geometric_pipeline = color_transformation.get_pipeline_geometric(prob)
		self.color_pipeline = color_transformation.get_pipeline_color(prob)

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

		if (COMPONENTS.images in self.components):
			X = Image.open(ID_img)
			X = np.asarray(X)

			if (self.mode is PHASE.train):
				prob_pre = np.random.rand(1)[0]

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

		if (self.N_CLASSES > 1):
			y = torch.as_tensor(float(y) , dtype=torch.long)
		else:
			y = torch.as_tensor(float(y) , dtype=torch.float32)

		#print(input_txt)
		#print(textual_concept)
		#return input_tensor
		#print(len(encoded_text))
		ID_img = ID_img.split('/')[-1]

		return input_tensor, encoded_text, y, ID_img



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

		#print(input_txt)
		#print(textual_concept)
		#return input_tensor
		#print(len(encoded_text))
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
			list_possible = 'gpt_4o_mini_fld_abcd'

		elif (possible_reports is REPORTS.char):
			list_possible = 'gpt_4o_mini_fld'

		elif (possible_reports is REPORTS.doc):
			list_possible = 'gpt_4_mini_as_doctor'

		elif (possible_reports is REPORTS.short):
			list_possible = 'short_reports'

		elif (possible_reports is REPORTS.abcd_o4):
			list_possible = 'gpt_4o_fld_abcd'
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (possible_reports is REPORTS.char_o4):
			list_possible = 'gpt_4o_fld'
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

		elif (possible_reports is REPORTS.doc_o4):
			list_possible = 'gpt_4o_fld_as_doctor'
			#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']


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

		#print(input_txt)
		#print(textual_concept)
		#return input_tensor
		#print(len(encoded_text))
		ID_img = ID_img.split('/')[-1]

		return encoded_text, ID_img



class Dataset_SD(data.Dataset):

	def __init__(self, list_IDs, phase, prob, 
				possible_reports, tokenizer, resize = False
				):
		self.list_IDs = list_IDs

		self.mode = phase
		self.prob = prob
		self.possible_reports = possible_reports
		
		self.tokenizer = tokenizer
		self.N_TOKENS = 77
		self.resize = resize
		if (self.resize):
			self.patch_size = 512
		else:
			self.patch_size = 224

		self.new_size = (self.patch_size, self.patch_size)  # Example: 300 pixels wide, 200 pixels tall

		self.geometric_pipeline = color_transformation.get_pipeline_geometric(prob, size=self.patch_size)
		self.color_pipeline = color_transformation.get_pipeline_color(prob)

		self.preprocess = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5, 0.5, 0.5],
									std=[0.5, 0.5, 0.5]),
		])
		
		
		

	def __len__(self):
		return len(self.list_IDs)

	def get_cancer(self, label):
		cancer = None

		if (label < 4):
			cancer = random.choice(["benign", "non-malignant"])
		elif(label == 4):
			cancer = "pre-cancerous"
		else:
			cancer = random.choice(["malignant", "non-benign", "tumor", "cancer"])
		
		return cancer

	def __getitem__(self, index):
		# Select sample
		ID_img = self.list_IDs[index,0]

		prob_pre = np.random.rand(1)[0]

		if (prob_pre > 0.75):
			ID_img_ = ID_img.replace('resized_images','images')
			if (os.path.exists(ID_img_)):
				ID_img = ID_img_

		y = int(self.list_IDs[index,1])

		try:
			ID_txt = self.list_IDs[index,2]
		except:
			ID_txt = None

		label_class = self.list_IDs[index,3]
		label_subclass = self.list_IDs[index,4]
		label_cancer = self.get_cancer(y)

		input_tensor = torch.as_tensor(float(-1) , dtype=torch.long)
		encoded_text = torch.as_tensor(float(-1) , dtype=torch.long)


		X = Image.open(ID_img)
		if (self.resize):
			X = X.resize(self.new_size)


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

		#random
		new_report_type = random.choice(self.possible_reports)
		ID_txt = ID_txt.replace('short_reports',new_report_type)

		input_txt = utils_txt.load_txt(ID_txt)

		prob_pre = np.random.rand(1)[0]
		if (prob_pre >= 0.5):
			elements = utils_SD_text.get_keywords(input_txt)
			prompt = utils_SD_text.fill_template(elements, label_class, label_subclass, label_cancer)

		else:
			subset_lines = utils_SD_text.sample_lines(input_txt)
			len_tokens = len(self.tokenizer.tokenize(subset_lines))

			b = len_tokens <= self.N_TOKENS
			i = 0

			while (b == False):
				subset_lines = utils_SD_text.sample_lines(input_txt)
				len_tokens = len(self.tokenizer.tokenize(subset_lines))

				b = len_tokens <= self.N_TOKENS
				i = i + 1

				if (i > 100):
					b = True
		
			prompt = utils_SD_text.clean_and_join(subset_lines)


		txt_tokens = self.tokenizer(
			prompt,
			truncation=True,
			padding="max_length",
			max_length=self.tokenizer.model_max_length,
			return_tensors="pt",
		)

		input_ids = txt_tokens.input_ids[0]
		attention_mask = txt_tokens.attention_mask[0]
		
		#print(input_txt)
		#print(textual_concept)
		#return input_tensor
		#print(len(encoded_text))
		ID_img = ID_img.split('/')[-1]

		return input_tensor, input_ids, attention_mask, ID_img



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




class Dataset_instance_VQA(data.Dataset):

	def __init__(self, list_IDs, list_symmetry, list_border, list_color, list_dermo, list_questions):


		self.list_IDs = list_IDs
		self.list_questions = list_questions  
		self.list_symmetry = list_symmetry  
		self.list_border = list_border  
		self.list_color = list_color  
		self.list_dermo = list_dermo  

		self.preprocess = transforms.Compose([
		transforms.ToTensor(),
		])

	def __len__(self):
		return len(self.list_IDs)


	def __getitem__(self, index):
		# Select sample
		sample_ID = self.list_IDs[index]

		color = self.list_color[index]
		symm = self.list_symmetry[index]
		dermo = self.list_dermo[index]
		border = self.list_border[index]

		with open(sample_ID, 'rb') as f:
			cls_sample = np.load(f)
		
		cls_question_0 = self.list_questions[0]
		cls_question_1 = self.list_questions[1]
		cls_question_2 = self.list_questions[2]
		cls_question_3 = self.list_questions[3]


		#cls_question_0 = torch.from_numpy(cls_question_0)
		#cls_question_0 = cls_question_0.float()

		#cls_question_1 = torch.from_numpy(cls_question_1)
		#cls_question_1 = cls_question_1.float()

		#cls_question_2 = torch.from_numpy(cls_question_2)
		#cls_question_2 = cls_question_2.float()

		#cls_question_3 = torch.from_numpy(cls_question_3)
		#cls_question_3 = cls_question_3.float()

		cls_sample = torch.tensor(cls_sample, dtype=torch.float)
		cls_question_0 = torch.tensor(cls_question_0, dtype=torch.float)
		cls_question_1 = torch.tensor(cls_question_1, dtype=torch.float)
		cls_question_2 = torch.tensor(cls_question_2, dtype=torch.float)
		cls_question_3 = torch.tensor(cls_question_3, dtype=torch.float)

		dermo = torch.tensor(dermo , dtype=torch.long)
		symm = torch.tensor(symm, dtype=torch.float32)
		border = torch.tensor(border , dtype=torch.float32)
		color = torch.tensor(color.tolist() , dtype=torch.float32)

		#print(dermo, symm, border, color)


		return cls_sample, cls_question_0, cls_question_1, cls_question_2, cls_question_3, symm, border, color, dermo

def perturb_features(features, P, prob=0.5):
	mask = torch.rand_like(features) < prob
	noise = (torch.rand_like(features) * 2 - 1) * P
	return features + noise * mask.float()

class Dataset_instance_VQA_json(data.Dataset):

	def __init__(self, list_IDs, list_questions, list_answers, cls_questions, cls_answers, mode = PHASE.train):


		self.list_IDs = list_IDs
		self.list_questions = list_questions  
		self.list_answers = list_answers  

		self.cls_questions = cls_questions  
		self.cls_answers = cls_answers 
		self.mode = mode


		self.preprocess = transforms.Compose([
		transforms.ToTensor(),
		])

	def __len__(self):
		return len(self.list_IDs)


	def __getitem__(self, index):
		# Select sample
		sample_ID = self.list_IDs[index]
		fname = sample_ID.split('/')[-1].split('.')[0]
		idx_question = self.list_questions[index]
		answers = self.list_answers[index]

		#print(sample_ID, idx_question, answers)

		with open(sample_ID, 'rb') as f:
			cls_sample = np.load(f)
		
		cls_question = self.cls_questions[idx_question]
		if (self.mode is PHASE.train or self.mode is PHASE.valid):
			valid_answers = [a for a in answers if a != 0 and a != 1]
		else:
			valid_answers = answers
		"""
		if (len(answers) == 1):
			label = answers[0]
			cls_answer = self.cls_answers[label]
		else:
			idx = np.random.randint(0,len(answers))
			label = answers[idx]
			cls_answer = self.cls_answers[label]
			#cls_answer = answers[1]
			#label = answers[1]
		"""

		if len(valid_answers) == 1:
			label = valid_answers[0]
			cls_answer = self.cls_answers[label]
		elif len(valid_answers) > 1:
			idx = np.random.randint(0, len(valid_answers))
			label = valid_answers[idx]
			cls_answer = self.cls_answers[label]
		else:
			# Fallback: use original logic or define a default
			# For example, pick randomly from full answers (including 0 or 1)
			idx = np.random.randint(0, len(answers))
			label = answers[idx]
			cls_answer = self.cls_answers[label]

		
		cls_sample = torch.tensor(cls_sample, dtype=torch.float)
		if (self.mode is PHASE.train):
			cls_sample = perturb_features(cls_sample, P=0.1)
		cls_question = torch.tensor(cls_question, dtype=torch.float)
		cls_answer = torch.tensor(cls_answer, dtype=torch.float)
		label = torch.as_tensor(float(label) , dtype=torch.long)

		#print(dermo, symm, border, color)


		return cls_sample, cls_question, cls_answer, label, fname


class Dataset_instance_VQA_json(data.Dataset):

	def __init__(self, list_IDs, list_questions, list_answers, cls_questions, cls_answers, mode = PHASE.train):


		self.list_IDs = list_IDs
		self.list_questions = list_questions  
		self.list_answers = list_answers  

		self.cls_questions = cls_questions  
		self.cls_answers = cls_answers 
		self.mode = mode


		self.preprocess = transforms.Compose([
		transforms.ToTensor(),
		])

	def __len__(self):
		return len(self.list_IDs)


	def __getitem__(self, index):
		# Select sample
		sample_ID = self.list_IDs[index]
		fname = sample_ID.split('/')[-1].split('.')[0]
		idx_question = self.list_questions[index]
		answers = self.list_answers[index]

		#print(sample_ID, idx_question, answers)

		with open(sample_ID, 'rb') as f:
			cls_sample = np.load(f)
		
		cls_question = self.cls_questions[idx_question]
		if (self.mode is PHASE.train or self.mode is PHASE.valid):
			valid_answers = [a for a in answers if a != 0 and a != 1]
		else:
			valid_answers = answers
		"""
		if (len(answers) == 1):
			label = answers[0]
			cls_answer = self.cls_answers[label]
		else:
			idx = np.random.randint(0,len(answers))
			label = answers[idx]
			cls_answer = self.cls_answers[label]
			#cls_answer = answers[1]
			#label = answers[1]
		"""

		if len(valid_answers) == 1:
			label = valid_answers[0]
			cls_answer = self.cls_answers[label]
		elif len(valid_answers) > 1:
			idx = np.random.randint(0, len(valid_answers))
			label = valid_answers[idx]
			cls_answer = self.cls_answers[label]
		else:
			# Fallback: use original logic or define a default
			# For example, pick randomly from full answers (including 0 or 1)
			idx = np.random.randint(0, len(answers))
			label = answers[idx]
			cls_answer = self.cls_answers[label]

		
		cls_sample = torch.tensor(cls_sample, dtype=torch.float)
		if (self.mode is PHASE.train):
			cls_sample = perturb_features(cls_sample, P=0.1)
		cls_question = torch.tensor(cls_question, dtype=torch.float)
		cls_answer = torch.tensor(cls_answer, dtype=torch.float)
		label = torch.as_tensor(float(label) , dtype=torch.long)

		#print(dermo, symm, border, color)


		return cls_sample, cls_question, cls_answer, label, fname


#dataloaders
#data loader at patch-level
class Dataset_eval_txt_generation(data.Dataset):

	def __init__(self, list_IDs, FLDs):

		self.list_IDs = list_IDs

		self.abcd_fold = FLDs[0]
		self.char_fold = FLDs[1]
		self.doc_fold = FLDs[2]
		self.short_fold = FLDs[3]
		self.generated_fold = FLDs[4]


	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID_img = self.list_IDs[index]
		
		fname = ID_img.split('.')[0]

		fname_abcd = self.abcd_fold + fname + '.txt'
		fname_char = self.char_fold + fname + '.txt'
		fname_doc = self.doc_fold + fname + '.txt'
		fname_short = self.short_fold + fname + '.txt'
		fname_generated = self.generated_fold + fname + '.txt'


		input_abcd = utils_txt.load_txt(fname_abcd)
		input_char = utils_txt.load_txt(fname_char)
		input_doc = utils_txt.load_txt(fname_doc)
		input_short = utils_txt.load_txt(fname_short)
		
		input_generated = utils_txt.load_txt(fname_generated)

		return fname, input_abcd, input_char, input_doc, input_short, input_generated

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


if __name__ == "__main__":
	pass