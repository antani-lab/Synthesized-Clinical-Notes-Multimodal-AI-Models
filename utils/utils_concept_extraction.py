import sys, getopt
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
import utils_data
import utils_txt
import random
from transformers import AutoTokenizer
import math
import re
from collections import defaultdict
import json
import utils_concept_extraction
from transformers import BertModel
import numpy as np
import torch
from tqdm import tqdm

# ====================
# CONFIGURATION
# ====================

GLOBAL_CONCEPTS = {
	"asymmetry": ["symmetry", "asymmetry", "symmetric", "asymmetric"],
	"border": ["irregular", "well-defined", "regular", "indistinct", "distinct", "sharp", "ill-defined"],
	"color": ["white", "pink", "red", "yellow", "brown", "dark", "black", "blue", "grey", "multicolored"],
	"structure": ["flat", "nodule", "plaque", "ulcer", "cyst", "pigment", "dot", "globule", "structureless",
				  "macule", "papule", "follicular", "plug"],
	"cancer": ["benign", "non-malignant","pre-cancerous", "malignant", "non-benign", "tumor", "cancer"],
	"lesion": ["seborrheic", "nevus","vascular", "dermatofibroma", "actinic", "basal", "melanoma", "squamous"],
	"subclasses": ["lentigo", "solar lentigo", "seborrheic", "lichenoid", "lichen", "planus", "compound", "blue", "halo", "dysplastic", "pyogenic", "angioma", "bowen", "kaposi", "actinic", "maligna", "metastatic"]
	#"subclasses": ["lentigo", "lichen", "planus", "compound", "halo", "dysplastic", "pyogenic", "angioma", "bowen", "kaposi", "metastatic"]
}

CONFLICTS = {
	'symmetry': ['asymmetry', 'asymmetric'],
	'asymmetry': ['symmetry', 'symmetric'],
	'symmetric': ['asymmetric', 'asymmetry'],
	'asymmetric': ['symmetric', 'symmetry'],
	'distinct': ['indistinct'],
	'indistinct': ['distinct'],
	'regular': ['irregular'],
	'irregular': ['regular'],
	'benign': ['non-benign','tumor','cancer','pre-cancerous'],
	'non-malignant': ['non-benign','tumor','cancer','pre-cancerous'],
	'non-benign': ['benign','non-malignant','pre-cancerous'],
	'tumor': ['benign','non-malignant','pre-cancerous'],
	'cancer': ['benign','non-malignant','pre-cancerous'],
	'pre-cancerous':["benign", "non-malignant", "malignant", "non-benign", "tumor", "cancer"]
}

SPELL_NORMALIZATION = {
	'asimmetry': 'asymmetry',
	'simmetry': 'symmetry',
	'symmetrical': 'symmetric',
	'asymmetrical': 'asymmetric',
	'yellowish': 'yellow',
	'bluish': 'blue',
	'brownish': 'brown',
	'reddish': 'red',
	'darker': 'dark',
	'dots': 'dot',
	'lighter': 'light',
	'plugs':'plug'
}

flattened_concepts = [concept for values in GLOBAL_CONCEPTS.values() for concept in values]
#flattened_concepts = list(set(flattened_concepts))


concept_to_index = {}

for i, c in enumerate(flattened_concepts):
	if c not in concept_to_index:  # keep only first occurrence
		concept_to_index[c] = i
	
def encode_flat(sample, concept_to_index):
	
	vector = np.zeros(len(concept_to_index))

	for s in sample:
		idx = concept_to_index[s]
		vector[idx] = 1
		
	return vector

def flatten_and_remove(input_list, value_to_remove):
	flattened = []
	for element in input_list:
		if isinstance(element, list):
			for item in element:
				if item != value_to_remove:
					flattened.append(item)
	return flattened

# ====================
# NORMALIZATION
# ====================
def spell_normalize(word):
	return SPELL_NORMALIZATION.get(word, word)

def normalize_text(text):
	words = re.findall(r'\b[\w-]+\b', text.lower())
	
	return ' '.join(spell_normalize(w) for w in words)
	

# ====================
# UNION-FIND FOR CONFLICTS
# ====================
PARENT = {}

def find(x):
	if PARENT[x] != x:
		PARENT[x] = find(PARENT[x])
	return PARENT[x]

def union(x, y):
	PARENT.setdefault(x, x)
	PARENT.setdefault(y, y)
	PARENT[find(x)] = find(y)

def build_conflict_groups(conflict_map):
	PARENT.clear()
	for key, values in conflict_map.items():
		k_norm = spell_normalize(key.lower())
		for v in values:
			v_norm = spell_normalize(v.lower())
			union(k_norm, v_norm)
	return {k: find(k) for k in PARENT}

# ====================
# CONCEPT MATCHING
# ====================

row_suffixes = {
	1: " border",     # Add to second list (index 1)
	3: " structure",
	#5: " lesion"
}

def add_suffix(list_concepts):
	# Modify the elements
	for row_idx, suffix in row_suffixes.items():
		if row_idx < len(list_concepts):  # Safety check
			list_concepts[row_idx] = [elem + suffix for elem in list_concepts[row_idx]]

	return list_concepts

def find_present_concepts(text, concept_dict, conflict_map=None):
	norm_text = normalize_text(text)
	group_map = build_conflict_groups(conflict_map or {})

	result = []
	used_groups = set()

	for category, concept_list in concept_dict.items():
		found = []

		for concept in concept_list:
			concept_norm = spell_normalize(concept.lower())
			pattern = r'\b' + re.escape(concept_norm) + r'\b'

			if not re.search(pattern, norm_text):
				continue

			group_id = group_map.get(concept_norm)
			if group_id and group_id in used_groups:
				continue

			if group_id:
				used_groups.add(group_id)

			found.append(concept)

		result.append(found if found else -1)

	#result = add_suffix(result)
	return result

def remove_random_number_up_to_x(lst, X):
	max_removable = min(X, len(lst))
	how_many = random.randint(0, max_removable)

	#print(f"Removing {how_many} elements (max allowed: {X})")

	indices = sorted(random.sample(range(len(lst)), how_many), reverse=True)
	for i in indices:
		lst.pop(i)
	return lst

def get_textual_concepts(text):

	textual_concept = None

	if (len(text) > 1):

		keywords = []

		for t in text:

			if (t != -1):

				for i in t:

					keywords.append(i)

		keywords = list(set(keywords))
		textual_concept = ', '.join(keywords)

	else:

		textual_concept = text[0]

	return textual_concept


def get_keywords_embeddings_PubMED(txt_encoder, device):

	features = np.empty((len(flattened_concepts), 768))		
	
	bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
	tokenizer = AutoTokenizer.from_pretrained(bert_chosen)  

	"""
	for name, param in model.base_encoder.named_parameters():
		param.requires_grad = False
	"""
	txt_encoder.to(device)
	txt_encoder.eval()

	for param in txt_encoder.parameters():
		param.requires_grad = False

	with torch.autocast(device_type='cuda', dtype=torch.float16):

		for i in tqdm(range(len(flattened_concepts))):
			
			input_txt = flattened_concepts[i]


			encoded_text = tokenizer(
				input_txt,
				add_special_tokens=True,
				return_token_type_ids=True,
				return_attention_mask=True,
				padding="max_length",  # Pads all sequences to the max_length in the batch
				truncation=True,  # Ensures no sequence exceeds max length
				max_length=512,  # Adjust based on model constraints
				return_tensors="pt"  # Directly returns PyTorch tensors
			).to(device)


			with torch.no_grad():
				# forward + backward + optimize
				
				input_ids_txt = encoded_text["input_ids"].squeeze(1)# (batch_size, seq_length)
				attention_mask_txt = encoded_text["attention_mask"].squeeze(1)  # (batch_size, seq_length)

				# 1. Pass the input directly to the model (BERT will internally compute embeddings)
				outputs_txt = txt_encoder(input_ids=input_ids_txt, attention_mask=attention_mask_txt)

				# 2. The encoder output is a tuple: (last_hidden_state, pooler_output, other outputs)
				#last_hidden_state = outputs_txt.last_hidden_state  # (batch_size, seq_length, hidden_dim)
				pooled_output = outputs_txt.pooler_output  # (batch_size, hidden_dim)
				
				pooled_output_np = pooled_output.cpu().data.numpy()

				del input_ids_txt, attention_mask_txt, outputs_txt

				features[i] = pooled_output_np
	
	features = torch.tensor(features, dtype=torch.float32)
	return features

if __name__ == "__main__":
	pass