import sys, getopt
import torch
from torch.utils import data
import numpy as np 
import pandas as pd
import torch.nn.functional as F
import torch.utils.data
import os
import argparse
import warnings
import torch.nn as nn
warnings.filterwarnings("ignore")
from enum_multi import ALG, PHASE, NETWORK, MOD, QUESTION, vit_pool
from transformers import BertModel
import vision_transformer
import utils_transformer
from functools import partial
from transformers import AutoTokenizer
from vit_pytorch import ViT
from einops import rearrange, repeat
import timm

class Encoder(torch.nn.Module):
	def __init__(self, CNN_TO_USE):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		super(Encoder, self).__init__()
		
		if 'resnet' in CNN_TO_USE:
			self.netcode = NETWORK.resnet18

		elif 'densenet' in CNN_TO_USE:
			self.netcode = NETWORK.densenet121

		elif 'mobilenet' in CNN_TO_USE:
			self.netcode = NETWORK.mobilenet_v2
		
		pre_trained_network = torch.hub.load('pytorch/vision:v0.10.0', CNN_TO_USE, pretrained=True)
		if (('resnet' in CNN_TO_USE) or ('resnext' in CNN_TO_USE)):
			fc_input_features = pre_trained_network.fc.in_features
		elif (('densenet' in CNN_TO_USE)):
			fc_input_features = pre_trained_network.classifier.in_features
		elif ('mobilenet' in CNN_TO_USE):
			fc_input_features = pre_trained_network.classifier[1].in_features

		if (('resnet' in CNN_TO_USE) or ('resnext' in CNN_TO_USE)):
			self.conv_layers = torch.nn.Sequential(*list(pre_trained_network.children())[:-1])
		elif ('densenet' in CNN_TO_USE):
			self.conv_layers = torch.nn.Sequential(*list(pre_trained_network.children())[:-1])
		elif ('mobilenet' in CNN_TO_USE):
			self.conv_layers = pre_trained_network.features

		self.fc_feat_in = fc_input_features
		
		if (torch.cuda.device_count()>1):
			self.conv_layers = torch.nn.DataParallel(self.conv_layers)

		self.dropout = torch.nn.Dropout(p = 0.2)
		self.activation = torch.nn.ReLU()

		if (self.netcode is NETWORK.densenet121 or self.netcode is NETWORK.mobilenet_v2):

			self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((1,1))
			#self.adaptive_pool = torch.nn.AvgPool2d((7,7))

	def forward(self, x):

		conv_layers_out = self.conv_layers(x)

		if (self.netcode is NETWORK.densenet121 or self.netcode is NETWORK.mobilenet_v2):
			conv_layers_out = self.activation(conv_layers_out)
			conv_layers_out = self.adaptive_pool(conv_layers_out)

		features = conv_layers_out.view(-1, self.fc_feat_in)
		
		return features


class ViT_Encoder(nn.Module):
	def __init__(self, CNN_TO_USE):
		super().__init__()

		self.netcode = NETWORK[CNN_TO_USE]

		if (self.netcode is NETWORK.HIPT):
			self.vit = timm.create_model("vit_small_patch16_224.augreg_in1k", pretrained=True)

		elif (self.netcode is NETWORK.ViT):
			self.vit = timm.create_model("timm/vit_tiny_patch16_224", pretrained=True)
			
		self.vit.reset_classifier(0)

		self.hidden_dim = self.vit.num_features  # e.g., 192 for vit_tiny

		self.pooler = torch.nn.Sequential(
			torch.nn.Linear(self.hidden_dim, self.hidden_dim),
			torch.nn.Tanh()
		)

	def forward(self, x):

		x = self.vit.forward_features(x)  # shape: (B, num_tokens, hidden_dim)
        
		# Extract CLS token (assumed at index 0)
		cls_token = x[:, 0]  # shape: (B, hidden_dim)

		# Apply pooler
		pooled = self.pooler(cls_token)  # shape: (B, hidden_dim)

		return pooled

class MultimodalArchitecture(torch.nn.Module):
	def __init__(self, device, CNN_TO_USE = 'densenet121', 
			  out_dim = 5, in_dim = 512, intermediate_dim=128, 
			  TEMPERATURE = 0.07, pretrained_path = None, patch_size = None):
		
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		super(MultimodalArchitecture, self).__init__()

		
		self.netcode = NETWORK[CNN_TO_USE]
		
		#image encoder
		#self.conv_layers = CNN_Encoder(CNN_TO_USE)
		self.hidden_space_len = intermediate_dim
		
		#IMAGE SPECIFIC
		if (self.netcode is NETWORK.HIPT or self.netcode is NETWORK.ViT):
			self.img_encoder = ViT_Encoder(CNN_TO_USE)
		
		else:
			self.img_encoder = Encoder(CNN_TO_USE)

		self.fc_feat_in = in_dim
		
		self.N_CLASSES = out_dim

		self.intermediate_layer_img = torch.nn.Linear(self.fc_feat_in, self.hidden_space_len)
		self.flag_intermediate = True

		self.activation_img = torch.nn.ReLU()
		self.dropout_img = torch.nn.Dropout(p=0.2)

		#Txt specific
		self.TEMPERATURE = TEMPERATURE
		self.device = device
		self.dropout_img = torch.nn.Dropout(p=0.1)
		self.clinical_bert_token_size = 768
		self.LayerNorm_txt = torch.nn.LayerNorm(self.clinical_bert_token_size, eps=1e-5)
		self.dropout_txt = torch.nn.Dropout(0.1)
		self.activation_txt = torch.nn.Tanh()
		try:
			bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
			
			#self.pooler = None
			self.txt_encoder = BertModel.from_pretrained(bert_chosen, 
														output_attentions=True, 
														output_hidden_states=True,
														attn_implementation="eager")
		except:
			try:
				bert_chosen = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
				self.txt_encoder = BertModel.from_pretrained(bert_chosen, 
															local_files_only = True, 
															force_download = True,
															output_attentions=True, 
															output_hidden_states=True,
															attn_implementation="eager")
			except:
				bert_chosen = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
				self.txt_encoder = BertModel.from_pretrained(bert_chosen, 
															output_attentions=True, 
															output_hidden_states=True,
															attn_implementation="eager")
				
		#from CLS_TOKEN to common embedding (like for images)
		self.embedding_output_txt = torch.nn.Linear(in_features=self.clinical_bert_token_size, out_features=self.hidden_space_len)		

		#COMMON LAYERS MULTIMODAL
		#output of first common layer
		self.activation = torch.nn.ReLU()
		self.intermediate_embedding = torch.nn.Linear(in_features=self.hidden_space_len, out_features=self.hidden_space_len)
		self.classifier = torch.nn.Linear(in_features=self.hidden_space_len, out_features=self.N_CLASSES)

	def forward(self, input_img, input_txt, flag_embedding_pubmed = False):
		img_prob = None
		intermediate_embedding_img = None
		txt_prob = None
		intermediate_embedding_txt = None
		total_inst_loss = None

		#phase = PHASE[phase]

		####process images (features pre-processed)
		if input_img is not None:
			
			conv_layers_out = self.img_encoder(input_img)

			x = self.intermediate_layer_img(conv_layers_out)
			#x = self.activation_img(x)
			#x = self.dropout(x)

			output_img = self.intermediate_embedding(x)
			
			intermediate_embedding_img = output_img

			#output_img = self.activation(output_img)
			#print(output_img.shape)
			img_prob = self.classifier(output_img)#.view(-1)


		if input_txt is not None and flag_embedding_pubmed == False:
			
			input_ids_txt = input_txt["input_ids"].squeeze(1)# (batch_size, seq_length)
			attention_mask_txt = input_txt["attention_mask"].squeeze(1)  # (batch_size, seq_length)

			outputs_txt = self.txt_encoder(input_ids=input_ids_txt, attention_mask=attention_mask_txt)

			pooled_output = outputs_txt.pooler_output  # (batch_size, hidden_dim)
			intermediate_txt = self.embedding_output_txt(pooled_output)

			output_txt = intermediate_txt

			output_txt = self.intermediate_embedding(intermediate_txt)
			output_txt = output_txt#.view(-1)

			intermediate_embedding_txt = output_txt

			#output_txt = self.activation(output_txt)

			txt_prob = self.classifier(output_txt)#.view(-1)

		elif input_txt is not None and flag_embedding_pubmed == True:
			
			pooled_output = input_txt  # (batch_size, hidden_dim)

			intermediate_txt = self.embedding_output_txt(pooled_output)

			output_txt = intermediate_txt

			output_txt = self.intermediate_embedding(intermediate_txt)
			output_txt = output_txt#.view(-1)

			intermediate_embedding_txt = output_txt

			txt_prob = self.classifier(output_txt)#.view(-1)

		return img_prob, intermediate_embedding_img, txt_prob, intermediate_embedding_txt



class FeatureToTextDecoder(nn.Module):
	def __init__(self, feature_dim=128, hidden_dim=512, num_layers=4, nhead=8, dropout=0.1, max_len=64):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.max_len = max_len

		tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

		# Special tokens
		PAD_ID = tokenizer.pad_token_id
		BOS_ID = tokenizer.cls_token_id  # or tokenizer.bos_token_id if defined
		EOS_ID = tokenizer.sep_token_id  # or tokenizer.eos_token_id if defined

		self.vocab_size = tokenizer.vocab_size

		self.feature_proj = nn.Linear(feature_dim, hidden_dim)

		self.embedding = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=PAD_ID)
		self.pos_encoding = nn.Parameter(torch.randn(max_len, hidden_dim))

		decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True)
		self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

		self.output_proj = nn.Linear(hidden_dim, self.vocab_size)

	def forward(self, features, tgt_input_ids):
		"""
		features: (batch_size, 128)
		tgt_input_ids: (batch_size, seq_len)
		"""
		seq_len = tgt_input_ids.shape[1]
		memory = self.feature_proj(features).unsqueeze(1)  # (batch, 1, hidden)
		tgt_embeddings = self.embedding(tgt_input_ids) + self.pos_encoding[:seq_len]
		tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt_input_ids.device)
		
		output = self.decoder(tgt=tgt_embeddings, memory=memory, tgt_mask=tgt_mask)
		logits = self.output_proj(output)  # (batch, seq_len, vocab)

		return logits




class MultiCategoryVQAModel(torch.nn.Module):
	def __init__(self, input_dim = 128, intermediate_dim = 128, N_CLASSES = 37):
		super().__init__()
		# Image encoder (e.g., ResNet)
		self.input_dim = input_dim
		self.intermediate_dim = intermediate_dim
		self.combined_dim = self.intermediate_dim + self.intermediate_dim   # ResNet18 + BERT pooler
		
		self.activation_relu = torch.nn.ReLU()
		self.activation_tanh = torch.nn.Tanh()

		self.encoder_img = torch.nn.Sequential(
				torch.nn.Linear(self.input_dim, self.intermediate_dim),
				self.activation_relu,
				torch.nn.Linear(self.intermediate_dim, self.intermediate_dim),
				#self.activation_relu,
				#torch.nn.Linear(self.intermediate_dim, self.intermediate_dim)
				)
		
		self.encoder_txt = torch.nn.Sequential(
				torch.nn.Linear(self.input_dim, self.intermediate_dim),
				self.activation_relu,
				torch.nn.Linear(self.intermediate_dim, self.intermediate_dim),
				#self.activation_relu,
				#torch.nn.Linear(self.intermediate_dim, self.intermediate_dim)
				)

		self.intermediate_layer = torch.nn.Sequential(
				torch.nn.Linear(self.combined_dim, self.intermediate_dim),
				self.activation_relu,
				torch.nn.Linear(self.intermediate_dim, self.intermediate_dim),
				#self.activation_relu,
				#torch.nn.Linear(self.intermediate_dim, self.intermediate_dim)
			)

		self.embedding_layer = torch.nn.Sequential(
				torch.nn.Linear(self.intermediate_dim, self.intermediate_dim),
				self.activation_relu,
				torch.nn.Linear(self.intermediate_dim, self.intermediate_dim),
				#self.activation_relu,
				#torch.nn.Linear(self.intermediate_dim, self.intermediate_dim)
			)
		
		self.classifier = torch.nn.Sequential(
				torch.nn.Linear(self.intermediate_dim, self.intermediate_dim),
				self.activation_relu,
				torch.nn.Linear(self.intermediate_dim, N_CLASSES),
			)
		#self.intermediate_1 = torch.nn.Linear(self.combined_dim, self.intermediate_dim)
		#self.intermediate_2 = torch.nn.Linear(self.intermediate_dim, self.intermediate_dim)

		# Four classifier heads, one per question
		#self.classifier_symmetry = torch.nn.Linear(self.intermediate_dim, 1)
		#self.classifier_border = torch.nn.Linear(self.intermediate_dim, 1)
		#self.classifier_color = torch.nn.Linear(self.intermediate_dim, 8)
		#self.classifier_structures = torch.nn.Linear(self.intermediate_dim, 7)

	def forward(self, img_feat, question_feat):
		
		x_img = self.encoder_img(img_feat)
		x_txt = self.encoder_txt(question_feat)

		combined = torch.cat([x_img, x_txt], dim=1)

		x = self.intermediate_layer(combined)

		embedding = self.embedding_layer(x)

		y = self.classifier(x)

		return embedding, y

if __name__ == "__main__":
	pass