import numpy as np 
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
#from pytorch_pretrained_bert.modeling import BertModel
#from torch.utils import data
from enum_multi import PHASE, MOD, REPORTS
from tqdm import tqdm
import json
from collections import defaultdict

import random

def get_specific_dataset(MAIN_FOLDER, DATASET, flag, CSV_FOLDER = None):

	list_images = []
	list_labels = []
	list_reports = []
	
	folder_path = MAIN_FOLDER+DATASET+'/resized_images/'
	#folder_reports = MAIN_FOLDER+DATASET+'/clinical_notes/gpt_4o_mini_fld/'
	folder_reports = MAIN_FOLDER+DATASET+'/clinical_notes/short_reports/'
	
	if (CSV_FOLDER is not None):
		folder_csv = CSV_FOLDER+DATASET+'/'
	else:
		folder_csv = MAIN_FOLDER+DATASET+'/resized_images/'

	flag_add = False
	
	if (flag is PHASE.train):

		try:
			filename = folder_csv + 'labels_train.csv'
			csv_file = pd.read_csv(filename, sep = ',', header = None).values
			flag_add = True
		except:
			pass


	elif (flag is PHASE.valid):

		try:
			filename = folder_csv + 'labels_valid.csv'
			csv_file = pd.read_csv(filename, sep = ',', header = None).values
			flag_add = True
		except:
			pass
	
	elif (flag is PHASE.test):

		try:
			filename = folder_csv + 'labels_test.csv'
			csv_file = pd.read_csv(filename, sep = ',', header = None).values
			flag_add = True
		except:
			filename = folder_csv + 'filtered_labels_test.csv'
			csv_file = pd.read_csv(filename, sep = ',', header = None).values
			flag_add = True

	elif (flag is PHASE.all):

		filename = folder_csv + 'labels.csv'
		csv_file = pd.read_csv(filename, sep = ',', header = None).values
		flag_add = True


	if (flag_add):

		for i in tqdm(range(len(csv_file))):
			
			fname = folder_path + csv_file[i,0]
			list_images.append(fname)
			list_labels.append(int(csv_file[i,1]))
			#list_labels.append(csv_file[i,1].astype(np.uint8))
			list_reports.append(folder_reports + csv_file[i,0].split('.')[0] + '.txt')
				
	list_labels = [int(x) for x in list_labels]
	list_images = np.array(list_images)
	list_labels = np.array(list_labels)
	list_reports = np.array(list_reports)

	data_to_use = np.vstack((list_images, list_labels, list_reports), dtype = 'object').T
	#data_to_use = np.array([[list_images[i], list_labels[i]] for i in range(len(list_labels[i]))])

	return data_to_use


def get_specific_dataset_features(MAIN_FOLDER, DATASET, flag, CSV_FOLDER = None, overwrite_flag = False, feat_flds = None):

	list_images = []
	list_labels = []
	list_reports = []
	
	folder_path = MAIN_FOLDER+DATASET+'/resized_images/'
	#folder_reports = MAIN_FOLDER+DATASET+'/clinical_notes/gpt_4o_mini_fld/'
	folder_reports = MAIN_FOLDER+DATASET+'/clinical_notes/short_reports/'
	
	if (CSV_FOLDER is not None):
		folder_csv = CSV_FOLDER+DATASET+'/'
	else:
		folder_csv = MAIN_FOLDER+DATASET+'/resized_images/'

	flag_add = False
	
	if (flag is PHASE.train):

		try:
			filename = folder_csv + 'labels_train.csv'
			csv_file = pd.read_csv(filename, sep = ',', header = None).values
			flag_add = True
		except:
			pass


	elif (flag is PHASE.valid):

		try:
			filename = folder_csv + 'labels_valid.csv'
			csv_file = pd.read_csv(filename, sep = ',', header = None).values
			flag_add = True
		except:
			pass
	
	elif (flag is PHASE.test):

		try:
			filename = folder_csv + 'labels_test.csv'
			csv_file = pd.read_csv(filename, sep = ',', header = None).values
			flag_add = True
		except:
			filename = folder_csv + 'filtered_labels_test.csv'
			csv_file = pd.read_csv(filename, sep = ',', header = None).values
			flag_add = True

	elif (flag is PHASE.all):

		filename = folder_csv + 'labels.csv'
		csv_file = pd.read_csv(filename, sep = ',', header = None).values
		flag_add = True


	if (flag_add):

		for i in tqdm(range(len(csv_file))):
			
			if (overwrite_flag == False):
				fname_sample = csv_file[i,0].split('.')[0]+'.npy'

				flag_b = False
				fold_idx = 0

				while (fold_idx < len(feat_flds) and flag_b == False):

					x = feat_flds[fold_idx] + fname_sample

					if (os.path.exists(x) == False):
						flag_b = True
						fname = folder_path + csv_file[i,0]
						list_images.append(fname)
						list_labels.append(int(csv_file[i,1]))
						#list_labels.append(csv_file[i,1].astype(np.uint8))
						list_reports.append(folder_reports + csv_file[i,0].split('.')[0] + '.txt')
					else:
						fold_idx = fold_idx + 1

			else:

				fname = folder_path + csv_file[i,0]
				list_images.append(fname)
				list_labels.append(int(csv_file[i,1]))
				#list_labels.append(csv_file[i,1].astype(np.uint8))
				list_reports.append(folder_reports + csv_file[i,0].split('.')[0] + '.txt')
				
	list_labels = [int(x) for x in list_labels]
	list_images = np.array(list_images)
	list_labels = np.array(list_labels)
	list_reports = np.array(list_reports)

	data_to_use = np.vstack((list_images, list_labels, list_reports), dtype = 'object').T
	#data_to_use = np.array([[list_images[i], list_labels[i]] for i in range(len(list_labels[i]))])

	return data_to_use




def get_instances_paths_from_bags(MAIN_FOLDER, list_datasets, flag, CSV_FOLDER = None):

	list_images = []
	list_labels = []
	list_reports = []

	for l in list_datasets:

		flag_add = False

		folder_path = MAIN_FOLDER+l+'/resized_images/'
		#folder_reports = MAIN_FOLDER+l+'/clinical_notes/gpt_4o_mini_fld/'
		folder_reports = MAIN_FOLDER+l+'/clinical_notes/short_reports/'

		if (CSV_FOLDER is not None):
			folder_csv = CSV_FOLDER+l+'/'
		else:
			folder_csv = MAIN_FOLDER+l+'/resized_images/'

		if (flag is PHASE.train):

			try:
				filename = folder_csv + 'labels_train.csv'
				csv_file = pd.read_csv(filename, sep = ',', header = None).values
				flag_add = True
			except Exception as e:
				print(e)
				pass

		elif (flag is PHASE.valid):

			try:
				filename = folder_csv + 'labels_valid.csv'
				csv_file = pd.read_csv(filename, sep = ',', header = None).values
				flag_add = True
			except Exception as e:
				print(e)
				pass
		
		elif (flag is PHASE.test):

			try:
				filename = folder_csv + 'labels_test.csv'
				csv_file = pd.read_csv(filename, sep = ',', header = None).values
				flag_add = True
			except Exception as e:
				print(e)
				pass

		elif (flag is PHASE.all):

			filename = folder_csv + 'labels.csv'
			csv_file = pd.read_csv(filename, sep = ',', header = None).values
			flag_add = True

		if (flag_add):

			for i in tqdm(range(len(csv_file))):
				
				fname = folder_path + csv_file[i,0]
				fname_report = folder_reports + csv_file[i,0].split('.')[0] + '.txt'

				if (os.path.exists(fname_report)):
					list_images.append(fname)
					list_labels.append(int(csv_file[i,1]))
					list_reports.append(fname_report)

	list_labels = [int(x) for x in list_labels]		
	list_images = np.array(list_images)
	list_labels = np.array(list_labels)
	list_reports = np.array(list_reports)

	data_to_use = np.vstack((list_images, list_labels, list_reports), dtype = 'object').T
	#data_to_use = np.array([[list_images[i], list_labels[i]] for i in range(len(list_labels[i]))])

	return data_to_use

def get_class_subclass(fname, dataset):

	i = 0
	b = False

	label_class = None
	label_subclass = None
	fname = fname.split('.')[0]

	while (i<len(dataset) and b == False):

		if (fname in dataset[i,0]):
			label_class = dataset[i,1]
			label_subclass = dataset[i,2]
			b = True
		else:
			i = i + 1
	
	return label_class, label_subclass

def get_instances_paths_SD(MAIN_FOLDER, list_datasets, flag, CSV_FOLDER = None):

	list_images = []
	list_labels = []
	list_reports = []
	list_classes = []
	list_subclasses = []

	for l in list_datasets:

		flag_add = False

		folder_path = MAIN_FOLDER+l+'/resized_images/'
		#folder_reports = MAIN_FOLDER+l+'/clinical_notes/gpt_4o_mini_fld/'
		folder_reports = MAIN_FOLDER+l+'/clinical_notes/short_reports/'

		if (CSV_FOLDER is not None):
			folder_csv = CSV_FOLDER+l+'/'
		else:
			folder_csv = MAIN_FOLDER+l+'/resized_images/'

		csv_classes_filename = folder_csv + "classes_subclasses.csv"
		csv_file_classes_subclasses = pd.read_csv(csv_classes_filename, sep = ',', header = None).values

		if (flag is PHASE.train):

			try:
				filename = folder_csv + 'labels_train.csv'
				csv_file = pd.read_csv(filename, sep = ',', header = None).values
				flag_add = True
			except Exception as e:
				print(e)
				pass

		elif (flag is PHASE.valid):

			try:
				filename = folder_csv + 'labels_valid.csv'
				csv_file = pd.read_csv(filename, sep = ',', header = None).values
				flag_add = True
			except Exception as e:
				print(e)
				pass
		
		elif (flag is PHASE.test):

			try:
				filename = folder_csv + 'labels_test.csv'
				csv_file = pd.read_csv(filename, sep = ',', header = None).values
				flag_add = True
			except Exception as e:
				print(e)
				pass

		elif (flag is PHASE.all):

			filename = folder_csv + 'labels.csv'
			csv_file = pd.read_csv(filename, sep = ',', header = None).values
			flag_add = True

		if (flag_add):

			for i in tqdm(range(len(csv_file))):
				
				fname = folder_path + csv_file[i,0]
				fname_report = folder_reports + csv_file[i,0].split('.')[0] + '.txt'
				label_class, label_subclass = get_class_subclass(csv_file[i,0], csv_file_classes_subclasses)

				if (os.path.exists(fname_report)):
					list_images.append(fname)
					list_labels.append(int(csv_file[i,1]))
					list_reports.append(fname_report)
					list_classes.append(label_class)
					list_subclasses.append(label_subclass)

	list_labels = [int(x) for x in list_labels]		
	list_images = np.array(list_images)
	list_labels = np.array(list_labels)
	list_reports = np.array(list_reports)
	list_classes = np.array(list_classes)
	list_subclasses = np.array(list_subclasses)


	data_to_use = np.vstack((list_images, list_labels, list_reports, list_classes, list_subclasses), dtype = 'object').T
	#data_to_use = np.array([[list_images[i], list_labels[i]] for i in range(len(list_labels[i]))])

	return data_to_use



def labels2int(array):
    # Step 1: Make a copy of the input array
    new_array = np.copy(array).astype(object)
    
    # Step 2: Iterate through each row and convert the second column to int
    for i in range(len(new_array)):
        new_array[i, 1] = int(new_array[i, 1])  # Use int() to convert string to integer

    return new_array


def filter_labels(arr, labels_to_remove, n_classes = 7):
    labels_to_remove = set(labels_to_remove)

    filenames = []
    labels = []
    centers = []

    flag_center = False
    for r in arr:
        label = int(r[1])
        if label not in labels_to_remove:
            filenames.append(r[0])
            labels.append(label)
            try:
                centers.append(r[2])
                flag_center = True
            except IndexError:
                pass

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



def save_prediction(checkpoint_path, N_CLASSES, phase, epoch, arrays_imgs, arrays_txts, DATASET, MODALITY):
	
	if (phase=='test'):
		storing_dir = checkpoint_path + '/' + phase + '/'
	else:
		storing_dir = checkpoint_path + '/' + phase + '/epoch_' + str(epoch) + '/metrics/'

	os.makedirs(storing_dir, exist_ok = True)
	if (phase=='test'):
		filename_val_imgs = storing_dir+DATASET+'_predictions_img.csv'
		filename_val_txts = storing_dir+DATASET+'_predictions_txt.csv'
	else:
		filename_val_imgs = storing_dir+'_predictions_img.csv'
		filename_val_txts = storing_dir+'_predictions_txt.csv'

	
	if (MODALITY is MOD.img):
		# Creating column names dynamically
		columns = ['filenames'] + [f'class_{i}' for i in range(N_CLASSES)]

		# Constructing the data dictionary
		File = {'filenames': arrays_imgs[:, 0]}
		for i in range(1, N_CLASSES + 1):
			File[f'class_{i - 1}'] = arrays_imgs[:, i]

		# Creating the DataFrame
		df = pd.DataFrame(File, columns=columns)

		fmt = ['%s'] + ['%.4f'] * (arrays_imgs.shape[1] - 1)

		np.savetxt(filename_val_imgs, df.values, fmt='%s',delimiter=',')

	elif (MODALITY is MOD.txt):
		# Creating column names dynamically
		columns = ['filenames'] + [f'class_{i}' for i in range(N_CLASSES)]

		# Constructing the data dictionary
		File = {'filenames': arrays_txts[:, 0]}
		for i in range(1, N_CLASSES + 1):
			File[f'class_{i - 1}'] = arrays_txts[:, i]

		fmt = ['%s'] + ['%.4f'] * (arrays_txts.shape[1] - 1)
		# Creating the DataFrame
		df = pd.DataFrame(File, columns=columns)

		np.savetxt(filename_val_txts, df.values, fmt='%s',delimiter=',')

	if (MODALITY is MOD.multimodal):
		# Creating column names dynamically
		columns = ['filenames'] + [f'class_{i}' for i in range(N_CLASSES)]

		# Constructing the data dictionary
		File = {'filenames': arrays_imgs[:, 0]}
		for i in range(1, N_CLASSES + 1):
			File[f'class_{i - 1}'] = arrays_imgs[:, i]

		# Creating the DataFrame
		df = pd.DataFrame(File, columns=columns)

		fmt = ['%s'] + ['%.4f'] * (arrays_imgs.shape[1] - 1)

		np.savetxt(filename_val_imgs, df.values, fmt='%s',delimiter=',')


		# Creating column names dynamically
		columns = ['filenames'] + [f'class_{i}' for i in range(N_CLASSES)]

		# Constructing the data dictionary
		File = {'filenames': arrays_txts[:, 0]}
		for i in range(1, N_CLASSES + 1):
			File[f'class_{i - 1}'] = arrays_txts[:, i]

		# Creating the DataFrame
		df = pd.DataFrame(File, columns=columns)

		fmt = ['%s'] + ['%.4f'] * (arrays_txts.shape[1] - 1)

		np.savetxt(filename_val_txts, df.values, fmt='%s',delimiter=',')

def save_loss_function(checkpoint_path, phase, epoch, value):

	storing_dir = checkpoint_path + '/' + phase + '/epoch_' + str(epoch) + '/'
	os.makedirs(storing_dir, exist_ok = True)

	filename_val = storing_dir+'loss_function.csv'
	array_val = [value]
	File = {'val':array_val}
	df = pd.DataFrame(File,columns=['val'])
	np.savetxt(filename_val, df.values, fmt='%s',delimiter=',')

def save_hyperparameters(checkpoint_path, N_CLASSES, EMBEDDING_bool, lr):

	filename_hyperparameters = checkpoint_path+'hyperparameters.csv'
	array_n_classes = [str(N_CLASSES)]
	array_lr = [str(lr)]
	array_embedding = [EMBEDDING_bool]
	File = {'n_classes':array_n_classes, 'lr':array_lr, 'embedding':array_embedding}

def get_instances_txt_generation(MAIN_FOLDER, list_datasets, flag, CNN_TO_USE, MODALITY, N_EXP, CSV_FOLDER = None, FILTER_LABELS = [1,3], REPORTS_TRAINING = 'abcd', flag_KEYWORDS = False, REPORT_AUGMENTATION = False):

	list_images = []
	list_features = []
	list_reports = []

	for l in list_datasets:
		
		FEAT_FLD = MAIN_FOLDER+l+'/features_'+CNN_TO_USE+'/'+MODALITY+'/'

		FEAT_FLD = FEAT_FLD + REPORTS_TRAINING + '/'

		if (flag_KEYWORDS):
			FEAT_FLD = FEAT_FLD + 'keywords/'
		else:
			FEAT_FLD = FEAT_FLD + 'no_keywords/'

		if (REPORT_AUGMENTATION):
			FEAT_FLD = FEAT_FLD + 'report_augmentation/'
		else:
			FEAT_FLD = FEAT_FLD + 'no_report_augmentation/'
 
		FEAT_FLD = FEAT_FLD+'/N_EXP_'+N_EXP+'/images/'

		flag_add = False

		folder_path = MAIN_FOLDER+l+'/resized_images/'

		folder_reports = MAIN_FOLDER+l+'/clinical_notes/short_reports/'
		
		if (CSV_FOLDER is not None):
			folder_csv = CSV_FOLDER+l+'/'
		else:
			folder_csv = MAIN_FOLDER+l+'/resized_images/'

		if (flag is PHASE.train):

			try:
				filename = folder_csv + 'labels_train.csv'
				csv_file = pd.read_csv(filename, sep = ',', header = None).values
				flag_add = True
			except Exception as e:
				print(e)
				pass

		elif (flag is PHASE.valid):

			try:
				filename = folder_csv + 'labels_valid.csv'
				csv_file = pd.read_csv(filename, sep = ',', header = None).values
				flag_add = True
			except Exception as e:
				print(e)
				pass
		
		elif (flag is PHASE.all):

			filename = folder_csv + 'labels.csv'
			csv_file = pd.read_csv(filename, sep = ',', header = None).values
			flag_add = True

		if (flag_add):

			for i in tqdm(range(len(csv_file))):
				
				fname = csv_file[i,0].split('.')[0]
				fname_sample = folder_path + fname
				fname_feature = FEAT_FLD + fname + '.npy'
				fname_report = folder_reports + fname + '.txt'
				label = int(csv_file[i,1])

				if (os.path.exists(fname_report) and label not in FILTER_LABELS):
					list_images.append(fname_sample)
					list_features.append(fname_feature)
					list_reports.append(fname_report)

	list_images = np.array(list_images)
	list_features = np.array(list_features)
	list_reports = np.array(list_reports)

	data_to_use = np.vstack((list_images, list_features, list_reports), dtype = 'object').T
	#data_to_use = np.array([[list_images[i], list_labels[i]] for i in range(len(list_labels[i]))])

	return data_to_use

def get_label_vqa(fname, dataset_color, dataset_border, dataset_symm, dataset_dermo):

	i = 0
	b = False
	color = None
	border = None
	symm = None
	dermo = None

	while(i<len(dataset_color) and b == False):

		current_fname = dataset_color[i,0].split('.')[0]

		if (current_fname in fname or fname in current_fname):
			b = True
			color = dataset_color[i,1:]
			border = dataset_border[i,1]
			symm = dataset_symm[i,1]
			dermo = dataset_dermo[i,1]

		else:
			i = i + 1

	return color, border, symm, dermo

def get_instances_vqa(MAIN_FOLDER, list_datasets, flag, CNN_TO_USE, MODALITY, N_EXP, CSV_FOLDER, TYPE_DOC = False, flag_KEYWORDS = False):

	
	list_features = []
	list_symmetry = []
	list_border = []
	list_color = []
	list_dermo = []


	for l in list_datasets:
		
		FEAT_FLD = MAIN_FOLDER+l+'/features_'+CNN_TO_USE+'/'+MODALITY+'/'

		if (TYPE_DOC):
			FEAT_FLD = FEAT_FLD + 'DOC/'
		else:
			FEAT_FLD = FEAT_FLD + 'NO_DOC/'

		if (flag_KEYWORDS):
			FEAT_FLD = FEAT_FLD + 'keywords/'
		else:
			FEAT_FLD = FEAT_FLD + 'no_keywords/'

		FEAT_FLD = FEAT_FLD+'/N_EXP_'+N_EXP+'//images/'

		flag_add = False

		folder_path = MAIN_FOLDER+l+'/resized_images/'
		folder_reports = MAIN_FOLDER+l+'/clinical_notes/short_reports/'

		folder_csv = CSV_FOLDER+l+'/'

		try:
			dermoscopic_fname = folder_csv + 'dermoscopic.csv'
			color_fname = folder_csv + 'color.csv'
			border_fname = folder_csv + 'border.csv'
			symmetry_fname = folder_csv + 'symmetry.csv'

			csv_file_dermo = pd.read_csv(dermoscopic_fname, sep = ',', header = None).values
			csv_file_color = pd.read_csv(color_fname, sep = ',', header = None).values
			csv_file_border = pd.read_csv(border_fname, sep = ',', header = None).values
			csv_file_symmetry = pd.read_csv(symmetry_fname, sep = ',', header = None).values

			
			if (flag is PHASE.train):

				try:
					filename = folder_csv + 'labels_train.csv'
					csv_file = pd.read_csv(filename, sep = ',', header = None).values
					flag_add = True
				except Exception as e:
					print(e)
					pass

			elif (flag is PHASE.valid):

				try:
					filename = folder_csv + 'labels_valid.csv'
					csv_file = pd.read_csv(filename, sep = ',', header = None).values
					flag_add = True
				except Exception as e:
					print(e)
					pass
			
			elif (flag is PHASE.all):

				filename = folder_csv + 'labels.csv'
				csv_file = pd.read_csv(filename, sep = ',', header = None).values
				flag_add = True

			if (flag_add):

				for i in tqdm(range(len(csv_file))):
					
					fname = csv_file[i,0].split('.')[0]
					fname_sample = folder_path + fname
					fname_feature = FEAT_FLD + fname + '.npy'

					list_features.append(fname_feature)

					color, border, symm, dermo = get_label_vqa(fname, csv_file_color, csv_file_border, csv_file_symmetry, csv_file_dermo)
					list_symmetry.append(symm)
					list_border.append(border)
					list_color.append(color)
					list_dermo.append(dermo)

		except Exception as ex:
			print(ex)
			pass

	list_features = np.array(list_features)
	list_symmetry = np.array(list_symmetry)
	list_border = np.array(list_border)
	list_color = np.array(list_color)
	list_dermo = np.array(list_dermo)

	#data_to_use = np.array([[list_images[i], list_labels[i]] for i in range(len(list_labels[i]))])

	return list_features, list_symmetry, list_border, list_color, list_dermo

def get_json_line(fname, json_lookup, elems = 30):
    for key in json_lookup:
        if key in fname or fname in key:
            qa_items = json_lookup[key]
            questions = [entry['question'] for entry in qa_items[:elems]]
            answers = [entry['finding'] for entry in qa_items[:elems]]
            return questions, answers

    # If no match is found, raise an error or return None
    raise ValueError(f"No matching JSON entry found for file: {fname}")


def build_json_lookup(json_data):
    lookup = defaultdict(list)
    for item in json_data:
        lookup[item['file_id']].append(item)
    return lookup

def get_instances_vqa_json(MAIN_FOLDER, list_datasets, flag, CNN_TO_USE, MODALITY, N_EXP, CSV_FOLDER, FILTER_LABELS = [1,3], REPORTS_TRAINING = 'abcd', flag_KEYWORDS = False, REPORT_AUGMENTATION = False):
	list_features = []
	list_questions = []
	list_answers = []

	for l in list_datasets:

		FEAT_FLD = MAIN_FOLDER+l+'/features_'+CNN_TO_USE+'/'+MODALITY+'/'

		FEAT_FLD = FEAT_FLD + REPORTS_TRAINING + '/'

		if (flag_KEYWORDS):
			FEAT_FLD = FEAT_FLD + 'keywords/'
		else:
			FEAT_FLD = FEAT_FLD + 'no_keywords/'

		if (REPORT_AUGMENTATION):
			FEAT_FLD = FEAT_FLD + 'report_augmentation/'
		else:
			FEAT_FLD = FEAT_FLD + 'no_report_augmentation/'

		FEAT_FLD = FEAT_FLD+'/N_EXP_'+N_EXP+'//images/'
		
		folder_csv = CSV_FOLDER + l + '/'
		json_fname = folder_csv + l + '_VQA_structured_data.json'

		# Load and index JSON
		with open(json_fname, "r") as f:
			json_data = [json.loads(line) for line in f]
		json_lookup = build_json_lookup(json_data)

		# Load labels based on phase
		flag_add = False
		try:
			if flag == PHASE.train:
				filename = folder_csv + 'labels_train.csv'
			elif flag == PHASE.valid:
				filename = folder_csv + 'labels_valid.csv'
			elif flag == PHASE.test:
				filename = folder_csv + 'labels_test.csv'
			elif flag == PHASE.all:
				filename = folder_csv + 'labels.csv'
			else:
				continue
			csv_file = pd.read_csv(filename, header=None).values#[:, 0]
			flag_add = True
		except Exception as e:
			print(e)

		if flag_add:
			for i in tqdm(range(len(csv_file))):
				fname = csv_file[i,0].split('.')[0]
				fname_feature = FEAT_FLD + fname + '.npy'
				label = int(csv_file[i,1])

				if (label not in FILTER_LABELS):
					try:
						questions, answers = get_json_line(fname, json_lookup)
						for q, a in zip(questions, answers):
							if q is not None and a is not None:
								list_features.append(fname_feature)
								list_questions.append(q)
								list_answers.append(a)
					except:
						pass

	return list_features, list_questions, list_answers

def get_flag_embeddings_exist(test_dataset, TYPE_REPORT):
	
	if (TYPE_REPORT is REPORTS.all or TYPE_REPORT is REPORTS.random):
		list_possible = ['gpt_4_mini_as_doctor', 'gpt_4o_mini_fld_abcd', 'gpt_4o_mini_fld', 'short_reports']
		#list_possible = ['gpt_4_mini_as_doctor', 'gpt_4o_mini_fld_abcd', 'short_reports']

	elif (TYPE_REPORT is REPORTS.abcd):
		list_possible = ['gpt_4o_mini_fld_abcd']

	elif (TYPE_REPORT is REPORTS.char):
		list_possible = ['gpt_4o_mini_fld']

	elif (TYPE_REPORT is REPORTS.doc):
		list_possible = ['gpt_4_mini_as_doctor']

	elif (TYPE_REPORT is REPORTS.short):
		list_possible = ['short_reports']

	elif (TYPE_REPORT is REPORTS.meta):
		list_possible = ['gpt_4o_mini_fld_abcd', 'gpt_4o_mini_fld', 'short_reports']
		#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

	elif (TYPE_REPORT is REPORTS.abcd_no_label):
		list_possible = ['gpt_4o_mini_fld_abcd_no_label']

	elif (TYPE_REPORT is REPORTS.char_no_label):
		list_possible = ['gpt_4o_mini_fld_no_label']

	elif (TYPE_REPORT is REPORTS.doc_no_label):
		list_possible = ['gpt_4_mini_as_doctor_no_label']

	
	elif (TYPE_REPORT is REPORTS.abcd_o4):
		list_possible = ['gpt_4o_fld_abcd']

	elif (TYPE_REPORT is REPORTS.char_o4):
		list_possible = ['gpt_4o_fld']

	elif (TYPE_REPORT is REPORTS.doc_o4):
		list_possible = ['gpt_4o_fld_as_doctor']

	elif (TYPE_REPORT is REPORTS.meta_o4):
		list_possible = ['gpt_4o_fld_abcd', 'gpt_4o_fld', 'short_reports']

	elif (TYPE_REPORT is REPORTS.all_o4):
		list_possible = ['gpt_4o_mini_fld_abcd', 'gpt_4o_mini_fld', 'gpt_4o_fld_as_doctor', 'short_reports']

	i = 0
	flag_ok = True

	while (i<len(test_dataset) and flag_ok):

		current_fname = test_dataset[i]
		current_flag = True

		x = 0
		flag_current = True
		while (x < len(list_possible) and flag_current):

			fname = current_fname.replace('short_reports',list_possible[x])

			current_flag = os.path.exists(fname)

			x = x + 1

		flag_ok = flag_ok and current_flag
		
		i = i + 1

	return flag_ok

if __name__ == "__main__":
	pass