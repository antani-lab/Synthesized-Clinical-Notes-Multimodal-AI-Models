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
	folder_reports = MAIN_FOLDER+DATASET+'/clinical_notes/short_reports/'
	
	if (CSV_FOLDER is not None):
		folder_csv = CSV_FOLDER+DATASET+'/'
	else:
		folder_csv = MAIN_FOLDER+DATASET+'/resized_images/'
	print(folder_csv)
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
	folder_reports = MAIN_FOLDER+DATASET+'/clinical_notes/shorts/'
	
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

	print(csv_file.shape)
	print(filename)

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




def get_instances_paths_from_bags(MAIN_FOLDER, list_datasets, flag, CSV_FOLDER = None, PERCENTAGE = -1):

	list_images = []
	list_labels = []
	list_reports = []

	for l in list_datasets:

		flag_add = False

		folder_path = MAIN_FOLDER+l+'/resized_images/'
		#folder_reports = MAIN_FOLDER+l+'/clinical_notes/gpt_4o_mini_fld/'
		folder_reports = MAIN_FOLDER+l+'/clinical_notes/shorts/'

		if (CSV_FOLDER is not None):
			folder_csv = CSV_FOLDER+l+'/'
		else:
			folder_csv = MAIN_FOLDER+l+'/resized_images/'

		if (flag is PHASE.train):
			
			if (PERCENTAGE == -1):

				try:
					filename = folder_csv + 'labels_train.csv'
					csv_file = pd.read_csv(filename, sep = ',', header = None).values
					flag_add = True
				except Exception as e:
					print(e)
					pass
			
			else:
				s = f"{PERCENTAGE:.2f}".replace(".", "")

				try:
					filename = folder_csv + "/labels_train_filtered_"+str(s)+".csv"
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



def get_instances_paths_txt(MAIN_FOLDER, list_datasets, flag, CSV_FOLDER = None, flag_synthetic = False, CSV_SYNTHETIC_folder = None, DATA_SYNTHETIC_folder = None):

	list_images = []
	list_labels = []
	list_reports = []

	CSV_SYNTHETIC_fname = CSV_SYNTHETIC_folder + 'synthetic_data.csv'

	for l in list_datasets:

		flag_add = False


		folder_reports_1 = MAIN_FOLDER+l+'/clinical_notes/short_reports/'
		folder_reports_2 = MAIN_FOLDER+l+'/clinical_notes/gpt_4o_mini_fld/'
		folder_reports_3 = MAIN_FOLDER+l+'/clinical_notes/gpt_4o_mini_fld_abcd/'

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
				
				fname_report_1 = folder_reports_1 + csv_file[i,0].split('.')[0] + '.txt'
				fname_report_2 = folder_reports_2 + csv_file[i,0].split('.')[0] + '.txt'
				fname_report_3 = folder_reports_3 + csv_file[i,0].split('.')[0] + '.txt'

				if (os.path.exists(fname_report_1)):
					list_reports.append(fname_report_1)
					list_labels.append(int(csv_file[i,1]))
				
				if (os.path.exists(fname_report_2)):
					list_reports.append(fname_report_2)
					list_labels.append(int(csv_file[i,1]))

				if (os.path.exists(fname_report_3)):
					list_reports.append(fname_report_3)
					list_labels.append(int(csv_file[i,1]))

		if (flag is PHASE.train and flag_synthetic):

			csv_file = pd.read_csv(CSV_SYNTHETIC_fname, sep = ',', header = None).values

			for i in range(len(csv_file)):
				list_reports.append(DATA_SYNTHETIC_folder + csv_file[i,0])
				list_labels.append(int(csv_file[i,1]))

	list_labels = [int(x) for x in list_labels]		
	list_labels = np.array(list_labels)
	list_reports = np.array(list_reports)

	data_to_use = np.vstack((list_reports, list_labels), dtype = 'object').T
	#data_to_use = np.array([[list_images[i], list_labels[i]] for i in range(len(list_labels[i]))])

	return data_to_use


def get_style_transfer(filename):
	current_sample = filename
	idx_to_split = -1
	if ('home' in filename):
		idx_to_split = 6
	else:
		idx_to_split = 5
	x = current_sample.split('/')
	filename = x[-1]
	current_dir = x[-3]
	MAIN_FOLDER = '/'.join(x[:idx_to_split])
	new_folder = MAIN_FOLDER + '/' + current_dir + '/resized_image_style_transfer/'

	sample_id = filename.split('.')[0]
	format = filename.split('.')[-1]

	idx_fitz = np.random.randint(0,6)

	new_sample_filename = new_folder + sample_id + '_' + str(idx_fitz) + '.' + format
	ID = new_sample_filename

	return ID

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

def get_instances_txt_generation(MAIN_FOLDER, list_datasets, flag, CNN_TO_USE, MODALITY, N_EXP, CSV_FOLDER = None, FILTER_LABELS = [], REPORTS_TRAINING = 'abcd', flag_KEYWORDS = False, REPORT_AUGMENTATION = False):

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

def get_flag_embeddings_exist(test_dataset, TYPE_REPORT):
	
	if (TYPE_REPORT is REPORTS.random):
		list_possible = ['doc', 'abcd', 'char', 'shorts']
		#list_possible = ['doc', 'abcd', 'shorts']

	elif (TYPE_REPORT is REPORTS.abcd):
		list_possible = ['abcd']

	elif (TYPE_REPORT is REPORTS.char):
		list_possible = ['char']

	elif (TYPE_REPORT is REPORTS.doc):
		list_possible = ['doc']

	elif (TYPE_REPORT is REPORTS.short):
		list_possible = ['shorts']

	elif (TYPE_REPORT is REPORTS.meta):
		list_possible = ['abcd', 'char',]
		#list_possible = ['abcd', 'shorts']

	elif (TYPE_REPORT is REPORTS.all):
		list_possible = ['abcd', 'char','shorts']
		#list_possible = ['abcd', 'shorts']

	###skingpt4
	elif (TYPE_REPORT is REPORTS.skingpt4_abcd):
		list_possible = ['skingpt4_abcd']

	elif (TYPE_REPORT is REPORTS.skingpt4_char):
		list_possible = ['skingpt4_char']

	elif (TYPE_REPORT is REPORTS.skingpt4_doc):
		list_possible = ['skingpt4_doc']

	elif (TYPE_REPORT is REPORTS.skingpt4_meta):
		list_possible = ['skingpt4_abcd', 'skingpt4_char']
		#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

	elif (TYPE_REPORT is REPORTS.skingpt4_all):
		list_possible = ['skingpt4_abcd', 'skingpt4_char', 'shorts']
		#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

	elif (TYPE_REPORT is REPORTS.skingpt4_p1):
		list_possible = ['skingpt4_p1']
		#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

	elif (TYPE_REPORT is REPORTS.skingpt4_p1_all):
		list_possible = ['skingpt4_p1', 'shorts']
		#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

	elif (TYPE_REPORT is REPORTS.skingpt4_p2):
		list_possible = ['skingpt4_p2']
		#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

	###dermlip
	elif (TYPE_REPORT is REPORTS.dermlip_abcd):
		list_possible = ['derm_1M_abcd']

	elif (TYPE_REPORT is REPORTS.dermlip_char):
		list_possible = ['derm_1M_char']

	elif (TYPE_REPORT is REPORTS.dermlip_doc):
		list_possible = ['derm_1M_doc']

	elif (TYPE_REPORT is REPORTS.dermlip_meta):
		list_possible = ['derm_1M_abcd', 'derm_1M_char']
		#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

	elif (TYPE_REPORT is REPORTS.dermlip_all):
		list_possible = ['derm_1M_abcd', 'derm_1M_char', 'shorts']
		#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

	elif (TYPE_REPORT is REPORTS.dermlip_p1):
		list_possible = ['derm_1M_p1']
		#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

	elif (TYPE_REPORT is REPORTS.dermlip_p1_all):
		list_possible = ['derm_1M_p1', 'shorts']

	elif (TYPE_REPORT is REPORTS.dermlip_p2):
		list_possible = ['derm_1M_p2']
		#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

	###medgemma
	elif (TYPE_REPORT is REPORTS.medgemma_abcd):
		list_possible = ['medgemma_abcd']

	elif (TYPE_REPORT is REPORTS.medgemma_char):
		list_possible = ['medgemma_char']

	elif (TYPE_REPORT is REPORTS.medgemma_doc):
		list_possible = ['medgemma_doc']

	elif (TYPE_REPORT is REPORTS.medgemma_meta):
		list_possible = ['medgemma_abcd', 'medgemma_char']
		#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

	elif (TYPE_REPORT is REPORTS.medgemma_all):
		list_possible = ['medgemma_abcd', 'medgemma_char', 'shorts']
		#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

	elif (TYPE_REPORT is REPORTS.whole):
		list_possible = ['medgemma_abcd', 'medgemma_char','derm_1M_p1', 'skingpt4_p1', 'abcd', 'char']
		#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

	elif (TYPE_REPORT is REPORTS.whole_all):
		list_possible = ['medgemma_abcd', 'medgemma_char','derm_1M_p1', 'skingpt4_p1', 'abcd', 'char', 'shorts']
		#list_possible = ['gpt_4o_mini_fld_abcd', 'short_reports']

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


def filter_empty_concepts(dataset):

	MAX_ELEMENTS = 1000000
	new_dataset = np.empty((MAX_ELEMENTS, 3), dtype = "object")

	cont = 0
	#for i in tqdm(range(10)):
	for i in tqdm(range(len(dataset))):
		fname = dataset[i,0]
		label = dataset[i,1]

		placeholder_report = dataset[i,2]

		for p in possible_reports:

			ID_txt = placeholder_report.replace('short_reports',p)

			input_txt = utils_txt.load_txt(ID_txt)

			try:
				concepts = utils_concept_extraction.find_present_concepts(input_txt, utils_concept_extraction.GLOBAL_CONCEPTS, utils_concept_extraction.CONFLICTS)
			except:
				
				input_txt = utils_txt.load_txt(ID_txt)
				concepts = utils_concept_extraction.find_present_concepts(input_txt, utils_concept_extraction.GLOBAL_CONCEPTS, utils_concept_extraction.CONFLICTS)


			concepts = utils_concept_extraction.flatten_and_remove(concepts, value_to_remove = -1)

			if (concepts != []):

				#print(concepts)
				#list_reports_divided_class[label].append(ID_txt)
				new_dataset[cont,0] = fname
				new_dataset[cont,1] = label
				new_dataset[cont,2] = ID_txt
				cont = cont + 1

	new_dataset = new_dataset[:cont]
	return new_dataset
	

if __name__ == "__main__":
	pass