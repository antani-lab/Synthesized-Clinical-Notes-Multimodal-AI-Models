import sys, getopt
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from enum_multi import MOD, TYPE_REPORT
import random
from tqdm import tqdm
import utils_data
import os

def eval_mAP(sorted_indices, query_labels, pool_labels):

    average_precisions = []

    for i in range(len(query_labels)):
        query_label = query_labels[i]
        ranked_indices = sorted_indices[i]
        ranked_labels = pool_labels[ranked_indices]

        relevant = (ranked_labels == query_label)
        num_relevant = np.sum(relevant)

        if num_relevant == 0:
            average_precisions.append(0.0)
            continue

        hits = 0
        precisions = []
        for rank, is_rel in enumerate(relevant, start=1):
            if is_rel:
                hits += 1
                precisions.append(hits / rank)

        ap = np.mean(precisions) if precisions else 0.0
        average_precisions.append(ap)

    mAP = np.mean(average_precisions)

    return mAP

def eval_precision_recall(sorted_indices, label_input, pool_labels, k = 5):

    precisions = []
    recalls = []
    avg_precision = 0
    avg_recall = 0

    for i in range(len(label_input)):

        label = label_input[i]

        retrieved_labels = pool_labels[sorted_indices[i]]

        k_elements = retrieved_labels[:k]

        relevant_k = np.sum(k_elements == label)

        relevant_tot = np.sum(retrieved_labels == label)

        precision = relevant_k / k
        precisions.append(precision)
        
        #recall = relevant_k / relevant_tot
        #recalls.append(recall)

    avg_precision = np.mean(precisions)
    #avg_recall = np.mean(recalls)
    return avg_precision, avg_recall

def is_array_in_list(array, array_list):
    return any(np.array_equal(array, existing) for existing in array_list)


def estimate_n_features(list_folder, csv_dataset):

    count = 0
    for feat_fold_test in list_folder:
        for i in range(len(csv_dataset)):
            fname = feat_fold_test + csv_dataset[i, 0].split('.')[0] + '.npy'
            if not os.path.isfile(fname):
                continue
            else:
                count += 1
    return count


def get_input_features(list_folder, csv_dataset, N_ELEMENTS = 100000):

    
    features = np.empty((N_ELEMENTS, 128))
    labels = np.empty((N_ELEMENTS))
    idx = 0

    for feat_fold_test in list_folder:

        for i in tqdm(range(len(csv_dataset))):
            fname = feat_fold_test + csv_dataset[i,0].split('.')[0] + '.npy'
            label = csv_dataset[i,1]

            try:
                with open(fname, 'rb') as f:
                    cls_img = np.load(f)
                
                features[idx] = cls_img
                labels[idx] = int(label)
                idx = idx + 1

            except Exception as e:
                print(e)

    return features, labels, idx


def get_pool_features(list_folder, csv_dataset, features, labels, idx, seen, MODALITY = MOD.img):

    idx_to_use = idx
    
    for feat_fold_test in list_folder:

        for i in tqdm(range(len(csv_dataset))):
            fname = feat_fold_test + csv_dataset[i,0].split('.')[0] + '.npy'
            label = csv_dataset[i,1]

            try:
                with open(fname, 'rb') as f:
                    cls_img = np.load(f)

                if (MODALITY is MOD.txt):
                    features[idx_to_use] = cls_img
                    labels[idx_to_use] = int(label)
                    idx_to_use = idx_to_use + 1

                elif (MODALITY is MOD.img):
                    key = cls_img.tobytes()
                    #if not np.any(np.all(elements_to_check == cls_img, axis=1)):
                    if (key not in seen):
                        features[idx_to_use] = cls_img
                        labels[idx_to_use] = int(label)
                        idx_to_use = idx_to_use + 1
                        seen.add(key)

            except Exception as e:
                #print(e)
                pass
    
    print(idx_to_use)
    return features, labels, idx_to_use, seen

def get_feature_dataset(main_fld, dataset):

    return main_fld.replace('DATASET', dataset)

def get_pool_data(POOL_DATASETS, MOD_INPUT, CSV_FOLDER, MAIN_FLD, DIM_RESIZE = 128, MAX_ELEMENT = 100000, flag_o4 = False):

    #POOL: evaluate the number of size needed 

    feature_pool = np.empty((MAX_ELEMENT, 128))
    label_pool = np.empty((MAX_ELEMENT))
    
    seen = set()

    idx = 0

    print("pool data")
    for d in POOL_DATASETS:
        csv_test_file = CSV_FOLDER + d + '/labels.csv'
        csv_dataset = pd.read_csv(csv_test_file, sep = ',', header = None).values

        set_to_filter = [1, 3]
        csv_dataset = utils_data.filter_labels(csv_dataset, set_to_filter) 

        fold_test = get_feature_dataset(MAIN_FLD, d)

        if (MOD_INPUT is MOD.img and flag_o4 == False):
            print("no o4")
            #list_folder = [fold_test + 'reports_abcd/', fold_test + 'reports_shorts/']
            list_folder = [fold_test + 'reports_abcd/', fold_test + 'reports_short/', fold_test + 'reports_doc/', fold_test + 'reports_char/']
            """
            list_folder = [fold_test + 'reports_abcd/', fold_test + 'reports_shorts/', fold_test + 'reports_doc/', fold_test + 'reports_char/', 
                            fold_test + 'reports_abcd_no_label/', fold_test + 'reports_doc_no_label/', fold_test + 'reports_char_no_label/']
            """
        elif (MOD_INPUT is MOD.img and flag_o4 == True):
            print("o4")
            #list_folder = [fold_test + 'reports_abcd/', fold_test + 'reports_shorts/']
            list_folder = [fold_test + 'reports_abcd_o4/', fold_test + 'reports_short/', fold_test + 'reports_doc_o4/', fold_test + 'reports_char_o4/']
            """
            list_folder = [fold_test + 'reports_abcd/', fold_test + 'reports_shorts/', fold_test + 'reports_doc/', fold_test + 'reports_char/', 
                            fold_test + 'reports_abcd_no_label/', fold_test + 'reports_doc_no_label/', fold_test + 'reports_char_no_label/']
            """
        
        elif (MOD_INPUT is MOD.txt):
            list_folder = [fold_test + 'images/']

        feature_pool, label_pool, idx, seen = get_pool_features(list_folder, csv_dataset, feature_pool, label_pool, idx, seen, MOD_INPUT)

    
    feature_pool = feature_pool[:idx]
    label_pool = label_pool[:idx]

    N_ELEM = idx
    feature_pool = np.reshape(feature_pool, (N_ELEM, DIM_RESIZE))

    return feature_pool, label_pool

def get_test_data(DATASET, MOD_INPUT, CSV_FOLDER, MAIN_FLD, DIM_RESIZE = 128, REPORT_type = TYPE_REPORT.abcd, MAX_ELEMENT = 500000):

    #print(REPORT_type)
    feature_input = np.empty((MAX_ELEMENT, 128))
    label_input = np.empty((MAX_ELEMENT))

    idx = 0

    #INPUT
    #print("input data")
    csv_test_file = CSV_FOLDER + DATASET + '/labels_test.csv'
    csv_dataset = pd.read_csv(csv_test_file, sep = ',', header = None).values

    set_to_filter = [1, 3]
    test_dataset = utils_data.filter_labels(csv_dataset, set_to_filter) 
    
    fold_test = get_feature_dataset(MAIN_FLD, DATASET)

    if (MOD_INPUT is MOD.img):
        list_folder = [fold_test + 'images/']

    elif (MOD_INPUT is MOD.txt):
        #list_folder = [fold_test + 'reports_abcd/', fold_test + 'reports_shorts/']
        #list_folder = [fold_test + 'reports_abcd/', fold_test + 'reports_shorts/', fold_test + 'reports_doc/']
        if (REPORT_type is TYPE_REPORT.abcd):
            list_folder = [fold_test + 'reports_abcd/']
        elif (REPORT_type is TYPE_REPORT.short):
            list_folder = [fold_test + 'reports_shorts/']
        elif (REPORT_type is TYPE_REPORT.doc):
            list_folder = [fold_test + 'reports_doc/']
        elif (REPORT_type is TYPE_REPORT.char):
            list_folder = [fold_test + 'reports_char/']
        elif (REPORT_type is TYPE_REPORT.abcd_no_label):
            list_folder = [fold_test + 'reports_abcd_no_label/']
        elif (REPORT_type is TYPE_REPORT.doc_no_label):
            list_folder = [fold_test + 'reports_doc_no_label/']
        elif (REPORT_type is TYPE_REPORT.char_no_label):
            list_folder = [fold_test + 'reports_char_no_label/']
        elif (REPORT_type is TYPE_REPORT.all):
            list_folder = [fold_test + 'reports_abcd/', fold_test + 'reports_shorts/', fold_test + 'reports_char/', fold_test + 'reports_doc/']
        elif (REPORT_type is TYPE_REPORT.abcd_o4):
            list_folder = [fold_test + 'reports_abcd_o4/']
        elif (REPORT_type is TYPE_REPORT.doc_o4):
            list_folder = [fold_test + 'reports_doc_o4/']
        elif (REPORT_type is TYPE_REPORT.char_o4):
            list_folder = [fold_test + 'reports_char_o4/']
        elif (REPORT_type is TYPE_REPORT.all_o4):
            list_folder = [fold_test + 'reports_abcd_o4/', fold_test + 'reports_doc_o4/', fold_test + 'reports_char_o4/', fold_test + 'reports_shorts/']
        

    feature_input, label_input, idx = get_input_features(list_folder, test_dataset)

    feature_input = feature_input[:idx]
    label_input = label_input[:idx]

    N_ELEM = idx
    feature_input = np.reshape(feature_input, (N_ELEM, DIM_RESIZE))

    return feature_input, label_input

def save_metric(MODEL_PATH, METRIC, DATASET, MODALITY, VALUE, type_report = 'abcd'):

    if (MODALITY == 'txt'):
        MODALITY = MODALITY + '_' + type_report

    fname_file = MODEL_PATH + METRIC + '_' + DATASET + '_' + MODALITY + '.csv'

    File = {'val' : [VALUE]}
    df = pd.DataFrame(File,columns=['val'])
    np.savetxt(fname_file, df.values, fmt='%s',delimiter=',')

if __name__ == "__main__":
	pass