# Synthesized-Clinical-Notes-Multimodal-AI-Models
This repository will include models and methods to generate synthetic clinical notes describing dermatology images. Clinical notes are paired with images to train a multimodal foundation model.

## Reference
If you find this repository useful in your research, please cite:
[1] Marini N., Liang Z., Rajaraman S., Xue Z., Antani S., Mitigating Hallucinations in Synthesized Clinical Texts to Improve Multimodal Deep Learning for Dermatology, Journal of Biomedical Informatics
[2] Marini N., Liang Z., Rajaraman S., Xue Z., Antani S., Synthesized Clinical Notes Enable Training Robust Multimodal AI Models from Unimodal Dermatology Datasets,


## Requirements
python==3.9.21, torch==2.8.0, torchvision==0.23.0, transformers==4.57.6, albumentations==2.0.8, huggingface-hub==0.36.2, imageio==2.37.0 , numba==0.60.0, numpy==1.25.2, pandas==2.2.3, pillow==11.1.0, scikit-image==0.24.0, scikit-learn==1.6.1, scipy==1.9.3, timm==1.0.19

## Datasets
All datasets are publicly available:
-   [BCN20000](https://api.isic-archive.com/collections/249/)
-   [derm12345](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DAXZ7P)
-   [Derm7pt](https://derm.cs.sfu.ca/Welcome.html)
-   [DermNet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet)
-   [FLUO_SC](https://data.mendeley.com/datasets/s8n68jj678/1)
-   [MRA_MIDAS](https://aimi.stanford.edu/datasets/mra-midas-Multimodal-Image-Dataset-for-AI-based-Skin-Cancer)
-   [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
-   [SKINL2](https://www.it.pt/AutomaticPage?id=3459)
-   [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k)
-   [Hospital Italiano Buenos Aires](https://api.isic-archive.com/collections/251/)
-   [PAD UFES 20](https://data.mendeley.com/datasets/zr7vgbcyr2/1)
-   [SD198](https://huggingface.co/datasets/resyhgerwshshgdfghsdfgh/SD-198)
-   MSK: composed from [MSK1](https://api.isic-archive.com/collections/289/), [MSK2](https://api.isic-archive.com/collections/290/), [MSK3](https://api.isic-archive.com/collections/288/), [MSK4](https://api.isic-archive.com/collections/287/), [MSK5](https://api.isic-archive.com/collections/286/)
-   [Milk10k](https://challenge.isic-archive.com/landing/milk10k/)
-   [PH2](https://www.fc.up.pt/addi/ph2%20database.html)

## Benchmark LLMs
The benchmark LLMs used to synthesize clinical notes are publicly available:
-   [GPT](https://developers.openai.com/api/reference/python)
-   [MedGemma](https://huggingface.co/google/medgemma-4b-it)
-   [SkinGPT4](https://github.com/JoshuaChou2018/SkinGPT-4?tab=readme-ov-file)
-   [DermLIP](https://huggingface.co/moxeeeem/dermlip-gpt2-captioner)
Create different environment sto use the LLMs, following the specific repository implementation.

## State-of-the-art foundation models
The benchmark foundation models used as benchmark for our model are publicly available:
-   [BioMedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
-   [MedImageInsights](https://huggingface.co/lion-ai/MedImageInsights)
-   [MONET](https://huggingface.co/suinleelab/monet)
-   [Derm1M](https://huggingface.co/datasets/redlessone/Derm1M/tree/main)
Create different environments to use the foundation, following the specific repository implementation.


## Repository organization
The repository includes four main folders, including:
-   Methods to pre-process the images (i.e. extract the skin lesion from the image and resize to 224x224)
-   Methods to synthesize clinical notes
-   Methods to train and test models
-   Csv files to load.

## Labels adopted:
The foundation model is trained in the first step with eight classes, as follows (with abbreviation, class id, malignacy): 
-   seborrheic keratosis, BEK, 0, benign
-   Dermatofibroma, DF, 1, benign
-   benign nevus, NEV, 2, benign
-   vascular lesion, VASC, 3, benign
-   acktinic keratosis, ACK, 4, pre-cancerous
-   basal cell cancer, BCC, 5, malignant
-   melanoma, BCC, 6, malignant
-   squamous cell cancer, SCC, 7, malignant

## Csv folder
The folder includes the csv files used to develop methods.
- labels.csv: it includes the ID of the samples and the corresponding label.
- labels_train.csv: if available, the IDs of the samples used for training, with the corresponding labels.
- labels_train.csv: if available, the IDs of the samples used for validation, with the corresponding labels.
- labels_test.csv: the IDs of the samples used for testing, with the corresponding labels.

Check the original repositories for a full perspective on metadata. 
The folder includes also a file including metadata collected from original repositories: classes_subclasses_metadata_mapping.csv. 
The file includes multiple column: 
-   sample ID: str 
-   label: int (numeric)
-   malignacy: benign, pre-cancerous, malignant
-   lesion: lesion type, as described in the original metadata.
-   subclass: specific type of lesion, as described in the original metadata
-   findings: if available, additional details described in the paper
-   matching class: class used to describe the lesion (matching across datasets) 
-   matching subclass: subclass used to describe the lesion (matching across datasets) 

## Pre-trained weights


## Acknoledgements
This research was supported by the Intramural Research Program of the National Institutes of Health (NIH). The contributions of the NIH author(s) were made as part of their official duties as NIH federal employees, are in compliance with agency policy requirements, and are considered Works of the United States Government. This work utilized the computational resources of the NIH HPC Biowulf cluster (https://hpc.nih.gov). 
