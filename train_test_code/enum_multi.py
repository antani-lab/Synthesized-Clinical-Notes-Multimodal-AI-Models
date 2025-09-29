from enum import Enum

class DATASET_TO_USE(Enum):
	HAM10000 = 0 #ok
	BCN20000 = 1 #ok
	DermNet = 2 #ok
	Derm7pt = 3 #ok
	DDI = 4 #ok
	Fitzpatrick17k = 5 #ok
	Hospital_Italiano_Buenos_Aires = 6 #ok
	PAD_UFES_20 = 7 #ok
	PH2 = 8 #ok
	SD198 = 9 #ok
	derm12345 = 10 #ok
	SKINL2 = 11 #ok
	MRA_MIDAS = 12


class NETWORK(Enum):
	resnet18 = 0 #ok
	densenet121 = 1
	mobilenet_v2 = 2
	HIPT = 3
	ViT = 4
	ViT2 = 5

class PORTION(Enum):
	skin = 0
	lesion = 1
	all = 2
	both = 3

class SELF_SUPERVISION(Enum):
	simCLR = 0
	MoCO = 1
	DINO = 2

class PHASE(Enum):
	train = 0
	valid = 1
	test = 2
	all = 3

class CLASSES(Enum):
    seborrheic = 0
    dermatofibroma = 1
    nevus = 2
    vascular = 3
    actinic = 4
    basal = 5
    melanoma = 6

class COLOR_SPACE(Enum):
	RGB = 0
	HSV = 1
	Lab = 2

class TYPE_DATA(Enum):
	original = 0
	modified = 1
	both = 2

class MOD(Enum):
	img = 0
	txt = 1
	multimodal = 2
	CLIP = 3
	NT_Xent = 4
	InfoNCE = 5
	InfoNCE_supervised = 6
	NT_Xent_only = 7
	InfoNCE_only = 8
	CLIP_only = 9

class TYPE_REPORT(Enum):
	abcd = 0
	short = 1
	char = 2
	doc = 3
	all = 4
	abcd_no_label = 5
	char_no_label = 6
	doc_no_label = 7
	images = 8
	abcd_o4 = 9
	char_o4 = 10
	doc_o4 = 11
	all_o4 = 12

class QUESTION(Enum):
	symmetric = 0
	color = 1
	border = 2
	dermo = 3

class COMPONENTS(Enum):
	images = 0
	reports = 1
	keywords = 2

class REPORTS(Enum):
	abcd = 0
	char = 1
	doc = 2
	short = 3 
	meta = 4 #abcd + char + short
	all = 5 #abcd + char + short + doc
	random = 6 #
	abcd_no_label = 7
	char_no_label = 8
	doc_no_label = 9
	images = 10
	abcd_o4 = 11
	char_o4 = 12
	doc_o4 = 13
	meta_o4 = 14 #abcd + char + short
	all_o4 = 15 #abcd + char + short + doc

class vit_pool(Enum):
	cls = 0
	mean = 1
	none = 2

if __name__ == "__main__":
	pass