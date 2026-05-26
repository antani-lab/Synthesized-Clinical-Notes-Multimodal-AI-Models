from enum import Enum

class TYPE_IMAGE(Enum):
	dermoscopy = 0
	natural = 1

class NETWORK(Enum):
	resnet18 = 0 #ok
	densenet121 = 1
	mobilenet_v2 = 2
	HIPT = 3
	ViT = 4
	ViT2 = 5
	PanDerm = 6
	

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
	NT_Xent_InfoNCE = 10
	NT_Xent_InfoNCE_supervised = 11


class TYPE_REPORT(Enum):
	abcd = 0 #abcd
	short = 1 #short
	char = 2 #char
	doc = 3 #doc
	meta = 4 #abcd + char
	all = 5 #abcd + char + short
	random = 6
	images = 7

	abcd_o4 = 8
	char_o4 = 9
	doc_o4 = 10
	all_o4 = 11
	meta_o4 = 12

	skingpt4_abcd = 13 # skingpt4_abcd
	skingpt4_char = 14 # skingpt4_char
	skingpt4_doc = 15 # skingpt4_doc
	skingpt4_p1 = 16 # skingpt4_p1
	skingpt4_p2 = 17 # skingpt4_p2
	skingpt4_meta = 18 # skingpt4_abcd + skingpt4_char
	skingpt4_all = 19 # skingpt4_abcd + skingpt4_char + short
	skingpt4_p1_all = 20 # skingpt4_p1 + short

	dermlip_abcd = 21 # dermlip_abcd
	dermlip_char = 22 # dermlip_char
	dermlip_doc = 23 # dermlip_doc
	dermlip_p1 = 24 # dermlip_p1
	dermlip_p2 = 25 # dermlip_p2
	dermlip_meta = 26 # dermlip_abcd + dermlip_char
	dermlip_all = 27 # dermlip_abcd + dermlip_char + short
	dermlip_p1_all = 28 # dermlip_p1 + short

	medgemma_abcd = 29 # medgemma_abcd
	medgemma_char = 30 # medgemma_char
	medgemma_doc = 31 # medgemma_doc
	medgemma_meta = 32 # medgemma_abcd + medgemma_char
	medgemma_all = 33 # medgemma_abcd + medgemma_char + short

	whole = 34 # medgemma_abcd + medgemma_char + dermlip_p1 + skingpt4_p1 + abcd + char
	whole_all = 35 # medgemma_abcd + medgemma_char + dermlip_p1 + skingpt4_p1 + abcd + char + shorts

class COMPONENTS(Enum):
	images = 0
	reports = 1
	keywords = 2

class REPORTS(Enum):
	abcd = 0 #abcd
	short = 1 #short
	char = 2 #char
	doc = 3 #doc
	meta = 4 #abcd + char
	all = 5 #abcd + char + short
	random = 6
	images = 7

	abcd_o4 = 8
	char_o4 = 9
	doc_o4 = 10
	all_o4 = 11
	meta_o4 = 12

	skingpt4_abcd = 13 # skingpt4_abcd
	skingpt4_char = 14 # skingpt4_char
	skingpt4_doc = 15 # skingpt4_doc
	skingpt4_p1 = 16 # skingpt4_p1
	skingpt4_p2 = 17 # skingpt4_p2
	skingpt4_meta = 18 # skingpt4_abcd + skingpt4_char
	skingpt4_all = 19 # skingpt4_abcd + skingpt4_char + short
	skingpt4_p1_all = 20 # skingpt4_p1 + short

	dermlip_abcd = 21 # dermlip_abcd
	dermlip_char = 22 # dermlip_char
	dermlip_doc = 23 # dermlip_doc
	dermlip_p1 = 24 # dermlip_p1
	dermlip_p2 = 25 # dermlip_p2
	dermlip_meta = 26 # dermlip_abcd + dermlip_char
	dermlip_all = 27 # dermlip_abcd + dermlip_char + short
	dermlip_p1_all = 28 # dermlip_p1 + short

	medgemma_abcd = 29 # medgemma_abcd
	medgemma_char = 30 # medgemma_char
	medgemma_doc = 31 # medgemma_doc
	medgemma_meta = 32 # medgemma_abcd + medgemma_char
	medgemma_all = 33 # medgemma_abcd + medgemma_char + short

	whole = 34 # medgemma_abcd + medgemma_char + dermlip_p1 + skingpt4_p1 + abcd + char
	whole_all = 35 # medgemma_abcd + medgemma_char + dermlip_p1 + skingpt4_p1 + abcd + char + shorts

class SD_setup(Enum):
	label = 0
	note = 1
	note_keyword = 2

class CONCEPTS(Enum):
	classes = 0
	subclasses = 1
	classes_matching = 2
	subclasses_matching = 3

class PARTITION(Enum):
	internal = 0
	external = 1
	dermoscopic = 2
	clinical = 3
	whole = 4

if __name__ == "__main__":
	pass