import sys, getopt
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
import torch

import random

def load_txt(ID_txt):
     
    with open(ID_txt, 'r', encoding='utf-8', errors='ignore') as file:
        #with open(ID_txt, 'r') as file:
        input_txt = file.read()
        file.close()
                                
    return input_txt

if __name__ == "__main__":
	pass