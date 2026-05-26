import sys, getopt
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
import torch

import random

def clean_sentence(input_txt, prob = 0.5):

    sentences = input_txt.split('\n')

    new_sentence = []
    for i in range(len(sentences)):

        subsentence = sentences[i]

        if ('1)' in subsentence or '2)' in subsentence or '3)' in subsentence or '4)' in subsentence or subsentence == ''):
            prob_pre = np.random.rand(1)[0]
            if (prob_pre > prob):
                new_sentence.append(subsentence)
        else:
            new_sentence.append(subsentence)

    augmented_sentence = '\n'.join(new_sentence)

    return augmented_sentence

def load_txt(ID_txt):
     
    with open(ID_txt, 'r', encoding='utf-8', errors='ignore') as file:
        #with open(ID_txt, 'r') as file:
        input_txt = file.read()
        file.close()
                                
    return input_txt


def generate_text(decoder, features, tokenizer, max_len=256):
    decoder.eval()
    B = features.size(0)
    BOS_ID = tokenizer.cls_token_id
    EOS_ID = tokenizer.sep_token_id
    PAD_ID = tokenizer.pad_token_id

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # Pre-allocate output tensor
            generated = torch.full((B, max_len), PAD_ID, dtype=torch.long, device=features.device)
            generated[:, 0] = BOS_ID

            finished = torch.zeros(B, dtype=torch.bool, device=features.device)

            step = 1
            while step < max_len and not finished.all():
                logits = decoder(features, generated[:, :step])  # (B, step, vocab_size)
                next_token_logits = logits[:, -1, :]             # (B, vocab_size)
                next_token = torch.argmax(next_token_logits, dim=-1)  # (B,)

                generated[:, step] = next_token

                finished |= (next_token == EOS_ID)
                step += 1
                
            

    return tokenizer.batch_decode(generated, skip_special_tokens=True)

if __name__ == "__main__":
	pass