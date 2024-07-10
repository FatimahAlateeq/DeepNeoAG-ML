#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
from tape import ProteinBertModel, TAPETokenizer

from tqdm import tqdm
import numpy as np
import glob
import argparse

# In[2]:
parser = argparse.ArgumentParser()
parser.add_argument("-in","--path_input", type=str, help="the path of input fasta file")
parser.add_argument("-out","--path_output", type=str, help="the path of output esm file")


def main(input_floder, out_folder, miss_txt):
    input_floder = glob.glob(input_floder + "/*")
    out_folder = out_folder
    miss_txt = miss_txt
    
    model = ProteinBertModel.from_pretrained('bert-base')
    tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model
    
    for path in tqdm(input_floder, desc="Processing", unit="file"):
        with open(path) as f:
            fasta = f.readlines()
        title = fasta[0][1:].strip()
        sequence = fasta[1].strip()
        out_path = os.path.join(out_folder, title)

        try:
            token_ids = torch.tensor([tokenizer.encode(sequence)])
            output = model(token_ids)
            sequence_output = output[0][:, 1:-1,:].cpu().detach().numpy()
        except Exception as e:
            log_mode = 'a' if os.path.exists(miss_txt) else 'w'
            with open(miss_txt, log_mode) as tape_miss:
                tape_miss.write(title + ".fasta\n")
            continue
            
        np.save(out_path + ".npy", sequence_output)
        # pooled_output = output[1]
        # print(sequence_output)
        # print(sequence_output.shape)
        # print(pooled_output.shape)
        # print(title + ".fasta")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.path_input, args.path_output,"tape_miss")
        
