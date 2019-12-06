#!/usr/bin/env python3

import os
import csv
import torch
import transformers as tr
import numpy as np
import gzip
from tqdm import tqdm

modelname = 'distilgpt2'
#modelname = 'gpt2-medium'
#modelname = 'gpt2'

tokenizer = None
model = None

def get_embedding_for_string(txt):
    global tokenizer
    global model
    if tokenizer is None:
        tokenizer = tr.GPT2Tokenizer.from_pretrained(modelname)
        model = tr.GPT2DoubleHeadsModel.from_pretrained(modelname)

    # Put the input inside a template to prime the network
    txt = "What is a " + txt + "?\n"

    input_ids = torch.tensor(tokenizer.encode(txt)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    # Get embedding based on final internal state of the GPT2 model
    emb = outputs[0][:,-1,:].detach().numpy()
    return emb

def process_all_embeddings():
    if os.path.isfile('emoji_names.cache'):
        cacheTime = os.path.getmtime('emoji_names.cache')
        csvTime = os.path.getmtime('emoji_names.csv')
        pyTime = os.path.getmtime(__file__)
        if (cacheTime > csvTime) and (cacheTime > pyTime):
            with gzip.open('emoji_names.cache', 'rb') as f:
                codeStrList = f.readline().decode('utf-8').split(';')
                embedList = np.load(f)
                embeds = {}
                for i in range(len(codeStrList)):
                    embeds[codeStrList[i]] = embedList[i, :]
                return embeds
                
    
    embeds = {}
    with open('emoji_names.csv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        lines = list(reader)
    codeStrList = []
    embedList = []
    for line in tqdm(lines):
        if len(line) == 2:
            codeStrList.append(line[0])
            embedList.append(get_embedding_for_string(line[1]))
    embedList = np.concatenate(embedList)
    with gzip.open('emoji_names.cache', 'wb') as f:
        f.write(";".join(codeStrList).encode('utf-8') + b'\n')
        np.save(f, embedList)
    embeds = {}
    for i in range(len(codeStrList)):
        embeds[codeStrList[i]] = embedList[i, :]
    return embeds

embedsTable = None
def get_embedding_for_emoji(codeString):
    global embedsTable
    if embedsTable is None:
        embedsTable = process_all_embeddings()
    return embedsTable.get(codeString, None)

if __name__ == '__main__':
    # Test example
    embed = get_embedding_for_emoji('u1f192')
    print(embed)
