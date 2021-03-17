import os
import json
import numpy as np
from transformers import BertModel,BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from my_process_json import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
import copy
from torch.utils import data

#CONSTANTS
DATASET_SIZE = 53914
MAX_LEN = 500
SEED = 520
batch_size = 16
model_version = 'biobert_v1.1_pubmed'



class DataProcessor(data.Dataset):
    
    def __init__(self,filename = 'train', size = None, tokenizer=None):
        data_reader = MyDataReader('/home/akobtan/NLP_Project/Project/clicr_dataset/'+filename+'1.0.json',bs=size)
        mydata = data_reader.send_batches()
        self.dataset_size = data_reader.get_data_size()
        self.paragraphs = [e['p'] for e in mydata].copy()
        self.paragraph_tags = [e['p_tags'] for e in mydata].copy()
        self.queries = [e['q'] for e in mydata].copy()
        self.query_tags = [e['q_tags'] for e in mydata].copy()

        assert all(len(self.paragraphs) == len(y) for y in [self.paragraph_tags, self.queries, self.query_tags])

        #creating embeddings
        self.tags_vals = ['B-ans','I-ans','O']
        self.tag2idx = {t: i for i, t in enumerate(self.tags_vals)}
        self.tag2idx['[PAD]'] = -100
        self.tag2idx['[SEP]'] = -100
        self.tag2idx['[CLS]'] = -100
        self.maxN = 0
        
        if tokenizer == None:
            #self.tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        else:
            self.tokenizer = tokenizer

    def __getitem__(self, index):
        return [self.paragraphs[index], self.paragraph_tags[index], self.queries[index], self.query_tags[index]]

    def __len__(self):
        return len(self.paragraphs)
        
        
    def create_tokenizedtexts_and_labels(self):
        split_size = 200
        p_tags = copy.deepcopy(self.paragraph_tags)
        q_tags = copy.deepcopy(self.query_tags)
        paragraphs = copy.deepcopy(self.paragraphs)
        queries = copy.deepcopy(self.queries)
        tokenized_texts = []
        labels = []
        for i,paragraph in enumerate(paragraphs):
            texts = []
            paragraph_tag = copy.deepcopy(p_tags[i])
            #number of splits for current paragraph
            n_splits = int(len(paragraph.split())/split_size)
            if n_splits > self.maxN:
                self.maxN = n_splits
            #tokenize split paragraph text appended by full query at the end with <sep> token between paragraph and query
            queries[i] = queries[i].replace("@placeholder",'[MASK]')
            texts += [s for s in ['[CLS] ' + ' '.join(paragraph.split()[(split_size*split):(split_size*(split+1))]) + ' [SEP] ' + queries[i] + ' [SEP]' for split in range(n_splits)]]
            tokenized_texts += [self.tokenizer.tokenize(s) for s in ['[CLS] ' + ' '.join(paragraph.split()[(split_size*split):(split_size*(split+1))]) + ' [SEP] ' + queries[i] + ' [SEP]' for split in range(n_splits)]]
            
            #tokenize remainder of splits
            if int(len(paragraph.split()) % split_size) > 0:
                texts += ['[CLS] ' + ' '.join(paragraph.split()[(n_splits*split_size):]) + ' [SEP] ']
                tokenized_texts += [self.tokenizer.tokenize('[CLS] ' + ' '.join(paragraph.split()[(n_splits*split_size):]) + ' [SEP] ' + queries[i] + ' [SEP]')]

            #create labels to be used for tagging from the text
            for split_index, sent in enumerate(texts):
                l = []
                for word_index,word in enumerate(sent.split()):
                    if '[SEP]' in word:
                        l += ['[SEP]']
                        break
                    if '[CLS]' in word:
                        l += ['[CLS]']
                        continue
                    if not paragraph_tag:
                        print(sent,'\n')
                        print(word)
                        print(split_index)
                    lab = paragraph_tag.pop(0)
                    word_list = self.tokenizer.tokenize(word)
                    for w in word_list:
                        l += [lab]
                query_tags = copy.deepcopy(q_tags[i])
                for word in queries[i].split():
                    lab = query_tags.pop(0)
                    word_list = self.tokenizer.tokenize(word)
                    for w in word_list:
                        l += [lab]
                l += ['[SEP]']
                labels += [l]
        self.labels = labels
        self.tokenized_texts = tokenized_texts
    
    
    
    def create_input_ids(self):
        self.input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in self.tokenized_texts],maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    
    
    def create_token_type_ids(self):
        token_type_ids = []
        for ipid in self.input_ids:
            type_id = 0
            token_type_id = []
            for myid in ipid:
                token_type_id.append(type_id)
                if myid == 102: 
                    if type_id%2==0:
                        type_id+= 1
                    else:
                        type_id=0
            token_type_ids.append(token_type_id)
        self.token_type_ids = token_type_ids
    

    
    def create_tags(self):
        self.tags = pad_sequences([[self.tag2idx.get(l) for l in lab] for lab in self.labels],
                     maxlen=MAX_LEN, value=self.tag2idx["[PAD]"], padding="post",
                     dtype="long", truncating="post")
        
    
    def create_attention_masks(self):
        self.attention_masks = [[float(i>0) for i in ii] for ii in self.input_ids]

        
    def get_tags_vals(self):
        return self.tags_vals
        
        
    def get_processed_data(self):
        self.create_tokenizedtexts_and_labels()
        self.create_input_ids()
        self.create_token_type_ids()
        self.create_tags()
        self.create_attention_masks()
        
        return self.input_ids, self.attention_masks, self.tags, self.token_type_ids, self.tokenizer
    
    
    def my_collate(self,batch):
        split_size = 200
        paragraph, paragraph_tags, query, query_tags = batch[0]
        paragraph_tags = copy.deepcopy(paragraph_tags)
        paragraph = copy.deepcopy(paragraph)
        query_tags = copy.deepcopy(query_tags)
        query = copy.deepcopy(query)
        
        
        #number of splits for current paragraph
        n_splits = int(len(paragraph.split())/split_size)
        query = query.replace("@placeholder",'[MASK]')
        
        
        texts = [s for s in ['[CLS] ' + ' '.join(paragraph.split()[(split_size*split):(split_size*(split+1))]) + ' [SEP] ' for split in range(n_splits)]]
        #tokenized_texts = [self.tokenizer.tokenize(s) for s in ['[CLS] ' + ' '.join(paragraph.split()[(split_size*split):(split_size*(split+1))]) + ' [SEP] ' + query + ' [SEP]' for split in range(n_splits)]]

        #tokenize remainder of splits
        if int(len(paragraph.split()) % split_size) > 0:
            texts += ['[CLS] ' + ' '.join(paragraph.split()[(n_splits*split_size):]) + ' [SEP] ']
            #tokenized_texts += [self.tokenizer.tokenize('[CLS] ' + ' '.join(paragraph.split()[(n_splits*split_size):]) + ' [SEP] ' + query + ' [SEP]')]

            
        """       
        p_texts = [self.tokenizer.tokenize(s) for s in ['[CLS] ' + ' '.join(paragraph.split()[(split_size*split):(split_size*(split+1))]) + ' [SEP] ' for split in range(n_splits)]]
        q_text = self.tokenizer.tokenize(query + ' [SEP]')
        #tokenize remainder of splits
        if int(len(paragraph.split()) % split_size) > 0 :
            p_texts += [self.tokenizer.tokenize('[CLS] ' + ' '.join(paragraph.split()[(split_size*n_splits):]) + ' [SEP]')]
        """
        
        labels = []
        p_texts = []
        q_texts = []
        #create labels to be used for tagging from the text
        for split_index, sent in enumerate(texts):
            l = []
            p_text = []
            for word_index,word in enumerate(sent.split()):
                if '[SEP]' in word:
                    l += ['[SEP]']
                    p_text += ['[SEP]']
                    break
                if '[CLS]' in word:
                    l += ['[CLS]']
                    p_text += ['[CLS]']
                    continue
                if not paragraph_tags:
                    print(sent,'\n')
                    print(word)
                    print(split_index)
                lab = paragraph_tags.pop(0)
                word_list = self.tokenizer.tokenize(word)
                for w in word_list:
                    l += [lab]
                    p_text += [w]
            q_tags = copy.deepcopy(query_tags)
            q_text = []
            for word in query.split():
                lab = q_tags.pop(0)
                word_list = self.tokenizer.tokenize(word)
                for w in word_list:
                    l += [lab]
                    q_text += [w]
            l += ['[SEP]']
            q_text += ['[SEP]']
            labels += [l]
            p_texts += [p_text]
            q_texts += [q_text]
            
        if paragraph_tags != []:
            print(paragraph)
            print(query)
            print(paragraph_tags)
            print(query_tags)
           
        #encode function here
        tr_inputs = []
        tr_ttids = []
        tr_masks = []
        tr_tags = []
        for i, p_text in enumerate(p_texts):
            encodings = self.tokenizer.encode_plus(text = p_text, text_pair = q_texts[i], add_special_tokens = False, max_length = None, pad_to_max_length = False, is_pretokenized = True, return_tensors = 'pt', return_token_type_ids = True, return_attention_mask = True)
            tr_inputs += [encodings['input_ids']]
            tr_ttids += [encodings['token_type_ids']]
            tr_masks += [encodings['attention_mask']]
            tr_tags += [torch.tensor([[self.tag2idx.get(l) for l in labels[i]]])]
            if len(encodings['input_ids'][0]) != len(tr_tags[-1][0]):
                print(encodings['input_ids'][0])
                print(len(encodings['input_ids'][0]))
                print(tr_tags[-1][0])
                print(len(tr_tags[-1][0]))
                print('ERROR!!!')
                exit()
        
        return tuple(t for t in [tr_inputs, tr_masks, tr_tags, tr_ttids])
    
    
