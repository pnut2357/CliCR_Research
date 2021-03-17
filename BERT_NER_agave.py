import os
import json
import numpy as np
from transformers import BertModel,BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import copy
from create_processed_data import *

import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam

print('Imported all libraries')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print('Loaded device:',device)

print(device)

print(torch.cuda.get_device_name(0))

DATASET_SIZE = 53914
MAX_LEN = 424
SEED = 520
bs = 1
MAX_N_SPLITS = 32


print('Loading training data')
train_data_processor = DataProcessor('train',size=None,tokenizer=None)
tags_vals = train_data_processor.get_tags_vals()


#Make sure no paragraph has only 'O' tags
paragraphs_tags =  train_data_processor.paragraph_tags
inp = None
for i, paragraph_tags in enumerate(paragraphs_tags):
    assert not all(p == 'O' for p in paragraph_tags)


print('Loading dev data')
valid_data_processor = DataProcessor('dev',size=None,tokenizer=train_data_processor.tokenizer)


torch.cuda.empty_cache()

train_dataloader = DataLoader(train_data_processor, shuffle=True, batch_size=bs, collate_fn=train_data_processor.my_collate)

valid_dataloader = DataLoader(valid_data_processor, shuffle=True, batch_size=bs, collate_fn=train_data_processor.my_collate)
print('Created train and dev data loaders')



HIDDEN_SIZE = 768
KERNEL_SIZE = 5
PADDING = 2
import torch.nn as nn
model = BertModel.from_pretrained("bert-base-uncased",num_labels=len(tags_vals))
class myNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Conv1d(HIDDEN_SIZE,len(tags_vals),kernel_size = KERNEL_SIZE, padding = PADDING)
        #perform softmax on classes dimension
        self.s = nn.Softmax(dim=1)

    def forward(self,last_hidden_state):
        last_hidden_state = last_hidden_state.transpose(1,2)
        logits = self.l(last_hidden_state)
        probs = self.s(logits)
        return probs

mynet = myNetwork()



model.cuda();
mynet.cuda();
print('Pushed BERT to GPU')

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters()) + list(mynet.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=1e-5)


from seqeval.metrics import f1_score
from seqeval.metrics import classification_report
from torch.nn import CrossEntropyLoss

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


print('Initiating training')
closs = CrossEntropyLoss(ignore_index=-100)
epochs = 4
max_grad_norm = 1.0
F_scores = np.zeros(epochs,float)
patience = 6
accumulation_steps = 16
for e in range(epochs):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    predictions , true_labels = [], []
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        for split in range(len(batch[0])):
            b_input_ids = batch[0].pop(0).to(device)
            b_input_mask = batch[1].pop(0).to(device)
            b_labels = batch[2].pop(0).to(device)
            b_ttids = batch[3].pop(0).to(device)
            # forward pass
            # loss = model(b_input_ids, token_type_ids=b_ttids, attention_mask=b_input_mask, labels=b_labels)
            if split == 0:
                last_hidden_state = model(b_input_ids, token_type_ids=b_ttids, attention_mask=b_input_mask)[0]
            else:
                last_hidden_state = torch.cat([last_hidden_state,model(b_input_ids, token_type_ids=b_ttids, attention_mask=b_input_mask)[0]], dim = 1)
                b_labels = torch.cat([prev_labels,b_labels],dim=1)

            prev_labels = copy.copy(b_labels)

        b_logits = mynet(last_hidden_state)
        b_logits = b_logits.transpose(1,2)

        logits = b_logits.view(-1,len(tags_vals))
        labels = b_labels.view(-1)

        loss = closs(logits, labels)
        loss = loss/accumulation_steps
        # backward pass
        loss.backward()
        b_logits = b_logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(b_logits, axis=2)])
        true_labels.append(label_ids)
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        if (step+1)%accumulation_steps == 0:
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            torch.nn.utils.clip_grad_norm_(parameters=mynet.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            model.zero_grad()
            mynet.zero_grad()
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags_vals[l_ii] if l_ii!=-100 else '[PAD]' for l in true_labels for l_i in l for l_ii in l_i]
    c_f1 = f1_score(valid_tags, pred_tags)
    rep = classification_report(valid_tags, pred_tags)
    print('Training classification report:',rep)
    print("Training F1-Score: {}".format(c_f1))



    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:

        with torch.no_grad():
            # add batch to gpu
            for split in range(len(batch[0])):
                b_input_ids = batch[0].pop(0).to(device)
                b_input_mask = batch[1].pop(0).to(device)
                b_labels = batch[2].pop(0).to(device)
                b_ttids = batch[3].pop(0).to(device)
               # forward pass
                # loss = model(b_input_ids, token_type_ids=b_ttids, attention_mask=b_input_mask, labels=b_labels)
                if split == 0:
                    last_hidden_state = model(b_input_ids, token_type_ids=b_ttids, attention_mask=b_input_mask)[0]
                else:
                    last_hidden_state = torch.cat([last_hidden_state,model(b_input_ids, token_type_ids=b_ttids, attention_mask=b_input_mask)[0]], dim = 1)
                    b_labels = torch.cat([prev_labels,b_labels],dim=1)

                prev_labels = copy.copy(b_labels)

            b_logits = mynet(last_hidden_state)
            b_logits = b_logits.transpose(1,2)

            logits = b_logits.view(-1,len(tags_vals))
            labels = b_labels.view(-1)

            tmp_eval_loss = closs(logits, labels)


        b_logits = b_logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(b_logits, axis=2)])
        true_labels.append(label_ids)


        tmp_eval_accuracy = flat_accuracy(b_logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss/nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags_vals[l_ii] if l_ii!=-100 else '[PAD]' for l in true_labels for l_i in l for l_ii in l_i]
    c_f1 = f1_score(valid_tags, pred_tags)
    rep = classification_report(valid_tags, pred_tags)
    print(rep)
    print("F1-Score: {}".format(c_f1))
    #k epochs no improvement
    F_scores[e] = c_f1
    prev_Fs = np.arange(e-patience,e)
    prev_indices = prev_Fs[prev_Fs>=0]
    if all(F_scores[e] - F_scores[prev_indices] < 0 ) and e > patience:
        print('Done training')
        break
    else:
        print('Still training')
        #save checkpoint
        #Nobody!

