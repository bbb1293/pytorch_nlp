#!/usr/bin/env python
# coding: utf-8

# # SST-2 Binary Text Classification with BERT Model (ref: [Transformers](https://huggingface.co/docs/transformers/training))

# ## Common Imports

# In[ ]:


import torch
import torch.nn as nn
import os
import argparse

parser = argparse.ArgumentParser(description="Set some arguments for training")
parser.add_argument("--gpu_num", type=int, help="gpu num you want to use", default=0)
parser.add_argument("--num_train_data", type=int, help="the number of the training data", default=32)
parser.add_argument("--num_seed", type=int, help="the number of the seeds", default=10)
parser.add_argument("--num_epochs", type=int, help="the number of the epochs", default=300)
parser.add_argument("--backt", action="store_true", help="augment training data by backtranslation")
parser.add_argument("--eda", action="store_true", help="augment training data by EDA")
parser.add_argument("--masked_lm", action="store_true", help="augment training data by masked language model")

args = parser.parse_args()


# ## Load Dataset

# In[ ]:


from datasets import load_dataset
from torch.utils.data import DataLoader 

def load_train_test_dataset(seed, num_train_data):
    dataset = load_dataset("sst2")

    # idx, sentence, label
    train_dataset = dataset["train"].shuffle(seed=seed).select(range(num_train_data))
    test_dataset = dataset["validation"]
    
    return (train_dataset, test_dataset)

# train_dataset, test_dataset = load_dataset(seed=SEED)


# ## Data Augmentation by Backtranslation (ref: [En to Fr](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr?text=My+name+is+Sarah+and+I+live+in+London), [Fr to En](https://huggingface.co/Helsinki-NLP/opus-mt-fr-en?text=Mon+nom+est+Wolfgang+et+je+vis+%C3%A0+Berlin))

# In[ ]:


from transformers import pipeline

if args.backt:
    en_to_others = [pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr"), pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")]
    others_to_en = [pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en"), pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")]


# In[ ]:


def aug_by_backt(train_dataset, en_to_others, others_to_en):
    
    sentences = [train_data["sentence"] for train_data in train_dataset]
    labels = [train_data["label"] for train_data in train_dataset]
    sentences_len = len(sentences)
    
    aug_by_backt_train_dataset = train_dataset
    for translator_idx in range(len(en_to_others)):
        tmp_sentence = [tmp_data['translation_text'] for tmp_data in en_to_others[translator_idx](sentences)]
        aug_sentence = [tmp_data['translation_text'] for tmp_data in others_to_en[translator_idx](tmp_sentence)]
        
        for sen_idx in range(sentences_len):
            aug_data = {'sentence': aug_sentence[sen_idx], 'label': labels[sen_idx]}
            aug_by_backt_train_dataset = aug_by_backt_train_dataset.add_item(aug_data)
            
    return aug_by_backt_train_dataset

# train_dataset = aug_by_backt(train_dataset, en_to_others, others_to_en)
# aug_train_dataset = aug_by_backt(train_dataset, en_to_others, others_to_en)


# ## Data Augmentation by EDA (ref: [EDA](https://github.com/jasonwei20/eda_nlp))

# In[ ]:


# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

from eda import *

# For the first time to load wordnet
'''
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
'''

# Generate more data with EDA
def aug_by_eda(train_dataset, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1, num_aug=3):
    sentences = [train_data["sentence"] for train_data in train_dataset]
    labels = [train_data["label"] for train_data in train_dataset]
    
    aug_by_eda_train_dataset = train_dataset
    aug_sentences = [eda(sentence) for sentence in sentences]
    
    for i in range(len(labels)):
        for aug_sentence in aug_sentences[i]:
            aug_data = {'sentence': aug_sentence, 'label': labels[i]}
            aug_by_eda_train_dataset = aug_by_eda_train_dataset.add_item(aug_data)
            
    return aug_by_eda_train_dataset

# train_dataset = aug_by_eda(train_dataset)
# aug_train_dataset = aug_by_eda(train_dataset)


# ## Data Augmentation by masked language model (ref: [bert_base_uncased](https://huggingface.co/bert-base-uncased?text=what+the+%5BMASK%5D+good))

# In[ ]:


from transformers import pipeline

if args.masked_lm:
    masked_lm = pipeline('fill-mask', model='bert-base-uncased')
    # masked_lm("Hello I'm a [MASK] model.")


# In[ ]:


import numpy as np

def aug_by_masked_lm(train_dataset, seed, masked_lm, aug_num=3):
    
    np.random.seed(seed)
    
    sentences = [train_data["sentence"] for train_data in train_dataset]
    labels = [train_data["label"] for train_data in train_dataset]
    sentences_len = len(sentences)
    
    aug_by_masked_lm_train_dataset = train_dataset
    for idx in range(sentences_len):
        splited_sentences = sentences[idx].split()
        
        for i in range(aug_num):
            target_idx = np.random.choice(len(splited_sentences))
            original_word = splited_sentences[target_idx]
            
            splited_sentences[target_idx] = "[MASK]"
            
            if labels[idx] == 1:
                splited_sentences.append("positive")
            else:
                splited_sentences.append("negative")
            
            converted_sentence = " ".join(splited_sentences)
            
            converted_word = masked_lm(converted_sentence)[0]["token_str"]
            splited_sentences[target_idx] = converted_word
            
            aug_data = {"sentence": " ".join(splited_sentences[:-1]), "label": labels[idx]}
            aug_by_masked_lm_train_dataset.add_item(aug_data)
            
            splited_sentences[target_idx] = original_word
        
    return aug_by_masked_lm_train_dataset

# train_dataset, test_dataset = load_train_test_dataset(0, 32)
# aug_by_masked_lm(train_dataset, seed=0, masked_lm=masked_lm)


# ## Transform Dataset

# In[ ]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Tokenize help method
def apply_transform(x):
    return tokenizer(x["sentence"], padding="max_length", truncation=True)

def transform_datasets(train_dataset, test_dataset, seed):
    tokenized_train_dataset = train_dataset.map(apply_transform, batched=True)
    tokenized_test_dataset = test_dataset.map(apply_transform, batched=True)
    
    # To fit the model's input
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(['sentence', 'idx'])
    tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(['sentence', 'idx'])
    tokenized_test_dataset = tokenized_test_dataset.rename_column("label", "labels")

    # labels, input_ids, token_type_idx, attention_mask
    # Convert format to torch
    tokenized_train_dataset.set_format("torch")
    tokenized_test_dataset.set_format("torch")
    
    tokenized_train_dataset = tokenized_train_dataset.train_test_split(test_size=0.5, seed=seed)
    train_dataloader = DataLoader(tokenized_train_dataset["train"], batch_size=8, shuffle=None)
    val_dataloader = DataLoader(tokenized_train_dataset["test"], batch_size=8, shuffle=None)
    test_dataloader = DataLoader(tokenized_test_dataset, batch_size=8, shuffle=None)
    
    return (train_dataloader, val_dataloader, test_dataloader)

# train_dataloader, val_dataloader, test_dataloader = transform_datasets()


# ## Train

# In[ ]:


from tqdm.auto import tqdm

def train_model(model, train_dataloader, val_dataloader, optimizer, lr_scheduler, num_epochs=300):
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))

    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf') 
    best_state_dict = {}

    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0

        for train_batch, val_batch in zip(train_dataloader, val_dataloader):
            train_batch = {k: v.to(DEVICE) for k, v in train_batch.items()}
            val_batch = {k: v.to(DEVICE) for k, v in val_batch.items()}
            outputs = model(**train_batch)
            loss = outputs.loss
            loss.backward()

            train_loss += loss.item()
            with torch.no_grad():
                cur_val_loss = model(**val_batch).loss.item()
                val_loss += cur_val_loss 
                
                if cur_val_loss < best_val_loss:
                    best_val_loss = cur_val_loss
                    best_state_dict = model.state_dict()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
    model.load_state_dict(best_state_dict)
    
    return (train_losses, val_losses)
        
# train_losses, val_losses = train_model()


# ## Evaluate

# In[ ]:


import evaluate

def evaluate_model(model, test_dataloader):
    metric = evaluate.load("accuracy")
    
    model.eval()

    progress_bar = tqdm(range(len(test_dataloader)))

    for test_batch in test_dataloader:
        test_batch = {k: v.to(DEVICE) for k, v in test_batch.items()}
        with torch.no_grad():
            outputs = model(**test_batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=test_batch["labels"])

        progress_bar.update(1)
        
    result = metric.compute()
    
    return result["accuracy"]


# In[ ]:


from transformers import AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from prettytable import PrettyTable

learning_rate = 1e-5

GPU_NUM = args.gpu_num
DEVICE = torch.device(f"cuda:{GPU_NUM}") if torch.cuda.is_available() else torch.device("cpu")
NUM_TRAIN_DATA = args.num_train_data
NUM_SEED = args.num_seed
NUM_EPOCHS = args.num_epochs

# accuracy = 0.0
for seed in range(NUM_SEED):
    train_dataset, test_dataset = load_train_test_dataset(seed=seed, num_train_data=NUM_TRAIN_DATA)

    if args.backt:
        train_dataset = aug_by_backt(train_dataset=train_dataset, en_to_others=en_to_others, others_to_en=others_to_en)
        
    if args.eda:
        train_dataset = aug_by_eda(train_dataset=train_dataset)
        
    if args.masked_lm:
        train_dataset = aug_by_masked_lm(train_dataset=train_dataset, seed=seed, masked_lm=masked_lm)
    
    train_dataloader, val_dataloader, test_dataloader = transform_datasets(train_dataset=train_dataset,
                                                                           test_dataset=test_dataset, 
                                                                           seed=seed)
    
    for train_data in train_dataloader:
        print(hash(train_data))
    
    '''
    # model preparation
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
    model.to(DEVICE)
    
    # training method
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=NUM_EPOCHS*len(train_dataloader)
    )
    
    train_model(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                optimizer=optimizer, lr_scheduler=lr_scheduler, num_epochs=NUM_EPOCHS)
    
    cur_accuracy = evaluate_model(model=model, test_dataloader=test_dataloader)
    accuracy += cur_accuracy
    
    print(f"Seed {seed} accuracy: {cur_accuracy}")
    
table_content = []    
table_content.append("O" if args.backt else "X")
table_content.append("O" if args.eda else "X")
table_content.append("O" if args.masked_lm else "X")
table_content.append(accuracy / NUM_SEED)

my_table = PrettyTable(["Backtranslation", "EDA", "Masked language model", "Accuracy"])
my_table.add_row(table_content)

print(my_table)
'''
