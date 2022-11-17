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
parser.add_argument("--dataset_name", type=str, help="the name of dataset you want to use (ex. sst2, amazon_polarity, imdb)", default='sst2')
parser.add_argument("--gpu_num", type=int, help="gpu num you want to use", default=0)
parser.add_argument("--num_train_data", type=int, help="the number of the training data", default=32)
parser.add_argument("--num_seed", type=int, help="the number of the seeds", default=10)
parser.add_argument("--num_epochs", type=int, help="the number of the epochs", default=300)
parser.add_argument("--num_aug", type=int, help="the number of augmentation data", default=3)
parser.add_argument("--backt", action="store_true", help="augment training data by backtranslation")
parser.add_argument("--eda", action="store_true", help="augment training data by EDA")
parser.add_argument("--masked_lm", action="store_true", help="augment training data by masked language model")
parser.add_argument("--afinn", action="store_true", help="augment training data by AFIIN using unlabeled data")

args = parser.parse_args()


# ## Preprocess Dataset

# In[ ]:


def preprocess_sst2(dataset):
    dataset = dataset.remove_columns("idx")
    return dataset
    
def preprocess_imdb(dataset):
    dataset = dataset.rename_column("text", "sentence")
    return dataset

def merge_sentence(data):
    data["sentence"] = data["title"] + " " + data["content"]
    
def preprocess_amazon_polarity(dataset):
    dataset = dataset.map(merge_sentence)
    dataset = dataset.remove_columns(["title", "content"])
    return dataset
    
preprocess_dict = {"sst2": preprocess_sst2, "amazon_polarity": preprocess_amazon_polarity, "imdb": preprocess_imdb}


# ## Load Dataset

# In[ ]:


from datasets import load_dataset
from torch.utils.data import DataLoader 

def load_train_test_dataset(seed, num_train_data, dataset_name):
    dataset = load_dataset(dataset_name)

    # idx, sentence, label
    pre_train_dataset = dataset["train"].shuffle(seed=seed)
    train_dataset = pre_train_dataset.select(range(num_train_data))
    train_rest_dataset = pre_train_dataset.select(range(num_train_data, len(pre_train_dataset)))
    test_dataset = dataset["validation"] if "validation" in dataset else dataset["test"]
    
    pre_train_dataset = preprocess_dict[dataset_name](pre_train_dataset)
    train_dataset = preprocess_dict[dataset_name](train_dataset)
    test_dataset = preprocess_dict[dataset_name](test_dataset)
    
    return (train_dataset, train_rest_dataset, test_dataset)

# train_dataset, train_rest_dataset, test_dataset = load_train_test_dataset(seed=0, num_train_data=32)


# ## Data Augmentation by Backtranslation (ref: [En to Fr](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr?text=My+name+is+Sarah+and+I+live+in+London), [Fr to En](https://huggingface.co/Helsinki-NLP/opus-mt-fr-en?text=Mon+nom+est+Wolfgang+et+je+vis+%C3%A0+Berlin))

# In[ ]:


from transformers import pipeline

if args.backt:
    en_to_others = [pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr"), pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")]
    others_to_en = [pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en"), pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")]


# In[ ]:


def aug_by_backt(train_dataset, en_to_others, others_to_en, num_aug):
    
    sentences = [train_data["sentence"] for train_data in train_dataset]
    labels = [train_data["label"] for train_data in train_dataset]
    sentences_len = len(sentences)
    
    aug_by_backt_train_dataset = train_dataset
    for i in range(num_aug):
        tmp_sentences = [tmp_data['translation_text'] for tmp_data in en_to_others[i % len(en_to_others)](sentences)]
        sentences = [tmp_data['translation_text'] for tmp_data in others_to_en[i % len(en_to_others)](tmp_sentences)]
        
        for sen_idx in range(sentences_len):
            aug_data = {'sentence': sentences[sen_idx], 'label': labels[sen_idx]}
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
    aug_sentences = [eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=num_aug) for sentence in sentences]
    
    for i in range(len(labels)):
        for aug_sentence in aug_sentences[i]:
            aug_data = {'sentence': aug_sentence, 'label': labels[i]}
            aug_by_eda_train_dataset = aug_by_eda_train_dataset.add_item(aug_data)
            
    return aug_by_eda_train_dataset

# train_dataset = aug_by_eda(train_dataset)
# aug_train_dataset = aug_by_eda(train_dataset)


# ## Data Augmentation by Masked Language Model (ref: [bert_base_uncased](https://huggingface.co/bert-base-uncased?text=what+the+%5BMASK%5D+good))

# In[ ]:


from transformers import pipeline

if args.masked_lm:
    masked_lm = pipeline('fill-mask', model='bert-base-uncased')
    # masked_lm("Hello I'm a [MASK] model.")


# In[ ]:


import numpy as np

def aug_by_masked_lm(train_dataset, seed, masked_lm, num_aug=3):
    
    np.random.seed(seed)
    
    sentences = [train_data["sentence"] for train_data in train_dataset]
    labels = [train_data["label"] for train_data in train_dataset]
    sentences_len = len(sentences)
    
    aug_by_masked_lm_train_dataset = train_dataset
    for idx in range(sentences_len):
        splited_sentences = sentences[idx].split()
        
        for i in range(num_aug):
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
            aug_by_masked_lm_train_dataset = aug_by_masked_lm_train_dataset.add_item(aug_data)
            
            splited_sentences[target_idx] = original_word
        
    return aug_by_masked_lm_train_dataset

# train_dataset, test_dataset = load_train_test_dataset(0, 32)
# aug_by_masked_lm(train_dataset, seed=0, masked_lm=masked_lm)


# ## Data Augmentation by AFINN using Unlabeled data (ref: [afinn](https://pypi.org/project/afinn/))

# In[ ]:


from afinn import Afinn

if args.afinn:
    afinn = Afinn()


# In[ ]:


from tqdm.auto import tqdm

# Generate more data with AFIIN 
def aug_by_afinn(train_dataset, unlabeled_dataset, afinn, data_num=500):
    aug_sentences = []
    labels = []
    
    data_num = min(data_num, len(unlabeled_dataset))
    progress_bar = tqdm(range(data_num))
    cnt = 0
    
    for unlabeled_data in unlabeled_dataset:
        sentence = unlabeled_data["sentence"]
        
        score = afinn.score(sentence)
        if score == 0:
            continue
            
        cnt += 1
        progress_bar.update(1)
        aug_sentences.append(sentence)
        labels.append(1 if score > 0 else 0)
        
        if cnt == data_num:
            break
    
    aug_by_afinn_train_dataset = train_dataset
    
    for i in range(len(labels)):
        aug_data = {'sentence': aug_sentences[i], 'label': labels[i]}
        aug_by_afinn_train_dataset = aug_by_afinn_train_dataset.add_item(aug_data)
        
    return aug_by_afinn_train_dataset
    
# aug_by_afinn(train_dataset, unlabeled_dataset, afinn)


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
    tokenized_train_dataset = tokenized_train_dataset.remove_columns('sentence')
    tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
    tokenized_test_dataset = tokenized_test_dataset.remove_columns('sentence')
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

DATASET_NAME = args.dataset_name
GPU_NUM = args.gpu_num
DEVICE = torch.device(f"cuda:{GPU_NUM}") if torch.cuda.is_available() else torch.device("cpu")
NUM_TRAIN_DATA = args.num_train_data
NUM_SEED = args.num_seed
NUM_EPOCHS = args.num_epochs
NUM_AUG = args.num_aug

MODEL_FOLDER = "./bert_model/"
if not os.path.exists(MODEL_FOLDER):
    os.mkdir(MODEL_FOLDER)

accuracy = []
acc_accuracy = 0.0

for seed in range(NUM_SEED):
    torch.manual_seed(seed)
    
    train_dataset, train_rest_dataset, test_dataset = load_train_test_dataset(seed=seed, num_train_data=NUM_TRAIN_DATA, dataset_name=DATASET_NAME)
    
    if args.backt:
        train_dataset = aug_by_backt(train_dataset=train_dataset, en_to_others=en_to_others, others_to_en=others_to_en, num_aug=NUM_AUG)
        
    if args.eda:
        train_dataset = aug_by_eda(train_dataset=train_dataset, num_aug=NUM_AUG)
        
    if args.masked_lm:
        train_dataset = aug_by_masked_lm(train_dataset=train_dataset, seed=seed, masked_lm=masked_lm, num_aug=NUM_AUG)
        
    if args.afinn:
        train_dataset = aug_by_afinn(train_dataset=train_dataset, unlabeled_dataset=train_rest_dataset, afinn=afinn)
    
    train_dataloader, val_dataloader, test_dataloader = transform_datasets(train_dataset=train_dataset,
                                                                           test_dataset=test_dataset, 
                                                                           seed=seed)
    
    # model preparation
    model_name = MODEL_FOLDER + 'model' + str(seed) + '.pth'
    if os.path.exists(model_name):
        model = torch.load(model_name)
    else:
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        torch.save(model, model_name)
    model.to(DEVICE)
    
    # training method
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=NUM_EPOCHS*len(train_dataloader)
    )
    
    train_model(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                optimizer=optimizer, lr_scheduler=lr_scheduler, num_epochs=NUM_EPOCHS)
    
    cur_accuracy = evaluate_model(model=model, test_dataloader=test_dataloader)
    acc_accuracy += cur_accuracy
    accuracy.append(cur_accuracy)
    
for i in range((NUM_SEED + 4) // 5):
    my_table = PrettyTable([j for j in range(i * 5, min((i + 1) * 5, NUM_SEED))])
    my_table.add_row(accuracy[(i * 5): min((i + 1) * 5, NUM_SEED)])
    print(my_table)
    
table_content = []    
table_content.append(DATASET_NAME)
table_content.append("O" if args.backt else "X")
table_content.append("O" if args.eda else "X")
table_content.append("O" if args.masked_lm else "X")
table_content.append("O" if args.afinn else "X")
table_content.append(acc_accuracy / NUM_SEED)

my_table = PrettyTable(["Dataset_name", "Backtranslation", "EDA", "Masked language model", "AFINN", "Average Accuracy"])
my_table.add_row(table_content)
print(my_table)

print(accuracy)

