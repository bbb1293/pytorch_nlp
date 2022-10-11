#!/usr/bin/env python
# coding: utf-8

# # SST-2 Binary Text Classification with BERT Model (ref: [Transformers](https://huggingface.co/docs/transformers/training), [EDA](https://github.com/jasonwei20/eda_nlp))

# ## Common Imports

# In[ ]:


import torch
import torch.nn as nn

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SEED = 0
train_data_cnt = 32
torch.manual_seed(SEED)


# ## Load Dataset

# In[ ]:


from datasets import load_dataset
from torch.utils.data import DataLoader 

dataset = load_dataset("sst2")

# idx, sentence, label
train_dataset = dataset["train"].shuffle(seed=SEED).select(range(train_data_cnt))
test_dataset = dataset["validation"]


# ## Data Augmentation by Backtranslation

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


# In[ ]:


from transformers import pipeline

en_to_others = [pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr"), pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")]
others_to_en = [pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en"), pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")]


# In[ ]:


# train_dataset = aug_by_backt(train_dataset, en_to_others, others_to_en)
aug_train_dataset = aug_by_backt(train_dataset, en_to_others, others_to_en)


# ## Data Augmentation by EDA

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
def aug_by_eda(train_dataset, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1, num_aug=9):
    sentences = [train_data["sentence"] for train_data in train_dataset]
    labels = [train_data["label"] for train_data in train_dataset]
    
    aug_by_eda_train_dataset = train_dataset
    aug_sentences = [eda(sentence) for sentence in sentences]
    
    for i in range(len(labels)):
        for aug_sentence in aug_sentences[i]:
            aug_data = {'sentence': aug_sentence, 'label': labels[i]}
            aug_by_eda_train_dataset = aug_by_eda_train_dataset.add_item(aug_data)
            
    return aug_by_eda_train_dataset


# In[ ]:


# train_dataset = aug_by_eda(train_dataset)
aug_train_dataset = aug_by_eda(train_dataset)


# ## Transform Dataset

# In[ ]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Tokenize help method
def apply_transform(x):
    return tokenizer(x["sentence"], padding="max_length", truncation=True)

tokenized_train_dataset = train_dataset.map(apply_transform, batched=True)
tokenized_test_dataset = test_dataset.map(apply_transform, batched=True)


# In[ ]:


# To fit the model's input
tokenized_train_dataset = tokenized_train_dataset.remove_columns(['sentence', 'idx'])
tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
tokenized_test_dataset = tokenized_test_dataset.remove_columns(['sentence', 'idx'])
tokenized_test_dataset = tokenized_test_dataset.rename_column("label", "labels")

# labels, input_ids, toekn_type_idx, attention_mask
# Convert format to torch
tokenized_train_dataset.set_format("torch")
tokenized_test_dataset.set_format("torch")


# In[ ]:


tokenized_train_dataset = tokenized_train_dataset.train_test_split(test_size=0.5, shuffle=False)
train_dataloader = DataLoader(tokenized_train_dataset["train"], batch_size=8, shuffle=None)
val_dataloader = DataLoader(tokenized_train_dataset["test"], batch_size=8, shuffle=None)
test_dataloader = DataLoader(tokenized_test_dataset, batch_size=8, shuffle=None)


# ## Model Preparation

# In[ ]:


from transformers import AutoModelForSequenceClassification

num_classes = 2

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
model.to(DEVICE)


# ## Training Methods

# In[ ]:


from torch.optim import AdamW

learning_rate = 1e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)


# In[ ]:


from transformers import get_scheduler

num_epochs = 200
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)


# ## Train

# In[ ]:


from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

train_losses = []
val_losses = []

model.train()
for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    
    for train_batch, val_batch in zip(train_dataloader, val_dataloader):
        train_batch = {k: v.to(DEVICE) for k, v in train_batch.items()}
        outputs = model(**train_batch)
        loss = outputs.loss
        loss.backward()
        
        train_loss += loss.item()
        with torch.no_grad():
            val_loss += model(**val_batch).loss.item()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        
    train_losses.append(train_loss)
    val_losses.append(val_loss)


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(train_losses, 'r')
plt.plot(val_losses, 'g')

plt.savefig("loss.png")


# ## Evaluate

# In[ ]:


import evaluate
from tqdm.auto import tqdm

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
    
metric.compute()

