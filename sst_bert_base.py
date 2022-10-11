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
val_dataset = dataset["validation"]


# ## Transform Dataset

# In[ ]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Tokenize help method
def apply_transform(x):
    return tokenizer(x["sentence"], padding="max_length", truncation=True)

tokenized_train_dataset = train_dataset.map(apply_transform, batched=True)
tokenized_val_dataset = val_dataset.map(apply_transform, batched=True)


# In[ ]:


# To fit the model's input
tokenized_train_dataset = tokenized_train_dataset.remove_columns(['sentence', 'idx'])
tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
tokenized_val_dataset = tokenized_val_dataset.remove_columns(['sentence', 'idx'])
tokenized_val_dataset = tokenized_val_dataset.rename_column("label", "labels")

# labels, input_ids, toekn_type_idx, attention_mask
# Convert format to torch
tokenized_train_dataset.set_format("torch")
tokenized_val_dataset.set_format("torch")


# In[ ]:


train_dataloader = DataLoader(tokenized_train_dataset, batch_size=8, shuffle=None)
val_dataloader = DataLoader(tokenized_val_dataset, batch_size=8, shuffle=None)


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
optim = AdamW(model.parameters(), lr=learning_rate)


# In[ ]:


from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optim, num_warmup_steps=0, num_training_steps=num_training_steps
)


# ## Evaluate before Training

# In[ ]:


import evaluate
from tqdm.auto import tqdm

metric = evaluate.load("accuracy")
model.eval()

progress_bar = tqdm(range(len(val_dataloader)))

for val_batch in val_dataloader:
    val_batch = {k: v.to(DEVICE) for k, v in val_batch.items()}
    with torch.no_grad():
        outputs = model(**val_batch)
        
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=val_batch["labels"])
                    
    progress_bar.update(1)
    
metric.compute()


# ## Train

# In[ ]:


progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for train_batch in train_dataloader:
        train_batch = {k: v.to(DEVICE) for k, v in train_batch.items()}
        outputs = model(**train_batch)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


# ## Evaluate

# In[ ]:


metric = evaluate.load("accuracy")
model.eval()

progress_bar = tqdm(range(len(val_dataloader)))

for val_batch in val_dataloader:
    val_batch = {k: v.to(DEVICE) for k, v in val_batch.items()}
    with torch.no_grad():
        outputs = model(**val_batch)
        
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=val_batch["labels"])
                    
    progress_bar.update(1)
    
metric.compute()

