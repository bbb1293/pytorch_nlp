# pytorch_nlp
For PyTorch practice and graduation

You don't need to download dataset or models by hand.

You just simply use awesome packages such as datasets, transformers, etc.

## Requirements
```bash
pip install -r requirements.txt
```

## Usage

```bash
(ml) udnet@dmcb-System:~/graduation/pytorch_nlp$ python sst2_bert.py -h
usage: sst2_bert.py [-h] [--gpu_num GPU_NUM] [--num_train_data NUM_TRAIN_DATA] [--num_seed NUM_SEED] [--num_epochs NUM_EPOCHS] [--backt] [--eda] [--masked_lm]

Set some arguments for training

options:
  -h, --help            show this help message and exit
  --gpu_num GPU_NUM     gpu num you want to use
  --num_train_data NUM_TRAIN_DATA
                        the number of the training data
  --num_seed NUM_SEED   the number of the seeds
  --num_epochs NUM_EPOCHS
                        the number of the epochs
  --backt               augment training data by backtranslation
  --eda                 augment training data by EDA
  --masked_lm           augment training data by masked language model
```

## Examples

If you want to see the result after training with augmented data by EDA with gpu 1

```bash
python sst2_bert.py --gpu_num 1 --eda
```

If you want to see the result after training with augmented data by backtranslation with gpu 2, with 30 seeds, with 200 epochs

```bash
python sst2_bert.py --gpu_num 2 --backt --num_seed 30 --num_epochs 200
```