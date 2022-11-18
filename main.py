import sys
from model import model
from datasets import load_dataset
from data import PTBDataset

from torch import nn
import torch
from transformers import BertTokenizer

from typing import Dict

# high level globals
torch.manual_seed(11172022)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# functions to apply to huggingface dataset object
def tokenize_function(examples: Dict):
    return tokenizer(examples['sentence'], return_tensors='pt')

def generate_mask(examples: Dict):
    return {'mask': ['MASK']}


def create_dataloaders():
    dataset = load_dataset('ptb_text_only')
    

    lm_datasets = dataset.map(tokenize_function, batched=True, batch_size=1000, num_proc=4, remove_columns=['sentence'])

    
    processed_train_set = [tokenizer(text=example['sentence'], return_tensors='pt') for example in dataset['train']]
    print(len(processed_train_set))

    train_data = PTBDataset(processed_train_set)

    return dataset


def generate_masks():
    pass

def train():
    pass


def eval():
    pass


def main() -> int:
    m: nn.Module = model
    print(create_dataloaders())    


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return 0

if __name__ == '__main__':
    sys.exit(main())