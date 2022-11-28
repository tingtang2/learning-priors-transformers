import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from tokenizers import Tokenizer
from transformers import AutoTokenizer



def generateMasks(text):
    print(text)

class WikiDataset(Dataset): #wrapper around arrow dataset
    def __init__(self, og_dataset):
        self.data = og_dataset
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        print("data len: {}".format(len(self.data)))
        
    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        ret_dict = self.tokenizer(text=text)
        ids = ret_dict["input_ids"]
        id_tensor = torch.tensor(ids)
        seq_len = id_tensor.shape[0]
        if (seq_len > 512):
            id_tensor = id_tensor[:512]
            # if seq_len % 2 != 0:
            #     id_tensor = id_tensor[: -1]
            # id_tensor.reshape(2, seq_len // 2)
        # print(id_tensor.shape)
        labels = torch.clone(id_tensor)
        rand = torch.rand(id_tensor.shape)
        mask_arr = (rand < 0.15) * (id_tensor != self.tokenizer.cls_token_id) * (id_tensor != self.tokenizer.sep_token_id)
        labels[~mask_arr] = -100
        id_tensor[mask_arr] = self.tokenizer.mask_token_id
        # print(max(id_tensor), min(id_tensor))
        # print(id_tensor, labels)
        
        return (id_tensor, labels)



def get_dataset():
    dataset = load_dataset("wikitext", 'wikitext-2-v1')
    train = dataset["train"]
    test = dataset["test"]
    train_set = WikiDataset(train)
    test_set = WikiDataset(test)
        
    return train_set, test_set


BATCH_SIZE = 1
def get_loaders():
    train_set, test_set = get_dataset()
    loaders_dict = {
            "train": torch.utils.data.DataLoader(
                train_set,
                batch_size=BATCH_SIZE,
                shuffle=True,
            ),
            "test": torch.utils.data.DataLoader(
                test_set,
                batch_size=BATCH_SIZE,
                shuffle=True,
            )
    }
    mask_token = train_set.tokenizer.mask_token_id
    return (mask_token, loaders_dict)
    
        