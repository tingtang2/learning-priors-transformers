from transformers import BertModel, BertTokenizer

# could also use 'prajjwal1/bert-small'
bert = BertModel.from_pretrained('bert-base-uncased')