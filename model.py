from transformers import BertForMaskedLM, BertConfig

config = BertConfig()
model = BertForMaskedLM(config=config)