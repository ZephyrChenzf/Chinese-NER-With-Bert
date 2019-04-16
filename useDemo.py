"""
@autor: chenzf
@file: useDemo.py
@time: 2019/3/19 3:27 PM

"""
import torch
from pytorch_pretrained_bert import BertTokenizer,BertModel,BertForMaskedLM
import logging
logging.basicConfig(level=logging.INFO)
import os
os.system("curl -d 'DDDDD=2017140433&upass=215035&0MKKey=''' '10.3.8.211'")
tokenizer=BertTokenizer.from_pretrained('bert-base-chinese',do_lower_case)
text = "[CLS] 李小龙是谁? [SEP]"
tokenized_text=tokenizer.tokenize(text)
# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 3
tokenized_text[masked_index] = '[MASK]'
# assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']
# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
# segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
segments_ids = [0, 0, 0, 0, 0, 0, 0, 0]
# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
model = BertModel.from_pretrained('bert-base-chinese')
model.eval()

# If you have a GPU, put everything on cuda
device=torch.device('cuda:2')
tokens_tensor = tokens_tensor.to(device)
segments_tensors = segments_tensors.to(device)
model.to(device)
# Predict hidden states features for each layer
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, segments_tensors)
# We have a hidden states for each of the 12 layers in model bert-base-uncased
assert len(encoded_layers) == 12

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-chinese')
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to(device)
segments_tensors = segments_tensors.to(device)
model.to(device)

# Predict all tokens
with torch.no_grad():
    predictions = model(tokens_tensor, segments_tensors)

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == '龙'

pass