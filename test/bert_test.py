import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertweetTokenizer, RobertaModel,RobertaTokenizer

bertweet = AutoModel.from_pretrained("vinai/bertweet-base")

# For transformers v4.x+:
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

# INPUT TWEET IS ALREADY NORMALIZED!
sample_txt = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :crying_face:"

encoding = tokenizer.encode_plus(
    sample_txt,
    # max_length=32,
    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
    return_token_type_ids=False,
    padding=True,
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt',  # Return PyTorch tensors
)
# print(f"cls: {tokenizer.cls_token_id}")
# encoding["input_ids"][0][0]=tokenizer.cls_token_id
print(encoding)

tag = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

print(tag)

with torch.no_grad():
    features = bertweet(input_ids=encoding['input_ids'],
                        attention_mask=encoding['attention_mask'])  # Models outputs are now tuples
    print(features)
    print(type(features))
    print(type(features.pooler_output))
    print(features.pooler_output.size())
    print(features.last_hidden_state.size())
    print(sum(features.pooler_output[0]))
