import csv
import emoji
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertweetTokenizer, RobertaModel, RobertaTokenizer

FILEPATH = '490A final project data - Kai Processing.csv'
SENTENCE = 'senetence'
LABEL = 'translate'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
print(f"total device: {torch.cuda.device_count()}")

bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)


def read_csv(filepath):
    with open(filepath, 'r', encoding='utf8') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        sentences = []
        labels = []
        for row in csv_reader:
            if row[LABEL] != '':
                sentences.append(row[SENTENCE])
                labels.append([c for c in row[LABEL] if c in emoji.UNICODE_EMOJI['en']])
    return sentences, labels


sentences, labels = read_csv(FILEPATH)
max_len = max([len(i) for i in sentences])
print(len(sentences))
print(f"max len: {max_len}")
print(sentences[0])
print(sentences[-1])

# INPUT TWEET IS ALREADY NORMALIZED!
sample_txt = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :crying_face:"


def get_sentence_vec(text):
    encoding = tokenizer.encode_plus(
        text,
        max_length=max_len,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        return_token_type_ids=False,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',  # Return PyTorch tensors
    )
    # print(f"cls: {tokenizer.cls_token_id}")
    # encoding["input_ids"][0][0]=tokenizer.cls_token_id
    print(f"sentence: {text}")
    # print(f"encoding: {encoding}")

    tags = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

    # print(f"tag: {tags}")

    with torch.no_grad():
        features = bertweet(input_ids=encoding['input_ids'],
                            attention_mask=encoding['attention_mask'])  # Models outputs are now tuples
        # print(features)
        # print(type(features))
        # print(type(features.pooler_output))
        # print(features.pooler_output.size())
        # print(features.last_hidden_state.size())
        # print(sum(features.pooler_output[0]))

        return features.pooler_output


record = [np.array(get_sentence_vec(sentences[0])[0])]
for sentence in sentences[:10]:

    vec = np.array(get_sentence_vec(sentence)[0])

    print(f"vec sum: {np.sum(vec)}")
    print(f"vec average: {np.average(vec)}")
    # print(f"distance: {np.linalg.norm(vec - record[-1])}")
    # print(f"first num: {vec[0]}")
    record.append(vec)
    print("-----------------------------------------------------------")

print("distance between each pair vector")
total_distance = []
record=record[1:]
for i in record:
    temp = []
    for k in record:
        distance = np.linalg.norm(i - k)
        temp.append(distance)
    total_distance.append(temp)

for i in total_distance:
    print(i)

# record[0].to('cuda:1')
# print(record[0].get_device())
# print(torch.cuda.get_device_name(record[0].get_device()))
