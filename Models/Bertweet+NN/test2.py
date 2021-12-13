import torch, csv, emoji, os, sys
from torch.functional import Tensor
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import AutoModel, AutoTokenizer 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from torchtext.data.metrics import bleu_score
from utils import DATAPATH
# total = np.empty((0,3), float)
# a = torch.tensor([1,2,3,4,5])
# b = torch.tensor([6,7,8,9,0])
# # print(torch.cat((a, total.unsqueeze(-3)), dim = -1))
# a = a.detach().cpu().numpy()
# b = b.detach().cpu().numpy()
# total = np.append(total,np.array([a]))
# print(total)

# input = torch.randn(128, 20)
# print(input)



# a = '👥🥳🛹 ❓️❌️👥'
# result = []
# for elem in a:
#     if elem == '':
#         break
#     print('"{}"'.format(elem))
#     result.append(elem)
# emojis = [elem for elem in a if elem ]
# print([c for c in a if c in emoji.UNICODE_EMOJI['en']])
# labels = [['👥', '🥳', '🛹', '❓', '❌', '👥'], ['🤔','🤔','🤔']]
# labels = MultiLabelBinarizer().fit_transform(labels)
# print(labels)
# strings = ["first", "", "second ", " "]
# print([x.strip() for x in strings if x.strip()])


# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-mpnet-base-v2')

# #Our sentences we like to encode
# sentences = ['This framework generates embeddings for each input sentence',
#     'Sentences are passed as a list of string.',
#     'The quick brown fox jumps over the lazy dog.']

# #Sentences are encoded by calling model.encode()
# embeddings = model.encode(sentences)

# #Print the embeddings
# for sentence, embedding in zip(sentences, embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", type(embedding))
#     print("")


# result = [1.1246e-10, 3.4884e-06, 7.0702e-01, 8.0756e-12, 3.6213e-13, 1.0572e-16,
#         9.9666e-01, 4.2305e-09, 5.1845e-11, 1.7367e-10, 2.8301e-07, 1.7293e-12,
#         2.3389e-11, 2.6987e-03, 1.6586e-05, 4.1745e-07, 4.9597e-06, 9.0212e-07,
#         5.3653e-09, 4.5397e-05, 2.5676e-14, 5.0964e-05, 6.0854e-07, 6.1588e-08,
#         1.6123e-10, 2.5158e-12, 4.4337e-16, 9.3633e-08, 9.5668e-09, 6.5706e-03,
#         6.0348e-10, 1.0242e-11, 1.8018e-12, 1.4354e-11, 5.0031e-05, 6.9297e-10,
#         7.8121e-06, 1.7787e-16, 8.8719e-08, 1.0000e+00, 1.6783e-10, 6.1059e-04,
#         7.8990e-09, 2.8641e-15, 4.3115e-08, 3.6854e-09, 1.8736e-06, 3.0031e-03,
#         9.7079e-08, 2.0148e-09]

# result = [elem for elem in result if elem>0]
# print(result)


# bertweet = AutoModel.from_pretrained("vinai/bertweet-base")

# # For transformers v4.x+: 
# tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

# # For transformers v3.x: 
# # tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

# # INPUT TWEET IS ALREADY NORMALIZED!
# line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :crying_face:"

# input_ids = torch.tensor([tokenizer.encode(line)])

# with torch.no_grad():
#     features = bertweet(input_ids)  # Models outputs are now tuples
#     print(features)


def read_csv(filepath):
    with open(filepath, 'r', encoding='utf8') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')   
        sentences = []
        labels = []
        for row in csv_reader:
            if row[LABEL] != '':
                try:
                    sentences.append(row[SENTENCE])
                    current_label = [c for c in row[LABEL] if c in emoji.UNICODE_EMOJI['en']]
                    # current_label = [random.choice(CLASSES)]
                    labels.append(current_label)
                    # print(current_label)
                except IndexError:
                    print(row)

    return sentences, labels


# 490A final project data - Emoji-50-01
# def read_csv(folderpath):
#     print('\n-----------------------Begining Data Loading-----------------------')
#     sentences = []
#     emojis = []
#     for filename in os.listdir(folderpath):
#         if filename.endswith(".csv") and filename.startswith("490A final project data"):
#             filename_components = filename.split("-")
#             file_index = filename_components[-1].strip('.csv')
#             print("\nLoading data for file '{}'.".format(file_index))
#             sys.stdout.flush()
#             data_file = os.path.join(DATAPATH, filename)
            
#             num_data_cur_file = 0
#             with open(data_file, 'r', encoding='utf8') as csv_file:
#                 csv_reader = csv.DictReader(csv_file, delimiter=',') 
#                 sentence_name = csv_reader.fieldnames[0]
#                 label_name = csv_reader.fieldnames[1]
#                 for row in csv_reader:
#                     if row[label_name] != '':
#                         try:
#                             sentences.append(row[sentence_name])
#                             emojis.append([c for c in row[label_name] if c in emoji.UNICODE_EMOJI['en']])
#                             num_data_cur_file += 1
#                         except IndexError:
#                             print(row)
#             print(f'Loaded {num_data_cur_file} labelled sentence and emoji data.')
    
#     print(f'\nLoaded total {len(sentences)} labelled sentence and emoji data.')
#     print('\n-----------------------Finished Data Loading-----------------------\n')
#     return sentences, emojis

# # read_csv(DATAPATH)
a = [["1","0","1","1"]]
b = [[["1","1","1","0"]]]
print(bleu_score(a,b,max_n=2,weights=[0.5,0.5]))

# def convert_str(input_list):
#     result = []
#     for outter in input_list:
#         temp = []
#         for inner in outter:
#             temp.append(str(inner))
#         result.append(temp)
#     return result

# print(convert_str([[0,1,2,3]]))
def convert_1d_str(input_1d_list):
    result = []
    for inner in input_1d_list:
        if isinstance(inner, torch.Tensor):
            result.append(str(int(inner.item())))
        else:
            result.append(str(inner))
    return result

def convert_2d_str(input_2d_list):
    result = []
    for outter in input_2d_list:
        result.append(convert_1d_str(outter))
    return result


# print(emoji_accuracy(b[0][0],a[0]))
c = torch.Tensor([[0,1,1,2,3,]])

def emojis_accuracy(candidates, references):
    correct = 0
    total = 0
    for candidate, reference in zip(candidates, references):
        if int(reference) == 1: 
            total+=1
            if int(candidate) == int(reference): 
                correct += 1
    return float(correct/total)
d = ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
e = ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0']

print(emojis_accuracy(d,e))