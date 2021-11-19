import torch, csv, emoji
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


result = [1.1246e-10, 3.4884e-06, 7.0702e-01, 8.0756e-12, 3.6213e-13, 1.0572e-16,
        9.9666e-01, 4.2305e-09, 5.1845e-11, 1.7367e-10, 2.8301e-07, 1.7293e-12,
        2.3389e-11, 2.6987e-03, 1.6586e-05, 4.1745e-07, 4.9597e-06, 9.0212e-07,
        5.3653e-09, 4.5397e-05, 2.5676e-14, 5.0964e-05, 6.0854e-07, 6.1588e-08,
        1.6123e-10, 2.5158e-12, 4.4337e-16, 9.3633e-08, 9.5668e-09, 6.5706e-03,
        6.0348e-10, 1.0242e-11, 1.8018e-12, 1.4354e-11, 5.0031e-05, 6.9297e-10,
        7.8121e-06, 1.7787e-16, 8.8719e-08, 1.0000e+00, 1.6783e-10, 6.1059e-04,
        7.8990e-09, 2.8641e-15, 4.3115e-08, 3.6854e-09, 1.8736e-06, 3.0031e-03,
        9.7079e-08, 2.0148e-09]

result = [elem for elem in result if elem>0]
print(result)