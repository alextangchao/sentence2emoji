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


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')

#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", type(embedding))
    print("")