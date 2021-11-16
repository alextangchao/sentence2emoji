import torch, csv, emoji
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import AutoModel, AutoTokenizer 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
# from sentence_transformers import SentenceTransformer


FILEPATH = '490A final project data - mmz Dataset.csv'
SENTENCE = 'Sentence'
LABEL = 'Translate'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

def read_csv(filepath):
    with open(filepath, 'r', encoding='utf8') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')   
        sentences = []
        labels = []
        for row in csv_reader:
            if row[LABEL] != '':
                try:
                    sentences.append(row[SENTENCE])
                    current_label = [c for c in row[LABEL] if c in emoji.UNICODE_EMOJI['en']][0]
                    labels.append(current_label)
                    print(current_label)
                except IndexError:
                    print(row)

    return sentences, labels

# def vectorize_sentence_sent_trans(filepath):
#     sent_trans = SentenceTransformer('all-mpnet-base-v2')
#
#     sentences, emojis = read_csv(filepath)
#     mlb = MultiLabelBinarizer()
#     labels = mlb.fit_transform(emojis)
#
#     for sentence, label in zip(sentences, labels):
#         feature = sent_trans.encode(sentence)
#         feature_label = np.concatenate((feature, label), axis=None)
#         print(len(feature))
#         total = np.array([])
#         features = np.array([])
#         if len(features) == 0:
#                 features = np.hstack((features, np.array(feature)))
#                 total = np.hstack((total, np.array(feature_label)))
#         else:
#             features = np.vstack((features, np.array(feature)))
#             total = np.vstack((total, np.array(feature_label)))
#
#     return features, labels, total, sentences, emojis, mlb.classes_

def vectorize_sentence(filepath):
    bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
    
    sentences, emojis = read_csv(filepath)
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(emojis)
    
    total = np.array([])
    features = np.array([])
    for text, label in zip(sentences, labels):
        encoding = tokenizer.encode_plus(
            text,
            # max_length=32,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            return_token_type_ids=False,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch tensors
        )
        with torch.no_grad():
            feature = bertweet(input_ids=encoding['input_ids'],
                        attention_mask=encoding['attention_mask']).pooler_output  # Models outputs are now tuples
            feature = feature[0].detach().cpu().numpy()
            feature_label = np.concatenate((feature, label), axis=None)
            if len(features) == 0:
                features = np.hstack((features, np.array(feature)))
                total = np.hstack((total, np.array(feature_label)))

            else:
                features = np.vstack((features, np.array(feature)))
                total = np.vstack((total, np.array(feature_label)))
                
    return features, labels, total, sentences, emojis, mlb.classes_

class emojiDataset(Dataset):
    def __init__(self):
        self.x, self.y, self.n_samples, self.sentences, self.emojis, self.y_classes = vectorize_sentence(FILEPATH)
        self.x = torch.from_numpy(self.x) 
        self.y = torch.from_numpy(self.y).type(torch.FloatTensor)

    def get_others(self):
        return self.sentences, self.emojis, self.y_classes

    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.n_samples.shape[0]


class toEmoji(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(768, 400)
        self.hidden1 = nn.Linear(400,200)
        self.output = nn.Linear(200,115)
        
    def forward(self, data):
        data = F.relu(self.input_layer(data))
        data = F.relu(self.hidden1(data))
        return self.output(data)

        # return F.log_softmax(data, dim=1)

# dataset = emojiDataset()

# data_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)
# data_iter = iter(data_loader)
# labels_classes, sentences = dataset.get_others()
# # next_data = data_iter.next()
# # features, labels = next_data
# # print(features, labels)

# toemoji = toEmoji()
# learn_rate = optim.Adam(toemoji.parameters(), lr=0.01)
# loss_func = nn.MSELoss()
# epochs = 5