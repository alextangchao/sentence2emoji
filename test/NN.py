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

FILEPATH = '490A final project data - Kai Processing.csv'
SENTENCE = 'senetence'
LABEL = 'translate'

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

def vectorize_sentence(filepath):
    bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
    
    sentences, emojis = read_csv(filepath)
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(emojis)
    
    total = np.array([])
    features = np.array([])
    for line, label in zip(sentences, labels):
        input_ids = torch.tensor([tokenizer.encode(line)])
        with torch.no_grad():
            feature = bertweet(input_ids).pooler_output  # Models outputs are now tuples
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
        self.input_layer = nn.Linear(768, 700)
        self.hidden1 = nn.Linear(700,600)
        self.hidden2 = nn.Linear(600,500)
        self.hidden3 = nn.Linear(500,400)
        self.hidden4 = nn.Linear(400,300)
        self.hidden5 = nn.Linear(300,200)
        self.output = nn.Linear(200,197)
        
    def forward(self, data):
        data = F.relu(self.input_layer(data))
        data = F.relu(self.hidden1(data))
        data = F.relu(self.hidden2(data))
        data = F.relu(self.hidden3(data))
        data = F.relu(self.hidden4(data))
        data = F.relu(self.hidden5(data))
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