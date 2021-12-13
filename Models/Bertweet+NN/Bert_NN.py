import torch, csv, emoji, random, os, sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy import stats
from transformers import AutoModel, AutoTokenizer 
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from timeit import default_timer as timer
from utils import EMOJIPATH, EMOJI_VOCAB
# from sentence_transformers import SentenceTransformer

CLASSES = EMOJI_VOCAB
# CLASSES = ['❌️','😂','👨‍⚕️'] # ,'👨','♥️','🏬','😞','🅰️','👨‍👨‍👧‍👦','😷','💊','😪','➡️','🎤','🌃','🤩','💀','🍽️','🤦','👃']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

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

def emojis_accuracy(candidates, references):
    correct = 0
    total = 0
    for candidate, reference in zip(candidates, references):
        if int(reference) == 1: 
            total+=1
            if int(candidate) == int(reference): 
                correct += 1
    return float(correct/total)

def read_csv(folderpath):
    print('\n-----------------------Begining Data Loading-----------------------')
    extract_start = timer()
    sentences = []
    emojis = []
    for filename in os.listdir(folderpath):
        if filename.endswith(".csv") and filename.startswith("490A final project data"):
            filename_components = filename.split("-")
            file_index = filename_components[-1].strip('.csv')
            print("\nLoading data for file '{}'.".format(file_index))
            sys.stdout.flush()
            data_file = os.path.join(folderpath, filename)
            
            num_data_cur_file = 0
            with open(data_file, 'r', encoding='utf8') as csv_file:
                csv_reader = csv.DictReader(csv_file, delimiter=',') 
                sentence_name = csv_reader.fieldnames[0]
                label_name = csv_reader.fieldnames[1]
                for row in csv_reader:
                    if row[label_name] != '':
                        try:
                            sentences.append(row[sentence_name])
                            emojis.append([c for c in row[label_name] if c in emoji.UNICODE_EMOJI['en']])
                            num_data_cur_file += 1
                        except IndexError:
                            print(row)
            print(f'Loaded {num_data_cur_file} labelled sentence and emoji data.')
    
    extract_end = timer()
    print('\nLoaded total {} labelled sentence and emoji data in {:.1f} minutes.'.format(len(sentences), (extract_end-extract_start)/60))
    print('\n-----------------------Finished Data Loading-----------------------\n')
    return sentences, emojis
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

def vectorize_sentences(filepath):
    bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
    
    # sentences, labels = read_csv(filepath)
    sentences, emojis = read_csv(filepath)
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(emojis)
    
    total = np.array([])
    features = np.array([])

    print('\n-----------------------Begining Feature Extraction-----------------------')
    extract_start = timer()

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
            
    
    # print(features.shape)
    SS = StandardScaler()
    # print(features)
    features = SS.fit_transform(features)
    # features = stats.zscore(features, axis=1, ddof=1)
    
    # return features, labels, total, sentences, labels, [0,1,2]

    extract_end = timer()
    print('Extracted total {} feature and label data in {:.1f} minutes.'.format(len(features), (extract_end-extract_start)/60))
    print('\n-----------------------Finished Feature Extraction-----------------------\n')

    return features, labels, total, sentences, emojis, mlb.classes_

class emojiDataset(Dataset):
    def __init__(self, features, labels):
        self.x = features
        self.y = labels
        self.n_samples = features.shape[0]
        self.x = torch.from_numpy(self.x) 
        # self.y = torch.FloatTensor(self.y)
        self.y = torch.from_numpy(self.y).type(torch.FloatTensor)

    def get_others(self):
        return self.sentences, self.emojis, self.y_classes

    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.n_samples


class toEmoji(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(768, 400)
        self.hidden1 = nn.Linear(400,200)
        self.output = nn.Linear(200,50)
        
    def forward(self, data):
        data = F.relu(self.input_layer(data))
        data = F.relu(self.hidden1(data))
        # data = self.output(data)
        return self.output(data)
        # return F.sigmoid(data)

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
# vectorize_sentence(FILEPATH)