from sklearn.utils import multiclass
import torch, csv, emoji, spacy, random
import numpy as np
import torch.nn as nn
from torch.nn import Sigmoid
import torch.optim as optim
from torch._C import device
from torch.nn.modules import dropout
from torch.nn.modules.sparse import Embedding
from torch.utils.data import DataLoader, Dataset
from torchtext.legacy.data import Example, Field, BucketIterator, Iterator
from transformers import AutoModel, AutoTokenizer 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from utils import DATAPATH, FILEPATH, EMOJIPATH, EMOJI_VOCAB


# CLASSES = EMOJI_VOCAB
CLASSES = ['❌️','😂','👨‍⚕️'] # ,'👨','♥️','🏬','😞','🅰️','👨‍👨‍👧‍👦','😷','💊','😪','➡️','🎤','🌃','🤩','💀','🍽️','🤦','👃']
SENTENCE = 'senetence'
LABEL = 'translate'

def read_csv(filepath):
    with open(filepath, 'r', encoding='utf8') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')   
        sentences = []
        labels = []
        for row in csv_reader:
            if row[LABEL] != '':
                try:
                    sentences.append(row[SENTENCE])
                    # current_label = [c for c in row[LABEL] if c in emoji.UNICODE_EMOJI['en']]
                    current_label = [random.choice(CLASSES)]
                    labels.append(current_label)
                    # print(current_label)
                except IndexError:
                    print(row)

    return sentences, labels

def vectorize_sentences(filepath):
    bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
    
    # sentences, labels = read_csv(filepath)
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
            
    
    # print(features.shape)
    SS = StandardScaler()
    # print(features)
    features = SS.fit_transform(features)
    # features = stats.zscore(features, axis=1, ddof=1)
    # print(labels.shape)
    # return features, labels, total, sentences, labels, [0,1,2]
    return features, labels, total, sentences, emojis, mlb.classes_

class emojiDataset(Dataset):
    def __init__(self):
        self.x, self.y, self.n_samples, self.sentences, self.emojis, self.y_classes = vectorize_sentences(FILEPATH)
        self.x = torch.from_numpy(self.x) 
        # self.y = torch.FloatTensor(self.y)
        self.y = torch.from_numpy(self.y).type(torch.LongTensor)

    def get_others(self):
        return self.sentences, self.emojis, self.y_classes

    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.n_samples.shape[0]

class Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dp):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        embedding_size = 250
        self.dropout = nn.Dropout(dp)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dp)
        self.fc = nn.Linear(hidden_size, output_size) #fullyconnected =>hidden size to output size

    def forward(self, x, hidden, cell):
        # x is (N) because we want one word at a time, and we want(1,N)
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # dropout shape: (1, N, input_size)
        if cell is None:
            cell = np.zeros((hidden.shape[0], hidden.shape[1], x.shape[1]))
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # shape of outputs: (1, N, hidden_size)

        prediction = self.fc(outputs)
        # shape of predictions: (1, N, length of vocab)
        prediction = prediction.squeeze(0)
        # shape of predictions: (N, length of vocab)

        return prediction, hidden, cell

class Sentence2emoji(nn.Module):
    def __init__(self, decoder):
        super(Sentence2emoji, self).__init__()
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(CLASSES)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden = source
        print(hidden.shape)

        # first token
        x = target[0]
        cell = None
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output
            # output would be (N, emoji_vocab_size)
            
            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs





device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# vectors, labels, n_samples, sentences, emojis, y_classes = vectorize_sentences(FILEPATH)

step = 0
epochs = 10
batch_size = 16

dataset = emojiDataset()
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
sentences, emojis, labels_classes = dataset.get_others()


input_size_decoder = 768
output_size = len(CLASSES)
hidden_size = 100
num_layers = 2
decoder_dropout = 0.5
decoder_net = Decoder(input_size_decoder, output_size, hidden_size, num_layers, decoder_dropout)

model = Sentence2emoji(decoder_net)
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()
S = Sigmoid()

for i in range(epochs):
    print(f'Epoch [{i} / {epochs}]')

    for data in data_loader:
        sentence, label = data
        # sentence with shape (batch_size,768) label with shape (batch_size, emoji_vocab_size)
        output = model(sentence, label)
        print(output.shape)
        break
    break
        # # output shape: (target length, batch size, output dim)

        # output = output[1:].reshape(-1, output.shape[2])
        # target = target[1:].reshape(-1)

        # optimizer.zero_grad()
        # loss = criterion(output, target)

        # loss.backward()

        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        # optimizer.step()
        # step += 1



