from sklearn.utils import multiclass
import torch, csv, emoji, spacy
import numpy as np
from torch._C import device
from torch.nn.modules import dropout
from torch.nn.modules.sparse import Embedding
from transformers import AutoModel, AutoTokenizer 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings("ignore")

import torch, spacy, random
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.data import Dataset, Example, Field, BucketIterator, Iterator
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")

english = Field(
    tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
)

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dp):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dp)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dp)

    def forward(self, x):
        # x is word index in vocab (vector with indices) with shape: (seq_length, N), N is batch_size
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_output_size, embedding_size, hidden_size, num_layers, dp):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dp)
        self.embedding = nn.Embedding(input_output_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dp)
        self.fc = nn.Linear(hidden_size, input_output_size) #fullyconnected =>hidden size to output size

    def forward(self, x, hidden, cell):
        # x is (N) because we want one word at a time, and we want(1,N)
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # shape of outputs: (1, N, hidden_size)

        prediction = self.fc(outputs)
        # shape of predictions: (1, N, length of vocab)
        prediction = prediction.squeeze(0)
        # shape of predictions: (N, length of vocab)

        return prediction, hidden, cell


class Sentence2emoji(nn.Module):
    def __init__(self, encoder, decoder):
        super(Sentence2emoji, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        print(source)
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(emoji_field.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        # first token
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output
            # output would be (N, emoji_vocab_size)
            
            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs



epochs = 20
learning_rate = 0.001
batch_size = 1

load_model = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(sentences_field.vocab)
input_size_decoder = len(emoji_field.vocab)
output_size = len(emoji_field.vocab)
encoder_embedding_size = 1
decoder_embedding_size = 1
hidden_size = 100
num_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5

writer = SummaryWriter(f'runs/loss_plot')
step = 0

data_iter = Iterator(dt, batch_size=2, sort_key=lambda x: len(x), device=device)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, encoder_dropout)

decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, num_layers, decoder_dropout)

model = Sentence2emoji(encoder_net, decoder_net)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = emoji_field.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

for i in range(epochs):
    print(f'Epoch [{i} / {epochs}]')

    for index, batch in enumerate(data_iter):
        input_data, target = batch

        output = model(input_data, target)
        # output shape: (target length, batch size, output dim)

        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        writer.add_scalar('Training loss', loss, global_step=step)
        step += 1