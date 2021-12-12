import json
import csv
import emoji
from collections import Counter
import matplotlib.pyplot as plt

DATA_PATH = ".\\..\\Data\\"
FILE_PATH = DATA_PATH + "490A final project data - Emoji-50-467.csv"
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
                    current_label = [c for c in row[LABEL] if c in emoji.UNICODE_EMOJI['en']]
                    labels.append(current_label)
                    # print(current_label)
                except IndexError:
                    print(row)

    return sentences, labels


sentences, labels = read_csv(FILE_PATH)

vocab = Counter()
vocab_num = [0]
emoji_num = [0]
for emoji_list in labels:
    for emoji in emoji_list:
        vocab[emoji] += 1
    vocab_num.append(len(vocab.keys()))
    emoji_num.append(emoji_num[-1] + len(emoji_list))
    # print(vocab)
    # print(vocab_num)
    # print(emoji_num)
    # print("---------------------------")

x_list = list(range(0, len(labels) + 1))
plt.plot(x_list, vocab_num)
# plt.plot(x_list, emoji_num)
plt.xticks(list(range(0, 501, 100)))
plt.xlabel("Number of Sentences")
plt.ylabel("Used Emoji Vocabulary")
plt.title("Number of Sentences vs. Used Emoji Vocabulary")
plt.show()

plt.plot(x_list, emoji_num, color="r")
# plt.plot(x_list, emoji_num)
plt.xticks(list(range(0, 501, 100)))
plt.xlabel("Number of Sentences")
plt.ylabel("Used Emoji Number")
plt.title("Number of Sentences vs. Used Emoji Number")
plt.show()
