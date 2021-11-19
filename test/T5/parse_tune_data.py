import json
import csv
import emoji
import re

FILE_PATH = ".\\..\\490A final project data - Emoji-50-467.csv"
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
                    current_label = [emoji.UNICODE_EMOJI['en'][c] for c in row[LABEL] if c in emoji.UNICODE_EMOJI['en']]
                    labels.append(current_label)
                    # print(current_label)
                except IndexError:
                    print(row)

    return sentences, labels


sentences, labels = read_csv(FILE_PATH)
# for i in range(5):
#     print(sentences[i], "|", labels[i])
# data = []
# for sentence, label in zip(sentences, labels):
#     data.append({"translation": {"en": sentence, "emoji": label}})


with open("fine_tune_data.json", "w", encoding="utf-8") as file:
    for sentence, label in zip(sentences, labels):
        # sentence=re.sub(r"\\.{5}","",sentence)
        data = json.dumps({"translation": {"en": sentence, "emoji": " ".join(label)}})
        file.write(data + "\n")
