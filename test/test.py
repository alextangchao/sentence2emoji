from sklearn.utils import multiclass
import torch, csv, emoji
import numpy as np
from transformers import AutoModel, AutoTokenizer 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings("ignore")

LR = LogisticRegression(multi_class='multinomial')
bertweet = AutoModel.from_pretrained("vinai/bertweet-base")

# For transformers v4.x+: 
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

# For transformers v3.x: 
# tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

# INPUT TWEET IS ALREADY NORMALIZED!
# line1 = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :crying_face:"
# line2 = "I am secretly deflating a random tire on Tweetman’s car every week."
# line3 = "y’all the second jab fatigue is real man 😭"
# lines = [line1, line2, line3]

# y = [0,1,0]

def read_csv(filepath, results=[], islist=False):
    flag = True if len(results) == 0 else False 
    with open(filepath, 'r', encoding='utf8') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')   
        counts = 0   
        sentences = []
        labels = []
        for row in csv_reader:
            sentences.append(row['Sentence'])
            labels.append([c for c in row['Label'] if c in emoji.UNICODE_EMOJI['en']])
            if flag:
                if islist:
                    results.append({row['Sentence']: row['Label']})
                else:
                    results.append({row['Sentence']: [row['Label']]})
            else:
                results[counts][row['Sentence']].append(row['Label'])
            counts += 1
    return sentences, labels, results

sentences, labels, rows = read_csv('490A final project data - Kai Finished.csv', [], True)
# print(labels)
labels = MultiLabelBinarizer().fit_transform(labels)
total = np.array([])
for line in sentences:
    input_ids = torch.tensor([tokenizer.encode(line)])
    with torch.no_grad():
        feature = bertweet(input_ids).pooler_output  # Models outputs are now tuples
        feature = feature[0].detach().cpu().numpy()
        if len(total) == 0:
            total = np.hstack((total, np.array(feature)))
        else:
            total = np.vstack((total, np.array(feature)))

print(total)
model = LR.fit(total, labels)
print(model.predict([total[1]]))
