import json
import csv
import emoji
import re


sentences = ["My family spoils my daughter so much and girl ainβt even here yet π­",
"Do yβall understand how fast whoever made the meme of this had to be to catch it at this perfect moment πππ",
"SOLAR POWER MONTH OMFGππππ",
"im fucking bored",]


sentence=re.sub(r"\\.{5}","",sentences[0])
print(sentence)