import json
import csv
import emoji
import re


sentences = ["My family spoils my daughter so much and girl ain’t even here yet 😭",
"Do y’all understand how fast whoever made the meme of this had to be to catch it at this perfect moment 😂😂😂",
"SOLAR POWER MONTH OMFG🌞🌞🌞🌞",
"im fucking bored",]


sentence=re.sub(r"\\.{5}","",sentences[0])
print(sentence)