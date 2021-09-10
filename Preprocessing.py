import nltk
from nltk.corpus import stopwords
import re
import string
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from dateutil.parser import parse
from dateparser.search import search_dates
from dateutil.relativedelta import relativedelta
import spacy
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer



# parsing CTIME reports
def parse_impression(full_report):
    impression_words = []
    #all_words = re.findall(r"[w']+", full_report)
    all_lines = full_report.split('.')
    for index in range(len(all_lines)):
        line = all_lines[index].lower().strip()
        if ('report release date' in line       or 'i have reviewed' in line     or 'electronically reviewed by' in line    or 'electronically signed by' in line    or 'attending md' in line      or 'electronic signature by:' in line):
            break
        keywords = ['lesion', 'intraventricular', 'mass', 'midline shift', 'hemorr', 'hematoma', 'hernia', 'sah', 'subarachnoid', 'effacement']
        for keyword in keywords:
            if keyword in line:
                impression_words.append(line +'.')
                break
    # Check if parsing failed
    if len([word for line in impression_words for word in line.split()]) < 2:
        return ''
    else:
        return '\n'.join(impression_words)


### processes clinical note text
def process(text):
    # Remove excess whitespace
    spaces = "                         "
    for i in range(22):
        text = text.replace(spaces[i:],"\n")
    newlines = "\n\n\n\n\n\n\n\n\n\n"
    for i in range(10):
        text = text.replace(newlines[i:],"\n")

    # Remove dates/propernouns for MIMIC-3
    start = text.find("[**")
    while start != -1:
        end = text.find("**]", start)
        if((text[start+3:start+7]).isnumeric()): # does it begin w/ a year?
            text = text.replace(text[start:end+3],"date") # remove dates
        else:
            text = text.replace(text[start:end+3],"propernoun") # replace w/ propernoun
        start = text.find("[**")
    return text

    text = sent_tokenize(text)
    exclude_list = ['hour', 'aspect', 'downward', 'could', 'including', 'represents', 'follow', 'described', 'noted', 'pm', 'given', 'representing', 'along', 'well', 'causing', 'versus', 'more', 'could', 'involving', 'third', 'fourth', 'size', 'along', 'mm', 'also', 'right', 'left', 'seen', 'measuring', 'cm', 'amount', 'age', 'approximately', 'clinical', 'showing', 'year', 'old', 'finding', 'impression', 'may', 'represent', 'appears', 'identify', 'identified', 'identified', 'comment', 'additional', 'though', 'narrative', 'relative', 'comparison', 'history', 'compared', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'both', 'each', 'other', 'such', 'only', 'own', 'so', 'than', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    lemmatizer = WordNetLemmatizer()
    popList = []
    counter = 0

    for i in range(len(text)):
        numNumbers = 0
        text[i] = text[i].replace("\n", " ")
        text[i] = re.sub('['+ string.punctuation + ']', ' ', text[i]) #remove punctuation
        tagged_sent = pos_tag(text[i].split())

        # eliminate all proper nouns
        ppns = [word for word,pos in tagged_sent if pos == 'NNP']
        for ppn in ppns:
            text[i] = text[i].replace(ppn, "")

        text[i] = text[i].lower() #lowercase

        words = text[i].split(' ')
        sent = []
        for j in range(len(words)):
            if (words[j] not in exclude_list and len(words[j]) > 1):
                word = lemmatizer.lemmatize(words[j]) # lemmatization
                if (word.isnumeric()):
                    numNumbers += len(int2word(int(word)).split())
                elif (word in exclude_list or len(word) <= 1): pass # remove stopwords + (1,2-letter words)
                else: sent.append(word)
        text[i] = " ".join(sent)

        if (numNumbers > len(text[i])/2):
            popList.append(i-counter)
            counter += 1

    for index in popList:
        text.pop(index) # remove all sentence with <= 3 words

    return text
