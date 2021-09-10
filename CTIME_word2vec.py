import statistics
import pandas as pd
import numpy as np
from gensim.models.callbacks import CallbackAny2Vec
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from Preprocessing import process, parse_impression
from gensim.models import KeyedVectors, FastText
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from numpy import empty
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from gensim.models import Word2Vec
from time import time

class callback(CallbackAny2Vec):
    ### Callback to print loss after each epoch.

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        print('Time to train this epoch: {} mins'.format(round((time() - t) / 60, 2)))
        #model.wv.save_word2vec_format('ctime_w2v_epoch' + str(self.epoch + 1) + '.bin', binary=True)
        self.epoch += 1


### PROCESSING (X = reports, Y = y)
AllReports = pd.read_csv("data/CTIME/CTIMEDatasetComb.csv")
processed = []
labelled = []
unlabelled = []
for i in range(len(AllReports)):
    if (np.isnan(AllReports["IC_Mass_Effect"][i])): unlabelled.append(i)
    else: labelled.append(i)
    report = parse_impression(AllReports['CTReport'][i])
    rep = ' '.join(process(report))
    processed.append(rep)
AllReports['PROCESSED'] = processed

### Translation (optional)
data = open("data/CTIME_dictionary.txt","r")
baseTerms = []
translatedTerms = []
for i in range(105):
    line = data.readline()
    term = line.split(sep = "|")
    baseTerms.append(term[0])
    translatedTerms.append(term[1][:-1])
data.close()

for i in range(len(AllReports)):
    AllReports['PROCESSED'][i] = " " + AllReports['PROCESSED'][i] + " "
    for j in range(len(baseTerms)):
        AllReports['PROCESSED'][i] = AllReports['PROCESSED'][i].replace(" "+baseTerms[j]+" ", " "+translatedTerms[j]+" ")

# Train Embeddings
reports = AllReports['PROCESSED']
for i, report in enumerate(reports):
    reports[i] = (word_tokenize(report))
    
t = time()

# Fresh
wv = Word2Vec(reports, size=300, window=30, min_count=50, sg=1, iter = 50)
print('Time to build the model (5 epochs): {} mins'.format(round((time() - t) / 60, 2)))
wv.wv.save_word2vec_format('ctime_fasttext_nodict.bin', binary=True)

# Pre-Trained
'''
model = KeyedVectors.load_word2vec_format("oncoshare_w2v_epoch2.bin", binary=True)
model_2 = Word2Vec(size=300, window=30, min_count=50, sg=1, compute_loss=True, callbacks=[callback()])
model_2.build_vocab(reports)
total_examples = model_2.corpus_count
model_2.build_vocab([list(model.vocab.keys())], update=True)
model_2.intersect_word2vec_format("oncoshare_w2v_epoch2.bin", binary=True)
model_2.train(reports, total_examples=total_examples, epochs=5)
model_2.wv.save_word2vec_format('MIMIC_ctime_nodict.bin', binary=True)
'''

# Vectorization
tfidf = TfidfVectorizer(sublinear_tf=True)
features = tfidf.fit_transform(AllReports['PROCESSED'])
dict = {val : idx for idx, val in enumerate(tfidf.get_feature_names())}

preTrainedPath = "ctime_w2v_dict.bin"
wv = KeyedVectors.load_word2vec_format(preTrainedPath, binary=True)

x = empty([len(AllReports['PROCESSED']),300])
i = 0
for report in AllReports['PROCESSED']:
    words = report.split()
    avgFeat = np.zeros(300)
    for word in words:
        try: # if in vocab
            vector = np.multiply(wv[word],features[i,dict[word]])
            avgFeat = np.add(avgFeat, vector)
        except: # if out of vocab
            ''''''
    if (len(words) == 0): x[i] = avgFeat
    else: x[i] = avgFeat/len(words)
    i += 1

# Train Model
y = AllReports['IC_Mass_Effect'][labelled]
xTrain = x[labelled]
xTest = x[unlabelled]

text_clf = XGBClassifier(max_depth = 6, n_estimators= 100, booster = "gbtree", reg_alpha = 0.2)
text_clf = text_clf.fit(xTrain, y)

# Get Predictions
predict = text_clf.predict_proba(xTest)
pred = []
for i in range(len(xTest)):
    pred.append(predict[i][1])

predictions = np.empty(len(AllReports['PROCESSED'])) * np.nan
counter = 0
for index in unlabelled:
    predictions[index] = pred[counter]
    counter += 1

# Save Predictions
AllReports['Predictions'] = predictions
AllReports.to_csv("data/CTIME/CTIMEDatasetClassified_Word2Vec.csv")
