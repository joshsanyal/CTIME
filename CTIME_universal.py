import tensorflow as tf
import statistics
import numpy
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from Preprocessing import process
from xgboost import XGBClassifier
from Preprocessing import parse_impression
from collections import Counter
from nltk import ngrams

### PROCESSING (X = reports, Y = y)
AllReports = pd.read_csv("data/CTIME/CTIMEDatasetComb.csv")
processed = []
labelled = []
unlabelled = []
for i in range(len(AllReports)):
    if (numpy.isnan(AllReports["IC_Mass_Effect"][i])): unlabelled.append(i)
    else: labelled.append(i)
    report = parse_impression(AllReports['CTReport'][i])
    rep = ' '.join(process(report))
    processed.append(rep)
AllReports['PROCESSED'] = processed

# Translate using the custom CTIME dictionary
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


# Apply the Universal Sent Encoder
x = AllReports['PROCESSED'][labelled]
y = AllReports['IC_Mass_Effect'][labelled]


# Prep Dataset
embed = tf.keras.models.load_model("/Users/josh/Documents/universal-sentence-encoder_4")
message_embeddings = embed(x).numpy()
xTest = AllReports['PROCESSED'][unlabelled]
test_embeddings = embed(xTest).numpy()

# Train Model
text_clf = XGBClassifier(max_depth = 6, n_estimators= 100, booster = "gbtree", reg_alpha = 0.2)
text_clf = text_clf.fit(message_embeddings, y)

# Generate Predictions
predict = text_clf.predict_proba(test_embeddings)
pred = []
for i in range(len(xTest)):
    pred.append(predict[i][1])

predictions = numpy.empty(len(AllReports['PROCESSED'])) * numpy.nan
counter = 0
for index in unlabelled:
    predictions[index] = pred[counter]
    counter += 1

# Save Predictions
AllReports['Predictions'] = predictions
AllReports.to_csv("data/CTIME/CTIMEDatasetClassified_UniversalSentenceEncoder.csv")