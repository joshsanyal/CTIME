import statistics
import numpy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from Preprocessing import process, parse_impression
from xgboost import XGBClassifier
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

# Translation
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

# Prep Data
x = AllReports['PROCESSED'][labelled]
y = AllReports['IC_Mass_Effect'][labelled]
xTest = AllReports['PROCESSED'][unlabelled]

# Train Model
text_clf = Pipeline([('vect', CountVectorizer(max_features=350)),('tfidf', TfidfTransformer()),
        ('clf', XGBClassifier(max_depth = 10, n_estimators= 100, booster = "gbtree", reg_alpha = 0.2)),])
text_clf = text_clf.fit(x, y)


# Generate predicctions
predict = text_clf.predict_proba(xTest)
pred = []
for i in range(len(xTest)):
    pred.append(predict[i][1])

predictions = numpy.empty(len(AllReports['PROCESSED'])) * numpy.nan
counter = 0
for index in unlabelled:
    predictions[index] = pred[counter]
    counter += 1

# Save predictions
AllReports['Predictions'] = predictions
AllReports.to_csv("data/CTIME/CTIMEDatasetClassified_v2.csv")


### SAVE MODEL WEIGHTS
list_of_tuples = list(zip(text_clf['vect'].vocabulary_, text_clf['clf'].feature_importances_))
weights = pd.DataFrame(list_of_tuples, columns = ['Word', 'Feature Importance'])
weights.to_csv("data/CTIME/CTIME_Weights.csv")
