from datetime import datetime
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import tensorflow_hub as hub
import tensorflow as tf
from Preprocessing import process, parse_impression


def display_documents_tsnescatterplot():
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

    # currently using word2vec vectorization method (switch with code from other methods as needed)
    tfidf = TfidfVectorizer(sublinear_tf=True)
    features = tfidf.fit_transform(AllReports['PROCESSED'])
    dict = {val : idx for idx, val in enumerate(tfidf.get_feature_names())}

    preTrainedPath = "MIMIC_ctime_nodict.bin"
    wv = KeyedVectors.load_word2vec_format(preTrainedPath, binary=True)

    x = np.empty([len(AllReports['PROCESSED']),300])
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

    y = AllReports['IC_Mass_Effect'][labelled]
    vec = x[labelled]

    # find the labels of the notes for coloring
    zeroIndices = []
    oneIndices = []
    for i, label in enumerate(y):
        if label == 0: zeroIndices.append(i)
        else: oneIndices.append(i)

    # tsne
    tsne = TSNE(n_components=2, random_state=1023)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(vec)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    # draw figure
    plt.rcParams.update({'font.size': 16})
    plt.scatter(x_coords[zeroIndices], y_coords[zeroIndices], color = "blue", label = "Negative")
    plt.scatter(x_coords[oneIndices], y_coords[oneIndices], color = "red", label = "Positive")
    #plt.legend()
    plt.show()


display_documents_tsnescatterplot()
