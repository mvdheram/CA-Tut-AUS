#!/usr/bin/env python3

from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from pprint import pprint
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import glob
import nltk
import nltk.classify
import numpy as np
import os
import pycrfsuite
import random
import re
import sklearn.metrics

# Base class for models
class GenModel(object):
    def __init__(self, word2vecModel=None):
        self.word2vecModel = word2vecModel

    # function which returns a vector for a given word (should also deal with unseen words)
    def getWordVector(self, word):
        # TODO Implement logic to extract vector for a given word
        return None

    # Tokenize text, include token position information.
    def tokens(self, txt):
        tokens = word_tokenize(txt)
        offset = 0
        for token in tokens:
            offset = txt.find(token, offset)
            yield token, offset, offset+len(token)
            offset += len(token)

    def sentences(self, txt):
        tokens = nltk.sent_tokenize(txt)
        offset = 0
        for token in tokens:
            offset = txt.find(token, offset)
            yield token, offset, offset+len(token)
            offset += len(token)

    # Load the texts and annotations.
    def load_data(self, data_path):
        txt_files = glob.glob(data_path + "/*.txt")

        self.data = []
        for txt_file in txt_files:
            doc = {}

            with open(txt_file) as f:
                doc["text"] = "".join(f.readlines())
                doc["tokens"] = [span for span in self.tokens(doc["text"])]

            with open(txt_file.replace(".txt", ".ann")) as f:
                doc["annotations"] = [[int(x) for x in line.split("\t")[1:3]] for line in f.readlines()]

            self.data.append(doc)

    def sample_loaded_data(self):
        pprint(random.choice(self.data))

    # Make B I O labels.
    def prepare_labels(self):
        self.data_labels = []
        for doc in self.data:
            labels = []
            inside = False
            for token, token_begin, token_end in doc["tokens"]:
                label = "O"
                # TASK BEGIN
                # use doc["annotations"]
                for ann_begin, ann_end in doc["annotations"]:
                   pass
                    # TASK BEGIN
                    # TODO Implement logic to assign IOB label to the current token i.e label = "I"
                    # TASK END
                # TASK END

                labels.append(label)
            self.data_labels.append(labels)

    def sample_labels(self):
        i = random.randint(0, len(self.data) - 1)
        pprint([x for x in zip([t[0] for t in self.data[i]["tokens"]], self.data_labels[i])])

    # Extract token length feature.
    def extract_length(self):
        for i, doc in enumerate(self.data):
            j = 0
            for token, t_b, t_e in doc["tokens"]:
                self.data_features[i][j]["length"] = len(token)
                j += 1

    # Extract bag-of-words features.
    def extract_bow(self):
        for i, doc in enumerate(self.data):
            bag = {}
            for token, t_b, t_e in doc["tokens"]:
                bag[token] = bag.get(token, 0) + 1
            max_bag = max(bag.values())
            bag = {k: v/max_bag for k, v in bag.items()}

            j = 0
            for token, t_b, t_e in doc["tokens"]:
                self.data_features[i][j]["bow"] = bag.get(token, 0)
                j += 1

    # Extract Part-of-Speech features.
    def extract_pos(self):
        for i, doc in enumerate(self.data):
            pos_tags = pos_tag([token for token, t_b, t_e in doc["tokens"]])
            j = 0
            for _, tag in pos_tags:
                self.data_features[i][j]["pos"] = tag
                j += 1

    # Extract token embeddings.
    def extract_embeddings(self):
        if self.word2vecModel:
            for i, doc in enumerate(self.data):
                j = 0
                for token, t_b, t_e in doc["tokens"]:
                    for k, val in enumerate(self.getWordVector(token)):
                        self.data_features[i][j]["w2v." + str(k)] = val
                    j += 1

    # Extract structural features.
    def extract_struct(self):
        for i, doc in enumerate(self.data):
            sentences = [x for x in self.sentences(doc["text"])]

            j = 0
            for token, token_begin, token_end in doc["tokens"]:
                value = None
                
                # TASK BEGIN
                # TODO Implement logic to assign IOB label to the current token i.e label = "I"
                # TASK END

                self.data_features[i][j]["structural"] = value
                j += 1

    def copy_features(self, features, copy_to, copy_from, prefix):
        # Copy all features from features[copy_from] to features[copy_to]
        # except the ones already copied by this function.
        for feature, value in features[copy_from].items():
            if feature[0] != ":":
                features[copy_to][":" + prefix + ":" + feature] = value

    def add_contextual_features(self, n):
        # For every token, add features from (n) previous and (n) next tokens
        # in the same document to it.
        for i, doc in enumerate(self.data_features):
            for j in range(0, len(doc)):
                for k in range(1, n+1):
                    if j - k >= 0:
                        self.copy_features(self.data_features[i], j, j - k, str(-k))
                    if j + k < len(doc):
                        self.copy_features(self.data_features[i], j, j + k, str(k))

    def extract_features(self):
        self.data_features = []
        for doc in self.data:
            self.data_features.append([{} for token in doc["tokens"]])

        self.extract_length()
        self.extract_bow()
        self.extract_struct()
        self.extract_embeddings()

    def sample_features(self):
        i = random.randint(0, len(self.data) - 1)
        pprint([x for x in zip([t[0] for t in self.data[i]["tokens"]], self.data_features[i][0:25])])

    def split_dataset(self):
        self.train_data, self.test_data = train_test_split([x for x in zip(self.data_features, self.data_labels)], test_size=0.2)

    def sample_prediction(self):
        i = random.randint(0, len(self.data) - 1)
        print("Expected; Actual")
        pprint([x for x in zip(self.data_labels[0], self.predictions[0:len(self.data_labels[0])])])

    def flat(self, data):
        features = []
        labels = []
        for doc_features, doc_labels in data:
            features.extend(doc_features)
            labels.extend(doc_labels)
        return zip(features, labels)

    def filter_features(self, flat_data, wanted):
        r = re.compile("^(:[^:]+:)?(" + "|".join(wanted) + ")")
        return [({k: v for k, v in features.items() if r.match(k)}, label) for features, label in flat_data]

    def evaluate(self):
        labels = ["B","I","O"]
        self.true_vals = [label for features, label in self.flat(self.test_data)]
        res = [["Precision", "Recall", "F1-Score"],[0,0,0]]
        # TASK BEGIN
        # TODO: Implement logic to compute Precision, Recall and F1-Score, i.e res[1][0] = calcPrec
        # TASK END
        return res

    def printEval(self, evalData):
        t = PrettyTable(evalData[0])
        for i, x in enumerate(evalData):
            if(i>0):
                t.add_row(evalData[i])
        print(t)

    # This is not used in python notebook.
    def pipeline(self, data_path):
        self.load_data(data_path)
        self.sample_loaded_data()
        self.prepare_labels()
        self.sample_labels()
        self.extract_features()
        self.sample_features()
        self.split_dataset()
        self.train(["length", "bow", "pos", "w2v", "structural"])
        self.predict()
        self.sample_prediction()
        self.printEval(self.evaluate())

class SvmModel(GenModel):
    def extract_features(self):
        super().extract_features()
        self.extract_pos()

    def train(self, wanted_features):
        self.svm = nltk.classify.SklearnClassifier(LinearSVC())
        self.svm.train(self.filter_features(self.flat(self.train_data), wanted_features))

    def predict(self):
        self.predictions = self.svm.classify_many([features for features, label in self.flat(self.test_data)])

class CrfModel(GenModel):
    def extract_features(self):
        super().extract_features()
        self.add_contextual_features(2)

    def features_for_crf(self, features):
        return [k + "=" + str(v) for k, v in features.items()]

    def train(self, wanted_features):
        trainer = pycrfsuite.Trainer(verbose=False)
        for features, label in self.filter_features(self.flat(self.train_data), wanted_features):
            trainer.append([self.features_for_crf(features)], [label])
        trainer.train('1.crfsuite')
        self.crf = pycrfsuite.Tagger()
        self.crf.open('1.crfsuite')

    def predict(self):
        self.predictions = [self.crf.tag([self.features_for_crf(features)])[0] for features, label in self.flat(self.test_data)]

if __name__ == "__main__":
    model = SvmModel()
    model.pipeline("data")

    model = CrfModel()
    model.pipeline("data")
