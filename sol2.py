    def __init__(self, word2vecModel=None):
        self.word2vecModel = word2vecModel
        i = 0
        if self.word2vecModel:
            vecs = np.zeros((len(word2vecModel.vocab), 300), dtype=np.float32)
            for word in self.word2vecModel.vocab:
                vecs[i] = self.word2vecModel.get_vector(word)
                i+=1
            self.averageVec = np.mean(vecs, axis=0)

    # function which returns a vector for a given word (should also deal with unseen words)
    def getWordVector(self, word):
        vec = self.averageVec
        if word in self.word2vecModel.vocab:
            vec = self.word2vecModel.get_vector(word)
        return vec

    def extract_struct(self):
        for i, doc in enumerate(self.data):
            sentences = [x for x in self.sentences(doc["text"])]

            j = 0
            inside = False
            for token, token_begin, token_end in doc["tokens"]:
                value = "O"
                for ann, ann_begin, ann_end in sentences:
                    if token_begin < ann_begin:
                        value = "O"
                        inside = False
                    elif token_end <= ann_end:
                        value = "I" if token_begin > ann_begin else "B"
                        inside = True
                    else:
                        continue
                    break

                self.data_features[i][j]["structural"] = value
                j += 1

    def evaluate(self):
        labels = ["B","I","O"]
        self.true_vals = [label for features, label in self.flat(self.test_data)]
        res = [["Precision", "Recall", "F1-Score"],[0,0,0]]
        res[1][0] = sklearn.metrics.precision_score(self.true_vals, self.predictions, labels=labels, average="macro")
        res[1][1] = sklearn.metrics.recall_score(self.true_vals, self.predictions,labels=labels, average="macro")
        res[1][2] = sklearn.metrics.f1_score(self.true_vals, self.predictions,labels=labels, average="macro")
        return res
