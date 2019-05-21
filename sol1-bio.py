def prepare_labels(self):
    self.data_labels = []
    for doc in self.data:
        labels = []
        inside = False
        for token, token_begin, token_end in doc["tokens"]:
            label = "O"
            for ann_begin, ann_end in doc["annotations"]:
                if token_begin < ann_begin:
                    label = "O"
                    inside = False
                elif token_end <= ann_end:
                    label = "I" if inside else "B"
                    inside = True
                else:
                    continue
                break

            labels.append(label)
        self.data_labels.append(labels)
