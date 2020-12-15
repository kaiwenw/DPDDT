import utils
import numpy as np


def parse_skin(fp):
    with open(fp) as f:
        text = f.read()

    data = []
    labels = []
    for line in text.splitlines():
        splitted = line.split('\t')
        data.append(np.array(splitted[:3], dtype=int))
        labels.append(splitted[-1])
    return np.array(data), np.array(labels, dtype=int)

if __name__ == "__main__":
    data, labels = parse_skin("../data/skin/Skin_NonSkin.txt")
    (train_data, train_labels, test_data, test_labels) = utils.split_train_test(data, labels, 0.1)
    utils.save_protobuf(train_data, train_labels, "skin_train")
    utils.save_protobuf(test_data, test_labels, "skin_test")

