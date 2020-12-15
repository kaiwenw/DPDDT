import numpy as np
from sklearn.preprocessing import OneHotEncoder
import utils

'''
Input:
    there are 34 attributes, missing marked with ?

Output:
    (data, labels)
'''
def parse_dermatology(fp):
    with open(fp) as f:
        text = f.read()

    enc = OneHotEncoder(handle_unknown='error', categories='auto')

    labels = []
    unencoded_data = []
    for line in text.splitlines():
        splitted = line.split(",")
        assert(len(splitted) == 35)

        missing_val = False
        for attr in splitted:
            if attr == '?':
                missing_val = True
                break
        if missing_val:
            continue

        labels.append(int(splitted[34]))
        unencoded_data.append(splitted[:34])

    data = enc.fit_transform(unencoded_data).toarray()
    print(data.shape)
    return (data, np.array(labels))

if __name__ == "__main__":
    (data, labels) = parse_dermatology("../data/dermatology/dermatology.data")
    print(data[:10])
    print(labels[:10])
    (train_data, train_labels, test_data, test_labels) = utils.split_train_test(data, labels)
    utils.save_protobuf(train_data, train_labels, "dermatology_train")
    utils.save_protobuf(test_data, test_labels, "dermatology_test")

