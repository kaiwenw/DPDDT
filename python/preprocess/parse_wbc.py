import numpy as np
import utils

'''
Input:
Feature 1 is id number, 
Feature 2-10 are ranging from 1-10
Feature 11 is the class, 2 being benign and 4 being malignant

Output: (data, labels)
where data is a list[list[float]] and labels is a list[int]

note the six patients with ? are removed
'''
def parse_wbc(fp):
    with open(fp) as f:
        text = f.read()

    data = []
    labels = []
    for line in text.splitlines():
        splitted = line.split(",")
        assert(len(splitted) == 11)
        has_question = False
        for i in range(len(splitted)):
            if splitted[i] == '?':
                has_question = True
                break
            splitted[i] = int(splitted[i])
        if has_question:
            continue
        features = splitted[:10]
        label = splitted[10]
        data.append(features)
        labels.append(label)
    return (data, labels)
    
if __name__ == "__main__":
    (data, labels) = parse_wbc("./bcw/breast-cancer-wisconsin.data")
    print(len(data), len(labels))
    print(data[:10], labels[:10])
    (train_data, train_labels, test_data, test_labels) = utils.split_train_test(data, labels)
    utils.save_protobuf(train_data, train_labels, "wbc_train")
    utils.save_protobuf(test_data, test_labels, "wbc_test")
