import numpy as np 
from sklearn.preprocessing import OneHotEncoder
import utils

'''
Attribute Values:
0: parents        usual, pretentious, great_pret
1: has_nurs       proper, less_proper, improper, critical, very_crit
2: form           complete, completed, incomplete, foster
3: children       1, 2, 3, more
4: housing        convenient, less_conv, critical
5: finance        convenient, inconv
6: social         non-prob, slightly_prob, problematic
7: health         recommended, priority, not_recom

8: Class Distribution (number of instances per class)
   class        N         N[%]
   ------------------------------
   not_recom    4320   (33.333 %)
   recommend       2   ( 0.015 %)
   very_recom    328   ( 2.531 %)
   priority     4266   (32.917 %)
   spec_prior   4044   (31.204 %)

Output: (data, labels)

where each attribute is one-hot encoded and labels is 0-4 from not_recom to spec_prior
'''
label_map = {
    'not_recom': 0,
    'recommend': 1,
    'very_recom': 2,
    'priority': 3,
    'spec_prior': 4,
        }

def parse_nursery(fp):
    with open(fp) as f:
        text = f.read()
    
    enc = OneHotEncoder(handle_unknown='error')

    unencoded_data = []
    labels = []
    for line in text.splitlines():
        splitted = line.split(",")
        assert(len(splitted) == 9)
        unencoded_data.append(splitted[:-1])
        labels.append(label_map[splitted[8]])

    data = enc.fit_transform(unencoded_data).toarray()
    assert(data.shape == (len(labels), 3+5+4+4+3+2+3+3))
    return (data, np.array(labels))
    
if __name__ == '__main__':
    (data, labels) = parse_nursery('../data/nursery/nursery.data')
    print(data[:10])
    print(labels[:10])
    (train_data, train_labels, test_data, test_labels) = utils.split_train_test(data, labels)
    utils.save_protobuf(train_data, train_labels, "nursery_train")
    utils.save_protobuf(test_data, test_labels, "nursery_test")

