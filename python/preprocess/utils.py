import numpy as np
import dataset_pb2
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def shuffle_data(data, labels):
    rand_idx = np.arange(len(data))
    data = data[rand_idx]
    labels = labels[rand_idx]
    return data, labels


def sec2str(secs):
    hours = secs // 3600;
    secs = secs - hours * 3600;
    minutes = secs // 60;
    secs = secs - minutes * 60;
    return "%dH %dM %dS" % (hours, minutes, secs)

'''
splits (data, labels) into train, test datasets
'''
def split_train_test(data, labels, test_ratio=0.1):
    data = np.array(data)
    labels = np.array(labels)
    assert(len(data) == len(labels))

    rand_idx = np.arange(len(data))
    np.random.shuffle(rand_idx)

    data = data[rand_idx]
    labels = labels[rand_idx]

    split_idx = int(len(data) * test_ratio)
    test_data = data[:split_idx]
    test_labels = labels[:split_idx]

    train_data = data[split_idx:]
    train_labels = labels[split_idx:]
    return (train_data, train_labels, test_data, test_labels)

# parses data with pandas putting numeric columns first
def parse_data_with_pandas(df, drop_cols, numeric_cols, label_col, nominal_cols):
    numeric_df = df[numeric_cols]
    label_df = df[label_col]
    if nominal_cols != []:
        nominal_df = df[nominal_cols]

    numeric = np.array(numeric_df.values)
    if nominal_cols != []:
        nominal = OneHotEncoder(categories='auto').fit_transform(nominal_df.values).toarray()
    labels = np.array(LabelEncoder().fit_transform(label_df.values.ravel()))

    for col in numeric_df:
        uniques = sorted(numeric_df[col].unique())
        print(col, ": ", uniques[0], " to ", uniques[-1])

    data = numeric
    if nominal_cols != []:
        data = np.concatenate((numeric, nominal), axis=1)
    print("Data has shape ", data.shape)
    print("Labels has shape ", labels.shape)
    return data, labels

'''
data is a 2d array (vector<vector<float>>)
labels is 1d array (vector<int>)
fp is filepath to save protobuf
'''
def save_protobuf(data, labels, fp):
    dataset = dataset_pb2.Dataset()
    dataset.numRows = len(data)
    dataset.numCols = len(data[0])
    distinct_labels = set()
    for i in range(dataset.numRows):
        assert(len(data[i]) == len(data[0]))
        dataset.data.extend(data[i])
        dataset.labels.append(labels[i])
        distinct_labels.add(labels[i])

    dataset.numLabels = len(distinct_labels)
    with open(fp, "wb") as f:
        f.write(dataset.SerializeToString())

    print("Successfully written %d x %d to %s" % (dataset.numRows, dataset.numCols, fp))

'''
read protobuf from filepath
'''
def read_protobuf(fp):
    dataset = dataset_pb2.Dataset()
    try:
        with open(fp, "rb") as f:
            dataset.ParseFromString(f.read())
            print("Num rows: %d, num cols: %d" % (dataset.numRows, dataset.numCols))
            print("Num distinct labels: %d" % (dataset.numLabels))
            assert(len(dataset.data) == dataset.numRows * dataset.numCols)
            data = []
            for i in range(dataset.numRows):
                data.append(dataset.data[i*dataset.numCols : (i+1)*dataset.numCols])
        return (data, dataset.labels)

    except IOError:
        print("IO ERROR!")
        return (None, None)
