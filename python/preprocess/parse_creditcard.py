import pandas
import utils
import numpy as np

def parse_creditcard(fp):
    df = pandas.read_excel(fp, sheet_name='Data')
    df = df[1:]
    drop_cols = []
    numeric_cols = ["X%d" % i for i in range(1,24)]
    label_col = ["Y"]
    return utils.parse_data_with_pandas(df, drop_cols, numeric_cols, label_col, [])

if __name__ == "__main__":
    data, labels = parse_creditcard('../data/creditcard/creditcard.xls')
    (train_data, train_labels, test_data, test_labels) = utils.split_train_test(data, labels)
    utils.save_protobuf(train_data, train_labels, "creditcard_train")
    utils.save_protobuf(test_data, test_labels, "creditcard_test")

