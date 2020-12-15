import pandas
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import utils
import numpy as np

# pdays is special can be -1
def parse_bank(fp):
    df = pandas.read_csv(fp, sep=';')
    drop_cols = []
    numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    nominal_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    label_col = ['y']
    return utils.parse_data_with_pandas(df, drop_cols, numeric_cols, label_col, nominal_cols)

if __name__ == "__main__":
    data, labels = parse_bank('../data/bank/bank-full.csv')
    (train_data, train_labels, test_data, test_labels) = utils.split_train_test(data, labels)
    utils.save_protobuf(train_data, train_labels, "bank_train")
    utils.save_protobuf(test_data, test_labels, "bank_test")

