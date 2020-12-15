import pandas
import utils
import numpy as np


def parse_adult(fp):
    nominal_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    label_col = ['label']
    all_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']
    df = pandas.read_csv(fp, names=all_cols, sep=', ')
    return utils.parse_data_with_pandas(df, [], numeric_cols, label_col, nominal_cols)

if __name__ == "__main__":
    data, labels = parse_adult("../data/adult/adult.data")
    (train_data, train_labels, test_data, test_labels) = utils.split_train_test(data, labels, 0.1)
    utils.save_protobuf(train_data, train_labels, "adult_train")
    utils.save_protobuf(test_data, test_labels, "adult_test")

