import pandas
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import utils
import numpy as np

def parse_diabetes(fp):
    df = pandas.read_csv(fp)
    drop_cols = ['encounter_id', 'patient_nbr', 'payer_code', 'medical_specialty', 'weight', 'race'] # 'diag_1', 'diag_2', 'diag_3']
    numeric_cols = ['time_in_hospital', 'num_procedures', 'num_lab_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
    label_col = ['readmitted']

    df = df.drop(drop_cols, axis=1)
    numeric_df = df[numeric_cols]
    label_df = df[label_col]
    nominal_df = df.drop(numeric_cols + label_col, axis=1)

    numeric = np.array(numeric_df.as_matrix())
    nominal = OneHotEncoder().fit_transform(nominal_df.as_matrix()).toarray()
    labels = np.array(LabelEncoder().fit_transform(label_df.as_matrix().ravel()))

    # first 8 columns are numeric, and the rest are nominal for total of 155 columns
    # 0: 1-14
    # 1: 0-6
    # 2: 1-132
    # 3: 1-81
    # 4: 0-42
    # 5: 0-76
    # 6: 0-21
    # 7: 1-16
    data = np.concatenate((numeric, nominal), axis=1)
    print("data shape: ", data.shape)
    print("labels shape: ", labels.shape)
    return data, labels

if __name__ == "__main__":
    data, labels = parse_diabetes('../data/diabetes/diabetic_data.csv')
    (train_data, train_labels, test_data, test_labels) = utils.split_train_test(data, labels)
    utils.save_protobuf(train_data, train_labels, "diabetes_train")
    utils.save_protobuf(test_data, test_labels, "diabetes_test")

