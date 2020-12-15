import utils
import pandas
import numpy as np

raw_name = """
duration: continuous.
protocol_type: symbolic.
service: symbolic.
flag: symbolic.
src_bytes: continuous.
dst_bytes: continuous.
land: symbolic.
wrong_fragment: continuous.
urgent: continuous.
hot: continuous.
num_failed_logins: continuous.
logged_in: symbolic.
num_compromised: continuous.
root_shell: continuous.
su_attempted: continuous.
num_root: continuous.
num_file_creations: continuous.
num_shells: continuous.
num_access_files: continuous.
num_outbound_cmds: continuous.
is_host_login: symbolic.
is_guest_login: symbolic.
count: continuous.
srv_count: continuous.
serror_rate: continuous.
srv_serror_rate: continuous.
rerror_rate: continuous.
srv_rerror_rate: continuous.
same_srv_rate: continuous.
diff_srv_rate: continuous.
srv_diff_host_rate: continuous.
dst_host_count: continuous.
dst_host_srv_count: continuous.
dst_host_same_srv_rate: continuous.
dst_host_diff_srv_rate: continuous.
dst_host_same_src_port_rate: continuous.
dst_host_srv_diff_host_rate: continuous.
dst_host_serror_rate: continuous.
dst_host_srv_serror_rate: continuous.
dst_host_rerror_rate: continuous.
dst_host_srv_rerror_rate: continuous.
"""

def parse_kddcup(fp):
    all_cols = []
    numeric_cols = []
    nominal_cols = []
    label_col = ['label']
    for line in raw_name.splitlines()[1:]:
        col = line.split(":")[0]
        col_type = line.split(":")[1][1:-1]
        if (col_type == 'continuous'):
            numeric_cols.append(col)
        elif (col_type == 'symbolic'):
            nominal_cols.append(col)
        else:
            assert(False)
        all_cols.append(col)

    df = pandas.read_csv(fp, names=all_cols+label_col)
    return utils.parse_data_with_pandas(df, [], numeric_cols, label_col, nominal_cols)

if __name__ == "__main__":
    data, labels = parse_kddcup("../data/kddcup/kddcup.data_10_percent")
    (train_data, train_labels, test_data, test_labels) = utils.split_train_test(data, labels, 0.1)
    utils.save_protobuf(train_data, train_labels, "kddcup_train")
    utils.save_protobuf(test_data, test_labels, "kddcup_test")
