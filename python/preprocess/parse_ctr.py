import pandas
import utils


def parse_ctr(fp):
    df = pandas.read_csv(fp)
    # even amount of 0/1
    ones = df.loc[df["click"] == 1]
    zeros = df.loc[df["click"] == 0]
    concat_df = pandas.concat([zeros.sample(550000), ones.sample(550000)])

    final_df = concat_df[
        [
            "click",
            "hour",
            "C1",
            "banner_pos",
            "site_category",
            "app_category",
            "device_type",
            "device_conn_type",
            "C14",
            "C15",
            "C16",
            "C17",
            "C18",
            "C19",
            "C20",
            "C21",
        ]
    ]

    nominal_cols = ["site_category", "app_category", "device_type", "device_conn_type"]
    numeric_cols = [
        "hour",
        "banner_pos",
        "C1",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
    ]
    label_col = ["click"]
    return utils.parse_data_with_pandas(
        final_df, [], numeric_cols, label_col, nominal_cols
    )


if __name__ == "__main__":
    data, labels = parse_ctr("../data/ctr/train")
    (train_data, train_labels, test_data, test_labels) = utils.split_train_test(
        data, labels, 0.1
    )
    utils.save_protobuf(train_data, train_labels, "ctr_train")
    utils.save_protobuf(test_data, test_labels, "ctr_test")
