import utils
import numpy as np
import struct

def parse_images(fp):
    with open(fp, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">"))
        data = data.reshape((size, nrows*ncols)).astype(float)
        return data

def parse_labels(fp):
    with open(fp, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">")).astype(int)
        return data

if __name__ == "__main__":
    train_data = parse_images("../raw_data/mnist/mnist500k-patterns")
    train_labels = parse_labels("../raw_data/mnist/mnist500k-labels")
    utils.save_protobuf(train_data, train_labels, "mnist500k_train")

    # test_data = parse_images("../data/mnist/test10k-patterns")
    # test_labels = parse_labels("../data/mnist/test10k-labels")
    # utils.save_protobuf(test_data, test_labels, "mnist60k_test")

