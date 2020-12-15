## [Scalable and Provably Accurate Algorithms for Differentially Private Distributed Decision Tree Learning](https://www2.isye.gatech.edu/~fferdinando3/cfp/PPAI20/papers/paper_12.pdf)
## [AAAI Workshop on Privacy-Preserving Artificial Intelligence](https://www2.isye.gatech.edu/~fferdinando3/cfp/PPAI20/) @ AAAI-2020
## Citation
```
K. W. Wang, T. Dick, and M.-F. Balcan. Scalable and provably accurate algorithms for differentially private distributed decision tree learning. In AAAI Workshop on Privacy-Preserving Artificial Intelligence, 2020

@inproceedings{Wang2020ScalableAP,
  title={Scalable and provably accurate algorithms for differentially private distributed decision tree learning},
  author={Kai Wen Wang and Travis Dick and Maria-Florina Balcan},
  booktitle={AAAI Workshop on Privacy-Preserving Artificial Intelligence},
  year={2020}
}
```

## Setup on Ubuntu 18.04 LTS
The DP-TopDown algorithm is implemented in C++, so we need `GCC` and `CMake` to 
compile it. \
First make sure `GCC v7.4` is installed. Run
```
sudo apt update && sudo apt install -y build-essential
```

Then, install `CMake` (at least version 3.12). \
[Download](https://cmake.org/download/), unzip, and cd into the folder. \
Then run
```
./bootstrap && make -j10
sudo make install
```


### [Install `protobuf` (version 3.6.1)](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md)
For reference: \
Run the following commands (taken from the link). \
Note: the make commands can take a while. 
```
sudo apt-get install -y autoconf automake libtool curl make g++ unzip
git clone --branch v3.6.1 --recursive https://github.com/protocolbuffers/protobuf.git
cd protobuf
./autogen.sh
./configure
make -j4
make check -j4
sudo make install -j4
sudo ldconfig
```

## Build
To build, simply run the following command.
```
mkdir build && cd build
cmake .. && make -j10
```

If this is the first time building, the build can fail because protobuf 
headers are out of date. To update the headers, run
```
For C++:
protoc -I=./data/ --cpp_out=third_party/protobuf/ ./data/dataset.proto

For Python: 
protoc -I=./data/ --python_out=python/ ./data/dataset.proto
```
You may need to install `protobuf` for Python.


## Running
`/cpp/single_run.cpp` is our main script. 
To parallelize the process, we used AWS Batch. 
It takes parameters via environment variables, as described below.

| ENVVAR                | DEFAULT VALUE                       |
|:---------------------:|:-----------------------------------:|
| SEED                  | 42                                  | 
| TRAINING_FRACTION     | 1                                   | 
| LEAF_PRIVACY_FRACTION | 0.5                                 |
| DATASET               | adult                               |
| BUDGET_FN             | decay                               |

### Running AWS Batch
We ran our experiments with a docker image in AWS Batch with Dockerfile in 
`aws/Dockerfile` which calls the script `aws/run.sh`. In the script, the seed
environment variable is set to be `AWS_BATCH_JOB_ARRAY_INDEX`, automatically
set by AWS Batch. In addition, we must specify an additional environment variable
`OUT_DIRECTORY` to store the outputted `.csv` files. 
There is an optional parameter `START_INDEX` which offsets the seed, 
since `AWS_BATCH_JOB_ARRAY_INDEX` ranges from 0 to `ARRAY_SIZE`. 

To run a bunch of experiments via a script, we use `aws/submit_job.py`.


### Testing Locally
We can also run `cpp/single_run.cpp` locally. An example test run would be
```
SEED=42 TRAINING_FRACTION=1 LEAF_PRIVACY_FRACTION=0.5 DATASET=adult BUDGET_FN=decay ./single_run
```
This run would sweep across all values of alpha and all private split algorithms.

## References to tested datasets
Our paper includes experiments on the following datasets.

| DATASET                | URL                                 |
|:---------------------: |:-----------------------------------:|
| MNIST                  | http://yann.lecun.com/exdb/mnist/ | 
| Adult                  | https://archive.ics.uci.edu/ml/datasets/Adult | 
| Bank                   | https://archive.ics.uci.edu/ml/datasets/Bank+Marketing |
| Creditcard             | https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients |
| KDDCup 1999            | http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html |
| Avazu CTR              | https://www.kaggle.com/c/avazu-ctr-prediction/ |
| Skin                   | https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation |

After preprocessing, our datasets are in `data.zip`. 

### Process for testing a new dataset
To train and test on a new dataset named `X`, do the following steps:
1) Create protobufs `X_train` and `X_test` with save functions in `cpp/utils.h` or `python/utils.py`
2) Include new protobufs into `./data`
3) Define new splitting class in `/cpp/split.h` and run scripts

## Formatting
Format C++ files with
```
clang-format -i path_to_cpp_file
```
and format Python files with
```
black path_to_python_file
```

