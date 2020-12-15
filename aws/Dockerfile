FROM ubuntu:latest

RUN apt-get update

# Make sure we have GCC, GIT, and wget
RUN apt-get install -y build-essential git wget

# Install python, pip, and the aws cli
RUN apt-get install -y python3-pip
RUN pip3 install awscli --upgrade --user

WORKDIR /user/

# Compile and install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.15.2/cmake-3.15.2.tar.gz
RUN tar xvfz cmake-3.15.2.tar.gz
RUN (cd cmake-3.15.2 && ./bootstrap && make && make install)

# Compile and install protobuf
RUN apt-get install -y autoconf automake libtool curl make g++ unzip
RUN git clone --branch v3.6.1 --recursive https://github.com/protocolbuffers/protobuf.git
RUN (cd protobuf && ./autogen.sh && ./configure && make && make check && make install && ldconfig)

# Copy and compile decision tree code
ADD private-DPDDT private-DPDDT
WORKDIR /user/private-DPDDT
RUN protoc -I=./data/ --cpp_out=third_party/protobuf/ ./data/dataset.proto
RUN mkdir build
WORKDIR /user/private-DPDDT/build
RUN cmake ..
RUN make
WORKDIR /user/

# Copy the datasets into the image
ADD data /user/private-DPDDT/data

# Copy run script and make it executable
ADD run.sh /user/run.sh
RUN chmod u+x /user/run.sh

CMD ["/user/run.sh"]
