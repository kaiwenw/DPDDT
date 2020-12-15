/** @file utils.h
 *  @brief Helper methods, mainly dealing with data related tasks such as Protobuf.
 */

#ifndef D3T_UTILS_H
#define D3T_UTILS_H

#include <fstream>
#include <random>

#include "../third_party/protobuf/dataset.pb.h"

#define WARNING_PRINTF(fmt, args...) \
    fprintf(stdout, "WARNING: %s:%d:%s: " fmt, __FILE__, __LINE__, __func__, ##args)

#if defined(DEBUG) && DEBUG > 0
#define INFO_PRINTF(fmt, args...) \
    fprintf(stdout, "INFO: %s:%d:%s: " fmt, __FILE__, __LINE__, __func__, ##args)
#else
#define INFO_PRINTF(fmt, args...)
#endif

#if defined(DEBUG) && DEBUG > 1
#define DEBUG_PRINTF(fmt, args...) \
    fprintf(stdout, "DEBUG: %s:%d:%s: " fmt, __FILE__, __LINE__, __func__, ##args)
#else
#define DEBUG_PRINTF(fmt, args...)
#endif

bool floatEq(float a, float b)
{
    return std::abs(a - b) < 1e-6;
}

/*
 * Partition up (data, labels) into parts with size specified by partitionSizes
 */
std::tuple<std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<int>>> partitionData(
    const std::vector<std::vector<float>>& data,
    const std::vector<int>& labels,
    const std::vector<int>& partitionSizes)
{
    std::vector<std::vector<std::vector<float>>> partitionData;
    std::vector<std::vector<int>> partitionLabels;
    int acc = 0;
    for (int partitionSize : partitionSizes) {
        std::vector<std::vector<float>> dataSlice(
            data.begin() + acc, data.begin() + acc + partitionSize);
        std::vector<int> labelSlice(labels.begin() + acc, labels.begin() + acc + partitionSize);
        partitionData.push_back(dataSlice);
        partitionLabels.push_back(labelSlice);
        acc += partitionSize;
    }
    return std::make_tuple(partitionData, partitionLabels);
}

std::string sec2str(int secs)
{
    int hours = secs / 3600;
    int minutes = (secs - 3600 * hours) / 60;
    secs = secs % 60;
    char buf[100];
    sprintf(buf, "%dH %dM %dS", hours, minutes, secs);
    return std::string(buf);
}

#if defined(DEBUG) && DEBUG > 1
template <typename T>
void printLabelCounts(const std::unordered_map<int, T>& labelCounts)
{
    std::vector<int> labels;
    for (auto& label2count : labelCounts) {
        labels.push_back(label2count.first);
    }
    std::sort(labels.begin(), labels.end());
    T sum = 0;
    for (int label : labels) {
        T curCount = labelCounts.at(label);
        std::cout << label << ":\t" << curCount << std::endl;
        sum += curCount;
    }
    std::cout << "total: " << sum << std::endl;
}

template <typename T>
void printSplitLabelCounts(const std::unordered_map<int, std::unordered_map<int, T>>& splitLabelCounts)
{
    std::vector<int> splitVals;
    for (auto& splitVal2count : splitLabelCounts) {
        splitVals.push_back(splitVal2count.first);
    }
    std::sort(splitVals.begin(), splitVals.end());
    for (int splitVal : splitVals) {
        std::cout << "for split " << splitVal << std::endl;
        printLabelCounts(splitLabelCounts.at(splitVal));
    }
}
#endif

/*
 * shuffles data_ and labels_
 */
void shuffleData(int seed, std::vector<std::vector<float>>& data, std::vector<int>& labels)
{
    assert(data.size() == labels.size());
    std::mt19937 rng(seed);
    shuffle(data.begin(), data.end(), rng);
    rng.seed(seed);
    shuffle(labels.begin(), labels.end(), rng);
}

/*
Returns the number of distinct labels_.
*/
size_t parseProtobuf(std::vector<std::vector<float>>& data,
                     std::vector<int>& labels,
                     const std::string& fp,
                     int seed,
                     float fraction)
{
    protoDataset::Dataset myData;
    std::fstream input(fp, std::ios::in | std::ios::binary);
    if (!input) {
        std::cerr << fp << " not found!" << std::endl;
        return 0;
    }
    else if (!myData.ParseFromIstream(&input)) {
        std::cerr << "failed to parse" << fp << std::endl;
        return 0;
    }
    size_t numCols = myData.numcols();
    size_t numRows = myData.numrows();
    size_t numLabels = myData.numlabels();
    if (myData.data().size() != (int)(numCols*numRows)) {
        std::cerr << "data size: " << myData.data().size() << " numCols * numRows: " << (numCols*numRows) << std::endl;
        assert(false);
    }

    size_t getNumRows = (size_t) (numRows * fraction);
    if (getNumRows > numRows) {
        std::cout << "getNumRows: " << getNumRows << " is larger than numRows: " << numRows << std::endl;
        std::cout << "defaulting to numRows" << std::endl;
        getNumRows = numRows;
    }

    // create a random permutation
    std::mt19937 rng(seed);
    std::vector<size_t> randIdx;
    for (size_t i = 0; i < numRows; i++) {
        randIdx.push_back(i);
    }
    shuffle(randIdx.begin(), randIdx.end(), rng);

    // use the first getNumRow's number of the permutation as data
    for (size_t r = 0; r < getNumRows; r++) {
        size_t idx = randIdx[r];
        std::vector<float> myRow = std::vector<float>();
        copy(myData.data().begin() + idx * numCols,
             myData.data().begin() + (idx + 1) * numCols, back_inserter(myRow));
        data.push_back(myRow);
        labels.push_back(myData.labels().Get(idx));
    }

    printf(
        "Successfully parsed %lu x %lu data_ and %lu labels_! There are %lu "
        "distinct labels_ \n",
        data.size(), data[0].size(), labels.size(), numLabels);
    assert(data.size() == labels.size() && data.size() == getNumRows && data[0].size() == numCols);
    return numLabels;
}

/*
Returns true if successful.
*/
bool saveProtobuf(const std::vector<std::vector<float>>& data,
                  const std::vector<int>& labels,
                  const char* fp)
{
    protoDataset::Dataset myData;
    myData.set_numcols((int)data[0].size());
    myData.set_numrows((int)data.size());
    for (size_t r = 0; r < data.size(); r++) {
        for (size_t c = 0; c < data[0].size(); c++) {
            myData.add_data(data[r][c]);
        }
        myData.add_labels(labels[r]);
    }

    std::fstream output(fp, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!myData.SerializeToOstream(&output)) {
        std::cerr << "failed to write dataset to " << fp << std::endl;
        return false;
    }
    printf("Successful!\n");
    return true;
}

#endif // D3T_UTILS_H
