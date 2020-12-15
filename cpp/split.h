/** @file split.h
 *  @brief Encapsulating the notion of splitting functions.
 *         Builds the splitting class for each dataset.
 */

#ifndef D3T_SPLIT_H
#define D3T_SPLIT_H

#include "utils.h"
#include <cmath>
#include <unordered_map>
#include <vector>

class Split {
public:
    Split(std::vector<int> labels) : labels(labels)
    {
        id = globalCounter;
        globalCounter++;
    }

    virtual ~Split() = default;
    virtual int applySplit(const std::vector<float>& datum) const = 0;
    virtual std::string toString() const = 0;

    std::vector<int> labels;
    int id;
    inline static int globalCounter = 0;
};

class ThresholdSplit : public Split {
public:
    ThresholdSplit(const std::vector<int>& attributes, float threshold)
        : Split({0, 1}), attributes(attributes), threshold(threshold)
    {
    }

    /*
     * applies split over the average of attributes for threshold
     */
    int applySplit(const std::vector<float>& datum) const override
    {
        float sum = 0.0;
        for (size_t i = 0; i < attributes.size(); i++) {
            sum += datum[attributes[i]];
        }
        return sum <= threshold * attributes.size();
    }

    std::string toString() const override
    {
        std::string result;
        for (int attr : attributes) {
            result += std::to_string(attr) + ",";
        }
        result += "\t threshold at " + std::to_string(threshold);
        return result;
    }

    std::vector<int> attributes;
    float threshold;
};

/*
 * Given x,y return 1 if y <= mx + b and 0 otherwise
 */
class ObliqueSplit : public Split {
public:
    ObliqueSplit(const std::vector<int>& xs, const std::vector<int>& ys, float m, float b)
        : Split({0, 1}), xs(xs), ys(ys), m(m), b(b)
    {
    }

    int applySplit(const std::vector<float>& datum) const override
    {
        float x = 0.0;
        float y = 0.0;
        for (int xAttr : xs) {
            x += datum[xAttr];
        }
        x = x / xs.size();
        for (int yAttr : ys) {
            y += datum[yAttr];
        }
        y = y / ys.size();
        return y <= m * x + b;
    }

    std::string toString() const override
    {
        return "";
    }
    std::vector<int> xs;
    std::vector<int> ys;
    float m;
    float b;
};

void addContinuous(std::vector<std::shared_ptr<Split>>& splittingClass,
                   const std::vector<int>& attributes,
                   float low,
                   float high,
                   int numThresholds)
{
    float jumpSize = (high - low) / numThresholds;
    for (int i = 0; i < numThresholds; i++) {
        std::shared_ptr<Split> splitFn = std::static_pointer_cast<Split>(
            std::make_shared<ThresholdSplit>(attributes, (i + 0.5) * jumpSize));
        splittingClass.push_back(splitFn);
    }
}

/*
 * Assumes pixel values from 0 to 255
 * Image is size width x height, create blocks of size blockWidth x blockHeight 
 * each block having numThresholds thresholds
 * that has values evenly spread from 0 to 255
 */
std::vector<std::shared_ptr<Split>> ImageBlockSplittingClass(
    int width, int height, int blockWidth, int blockHeight, int numThresholds)
{
    std::vector<std::shared_ptr<Split>> splittingClass;
    assert(width % blockWidth == 0);
    assert(height % blockHeight == 0);
    for (int blockRow = 0; blockRow < height / blockHeight; blockRow++) {
        for (int blockCol = 0; blockCol < width / blockWidth; blockCol++) {
            // get the block
            std::vector<int> attributes = {};
            for (int innerRow = 0; innerRow < blockHeight; innerRow++) {
                for (int innerCol = 0; innerCol < blockWidth; innerCol++) {
                    int row = blockRow * blockHeight + innerRow;
                    int col = blockCol * blockWidth + innerCol;
                    attributes.push_back(col * width + row);
                }
            }
            addContinuous(splittingClass, attributes, 0.0, 255.0, numThresholds);
        }
    }
    INFO_PRINTF(
        "ImageBlockSplittingClass(width=%d,height=%d,blockWidth=%d,blockHeight="
        "%d,numThresholds=%d) created splitting class of size %zu\n",
        width, height, blockWidth, blockHeight, numThresholds, splittingClass.size());
    return splittingClass;
}


/*
 * After preprocessing: 105 cols
 * The 6 continuous features (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) are first.
 * 0: age, continuous, around 18-80
 * 1: fnlwgt, continuous, around 0-800000
 * 2: education-num, continuous 1-16
 * 3: capital-gain, continuous, 0-20000
 * 4: capital-loss, continuous, 0-25000
 * 5: hours-per-week, continuous, 0-100
 * For these continuous features, use numThresholds (default = 10)
 * Then, the rest, from index 6 to 104 is binary one hot encodings
 *
 * For continuous x-y, I'll have 10 splitting function evenly spaced; jump = (y-x)/10 and give for i = 0,1,2,..,9: (i+0.5)*jump
 *
 * Adult dataset has 24.78% with >50K and 75.22% with <=50K
 */
std::vector<std::shared_ptr<Split>> AdultSplittingClass(int numThresholds)
{
    std::vector<std::shared_ptr<Split>> splittingClass;
    addContinuous(splittingClass, {0}, 18, 80, numThresholds);
    addContinuous(splittingClass, {1}, 0, 800000, numThresholds);
    addContinuous(splittingClass, {2}, 1, 16, numThresholds);
    addContinuous(splittingClass, {3}, 0, 20000, numThresholds);
    addContinuous(splittingClass, {4}, 0, 25000, numThresholds);
    addContinuous(splittingClass, {5}, 0, 100, numThresholds);
    for (int i = 6; i < 108; i++) {
        std::vector<int> attribute = {i};
        std::shared_ptr<Split> splitFn = std::static_pointer_cast<Split>(
            std::make_shared<ThresholdSplit>(attribute, 0.5));
        splittingClass.push_back(splitFn);
    }
    INFO_PRINTF(
        "AdultSplittingClass(numThresholds=%d) created splitting class of size "
        "%zu\n",
        numThresholds, splittingClass.size());
    return splittingClass;
}


/*
 * nursery dataset is 27 one-hot encoded
 */
std::vector<std::shared_ptr<Split>> NurserySplittingClass()
{
    std::vector<std::shared_ptr<Split>> splittingClass;
    for (int i = 0; i < 27; i++) {
        std::vector<int> attribute = {i};
        std::shared_ptr<Split> splitFn = std::static_pointer_cast<Split>(
            std::make_shared<ThresholdSplit>(attribute, 0.5));
        splittingClass.push_back(splitFn);
    }
    INFO_PRINTF("NurserySplittingClass() created splitting class of size %zu\n",
                splittingClass.size());
    return splittingClass;
}

/*
0 age :  18  to  95
1 balance :  -8019  to  102127
2 day :  1  to  31
3 duration :  0  to  4918
4 campaign :  1  to  63
5 pdays :  -1  to  871
6 previous :  0  to  275
 */
std::vector<std::shared_ptr<Split>> BankSplittingClass()
{
    std::vector<std::shared_ptr<Split>> splittingClass;
    addContinuous(splittingClass, {0}, 18, 95, 10);
    addContinuous(splittingClass, {1}, -8019, 102127, 10);
    addContinuous(splittingClass, {2}, 1, 31, 10);
    addContinuous(splittingClass, {3}, 0, 4918, 10);
    addContinuous(splittingClass, {4}, 1, 63, 10);
    addContinuous(splittingClass, {5}, 0, 871, 10);
    std::vector<int> attribute = {5};
    splittingClass.push_back(std::static_pointer_cast<Split>(
        std::make_shared<ThresholdSplit>(attribute, -0.5)));
    addContinuous(splittingClass, {6}, 0, 275, 10);
    for (int i = 7; i < 51; i++) {
        attribute = {i};
        std::shared_ptr<Split> splitFn = std::static_pointer_cast<Split>(
            std::make_shared<ThresholdSplit>(attribute, 0.5));
        splittingClass.push_back(splitFn);
    }
    INFO_PRINTF("BankSplittingClass() created splitting class of size %zu\n",
                splittingClass.size());
    return splittingClass;
}

std::vector<std::shared_ptr<Split>> CreditcardSplittingClass()
{
    std::vector<std::shared_ptr<Split>> splittingClass;
    addContinuous(splittingClass, {0}, 10000, 1000000, 10);
    addContinuous(splittingClass, {1}, 1, 2, 1);
    addContinuous(splittingClass, {2}, 0, 6, 6);
    addContinuous(splittingClass, {3}, 0, 3, 3);
    addContinuous(splittingClass, {4}, 21, 79, 10);
    addContinuous(splittingClass, {5}, -2, 8, 10);
    addContinuous(splittingClass, {6}, -2, 8, 10);
    addContinuous(splittingClass, {7}, -2, 8, 10);
    addContinuous(splittingClass, {8}, -2, 8, 10);
    addContinuous(splittingClass, {9}, -2, 8, 10);
    addContinuous(splittingClass, {10}, -2, 8, 10);
    addContinuous(splittingClass, {11}, -165580, 964511, 10);
    addContinuous(splittingClass, {12}, -69777, 983931, 10);
    addContinuous(splittingClass, {13}, -157264, 1664089, 10);
    addContinuous(splittingClass, {14}, -170000, 891586, 10);
    addContinuous(splittingClass, {15}, -81334, 927171, 10);
    addContinuous(splittingClass, {16}, -339603, 961664, 10);
    addContinuous(splittingClass, {17}, 0, 873552, 10);
    addContinuous(splittingClass, {18}, 0, 1684259, 10);
    addContinuous(splittingClass, {19}, 0, 896040, 10);
    addContinuous(splittingClass, {20}, 0, 621000, 10);
    addContinuous(splittingClass, {21}, 0, 426529, 10);
    addContinuous(splittingClass, {22}, 0, 528666, 10);
    INFO_PRINTF(
        "CreditcardSplittingClass() created splitting class of size %zu\n",
        splittingClass.size());
    return splittingClass;
}

std::vector<std::shared_ptr<Split>> SkinSplittingClass(int numThresh)
{
    std::vector<std::shared_ptr<Split>> splittingClass;
    addContinuous(splittingClass, {0}, 0, 255, numThresh);
    addContinuous(splittingClass, {1}, 0, 255, numThresh);
    addContinuous(splittingClass, {2}, 0, 255, numThresh);
    INFO_PRINTF("SkinSplittingClass(%d) created splitting class of size %zu\n", numThresh, splittingClass.size());
    return splittingClass;
}

/*
 * duration :  0  to  58329
 * src_bytes :  0  to  693375640
 * dst_bytes :  0  to  5155468
 * wrong_fragment :  0  to  3
 * urgent :  0  to  3
 * hot :  0  to  30
 * num_failed_logins :  0  to  5
 * num_compromised :  0  to  884
 * root_shell :  0  to  1
 * su_attempted :  0  to  2
 * num_root :  0  to  993
 * num_file_creations :  0  to  28
 * num_shells :  0  to  2
 * num_access_files :  0  to  8
 * num_outbound_cmds :  0  to  0
 * count :  0  to  511
 * srv_count :  0  to  511
 * serror_rate :  0.0  to  1.0
 * srv_serror_rate :  0.0  to  1.0
 * rerror_rate :  0.0  to  1.0
 * srv_rerror_rate :  0.0  to  1.0
 * same_srv_rate :  0.0  to  1.0
 * diff_srv_rate :  0.0  to  1.0
 * srv_diff_host_rate :  0.0  to  1.0
 * dst_host_count :  0  to  255
 * dst_host_srv_count :  0  to  255
 * dst_host_same_srv_rate :  0.0  to  1.0
 * dst_host_diff_srv_rate :  0.0  to  1.0
 * dst_host_same_src_port_rate :  0.0  to  1.0
 * dst_host_srv_diff_host_rate :  0.0  to  1.0
 * dst_host_serror_rate :  0.0  to  1.0
 * dst_host_srv_serror_rate :  0.0  to  1.0
 * dst_host_rerror_rate :  0.0  to  1.0
 * dst_host_srv_rerror_rate :  0.0  to  1.0
 */
std::vector<std::shared_ptr<Split>> KDDCupSplittingClass()
{
    std::vector<std::shared_ptr<Split>> splittingClass;
    addContinuous(splittingClass, {0}, 0, 58329, 10);
    addContinuous(splittingClass, {1}, 0, 693375640, 10);
    addContinuous(splittingClass, {2}, 0, 5155468, 10);
    addContinuous(splittingClass, {3}, 0, 3, 10);
    addContinuous(splittingClass, {4}, 0, 3, 10);
    addContinuous(splittingClass, {5}, 0, 30, 10);
    addContinuous(splittingClass, {6}, 0, 5, 10);
    addContinuous(splittingClass, {7}, 0, 884, 10);
    addContinuous(splittingClass, {8}, 0, 1, 10);
    addContinuous(splittingClass, {9}, 0, 2, 10);
    addContinuous(splittingClass, {10}, 0, 993, 10);
    addContinuous(splittingClass, {11}, 0, 28, 10);
    addContinuous(splittingClass, {12}, 0, 2, 10);
    addContinuous(splittingClass, {13}, 0, 8, 10);
    addContinuous(splittingClass, {14}, 0, 0, 10);
    addContinuous(splittingClass, {15}, 0, 511, 10);
    addContinuous(splittingClass, {16}, 0, 511, 10);
    addContinuous(splittingClass, {17}, 0.0, 1.0, 10);
    addContinuous(splittingClass, {18}, 0.0, 1.0, 10);
    addContinuous(splittingClass, {19}, 0.0, 1.0, 10);
    addContinuous(splittingClass, {20}, 0.0, 1.0, 10);
    addContinuous(splittingClass, {21}, 0.0, 1.0, 10);
    addContinuous(splittingClass, {22}, 0.0, 1.0, 10);
    addContinuous(splittingClass, {23}, 0.0, 1.0, 10);
    addContinuous(splittingClass, {24}, 0, 255, 10);
    addContinuous(splittingClass, {25}, 0, 255, 10);
    addContinuous(splittingClass, {26}, 0.0, 1.0, 10);
    addContinuous(splittingClass, {27}, 0.0, 1.0, 10);
    addContinuous(splittingClass, {28}, 0.0, 1.0, 10);
    addContinuous(splittingClass, {29}, 0.0, 1.0, 10);
    addContinuous(splittingClass, {30}, 0.0, 1.0, 10);
    addContinuous(splittingClass, {31}, 0.0, 1.0, 10);
    addContinuous(splittingClass, {32}, 0.0, 1.0, 10);
    addContinuous(splittingClass, {33}, 0.0, 1.0, 10);
    for (int i = 34; i < 121; i++) {
        std::vector<int> attribute = {i};
        std::shared_ptr<Split> splitFn = std::static_pointer_cast<Split>(
            std::make_shared<ThresholdSplit>(attribute, 0.5));
        splittingClass.push_back(splitFn);
    }
    INFO_PRINTF("KDDCupSplittingClass() created splitting class of size %zu\n", splittingClass.size());
    return splittingClass;
}


/*
 * hour :  14102100  to  14103023
banner_pos :  0  to  7
C1 :  1001  to  1012
C14 :  375  to  24052
C15 :  120  to  1024
C16 :  20  to  1024
C17 :  112  to  2758
C18 :  0  to  3
C19 :  33  to  1959
C20 :  -1  to  100248
C21 :  1  to  255
 */
std::vector<std::shared_ptr<Split>> CTRSplittingClass() {
    std::vector<std::shared_ptr<Split>> splittingClass;
    addContinuous(splittingClass, {0}, 14102100, 14103023, 10); // hour
    addContinuous(splittingClass, {1}, 0, 7, 7); // banner_pos
    addContinuous(splittingClass, {2}, 1001, 1012, 7); // C1
    addContinuous(splittingClass, {3}, 375,24052,100); // C14
    addContinuous(splittingClass, {4}, 120, 1024, 4); // C15
    addContinuous(splittingClass, {5}, 20, 1024, 4); // C16
    addContinuous(splittingClass, {6}, 112, 2758, 40); // C17
    addContinuous(splittingClass, {7}, 0, 3, 3); // C18
    addContinuous(splittingClass, {8}, 33, 1839, 10); // C19
    addContinuous(splittingClass, {9}, 100000, 100248, 15); // C20
    addContinuous(splittingClass, {10}, 1, 255, 10); // C21
    for (int i = 11; i < 64; i++) {
        std::vector<int> attribute = {i};
        std::shared_ptr<Split> splitFn = std::static_pointer_cast<Split>(
                std::make_shared<ThresholdSplit>(attribute, 0.5));
        splittingClass.push_back(splitFn);
    }

    INFO_PRINTF("CTRSplittingClass() created splitting class of size %zu\n", splittingClass.size());
    return splittingClass;
}


class SplittingCriterion {
public:
    SplittingCriterion(int numLabels) : numLabels(numLabels)
    {
    }

    virtual ~SplittingCriterion() = default;

    virtual float calcG(std::unordered_map<int, float>& counts) = 0;
    virtual float calcG(std::unordered_map<int, int>& counts)
    {
        std::unordered_map<int, float> input;
        for (auto& label2count : counts) {
            input.insert({label2count.first, (float)label2count.second});
        }
        return calcG(input);
    }

    virtual float sensitivity(int totalSize) = 0;

    int numLabels;
};

class Entropy : public SplittingCriterion {
public:
    Entropy(int numLabels) : SplittingCriterion(numLabels)
    {
    }

    float calcG(std::unordered_map<int, float>& counts) override
    {
        float result = 0.0;
        float total = 0.0;
        for (auto& label2count : counts) {
            total += label2count.second;
        }
        for (auto& label2count : counts) {
            float p = label2count.second / total;
            result -= p * log(p) / log(numLabels);
        }
        return result;
    }

    float sensitivity(int m) override
    {
        // assuming boolean splits
        float numSplitLabels = 2;
        return numSplitLabels / m + (float)numLabels * log(m) / m * (numSplitLabels + 1);
    }
};

class Gini : public SplittingCriterion {
public:
    Gini(int numLabels) : SplittingCriterion(numLabels)
    {
    }

    float calcG(std::unordered_map<int, float>& counts) override
    {
        float total = 0.0;
        for (auto& label2count : counts) {
            total += label2count.second;
        }
        float result = 1.0;
        for (auto& label2count : counts) {
            float p = label2count.second / total;
            result -= p * p;
        }
        return result;
    }

    float sensitivity(int m) override
    {
        float md = (float)m;
        return 1. - pow(md / (md + 1), 2) - pow(1 / (md + 1), 2);
    }
};

#endif // D3T_SPLIT_H
