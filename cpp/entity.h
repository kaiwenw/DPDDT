/** @file entity.h
 *  @brief Data entities (i.e. hospitals, private organizations, etc.) with the
 *         data of their own clients but does not want to share with other 
 *         entities. They agree on a common protocol orchestrated by the 
 *         coordinator and publish noised statistics of data to build a decision tree.
 */

#ifndef D3T_ENTITY_H
#define D3T_ENTITY_H

#include "noise.h"
#include "split.h"
#include "utils.h"
#include <limits>
#include <unordered_map>
#include <vector>

class EntityNode {
public:
    EntityNode(int id) : id(id), isLeaf(true)
    {
    }

    const int id;
    bool isLeaf;
    std::vector<int> idxs;
    std::unordered_map<int, std::shared_ptr<EntityNode>> children;
};

class Entity {
public:
    Entity(bool turnOffNoise,
           int entityIdx,
           int seed,
           std::vector<std::vector<float>> data,
           std::vector<int> labels,
           std::vector<std::shared_ptr<Split>> splittingClass,
           std::shared_ptr<SplittingCriterion> splittingCriterion)
        : privacyNoise_(entityIdx + seed, turnOffNoise),
          data_(data),
          labels_(labels),
          splittingClass(std::move(splittingClass)),
          splittingCriterion(std::move(splittingCriterion)),
          root(std::make_shared<EntityNode>(0))
    {
        for (size_t i = 0; i < data.size(); i++) {
            root->idxs.push_back((int)i);
        }
        id2node_.push_back(root);
        assert(data.size() == labels.size());
        INFO_PRINTF("Constructed entity %d with %zu data points\n", entityIdx,
                    data.size());
    }

    /*
     * Split leaf with given split function
     */
    void splitLeafWithFn(int id, const std::shared_ptr<Split>& splitFn)
    {
        std::shared_ptr<EntityNode> node = id2node_[id];
        assert(node->isLeaf);

        for (size_t i = 0; i < splitFn->labels.size(); i++) {
            std::shared_ptr<EntityNode> child =
                std::make_shared<EntityNode>((int)id2node_.size());
            id2node_.push_back(child);

            int label = splitFn->labels[i];
            assert(node->children.find(label) == node->children.end());
            node->children.insert({label, child});
        }

        for (size_t i = 0; i < node->idxs.size(); i++) {
            int label = splitFn->applySplit(data_[node->idxs[i]]);
            assert(node->children.find(label) != node->children.end());
            node->children[label]->idxs.push_back(node->idxs[i]);
        }

        node->isLeaf = false;
    }

    std::unordered_map<int, float> getSplitCounts(int id,
                                                   const std::shared_ptr<Split>& splitFn,
                                                   float privacyEps) const
    {
        std::unordered_map<int, float> noisedCounts;
        for (auto& split2count : splitCounts(id, splitFn)) {
            float noisedCount =
                split2count.second + privacyNoise_.laplace(1.0 / privacyEps);
            noisedCounts.insert({split2count.first, clipCount(noisedCount)});
        }
        return noisedCounts;
    }

    std::unordered_map<int, std::unordered_map<int, float>> getSplitLabelCounts(
        int id, const std::shared_ptr<Split>& splitFn, float privacyEps) const
    {
        std::unordered_map<int, std::unordered_map<int, float>> result;
        for (auto& split2labelCount : splitLabelCounts(id, splitFn)) {
            result.insert({split2labelCount.first, {}});
            for (auto& label2count : split2labelCount.second) {
                float noisedCount =
                    label2count.second + privacyNoise_.laplace(1.0 / privacyEps);
                result[split2labelCount.first].insert(
                    {label2count.first, clipCount(noisedCount)});
            }
        }
        return result;
    }

    std::unordered_map<int, float> getLabelCounts(int id, float privacyEps) const
    {
        std::unordered_map<int, float> result;
        for (auto& label2count : labelCounts(id)) {
            float noisedCount =
                label2count.second + privacyNoise_.laplace(1.0 / privacyEps);
            result.insert({label2count.first, clipCount(noisedCount)});
        }
        return result;
    }

    float getTotalCount(int id, float privacyEps) const
    {
        float noisedCount = totalCount(id) + privacyNoise_.laplace(1.0 / privacyEps);
        if (noisedCount < 0.) {
            return 0.;
        } else if (noisedCount > (float)data_.size()) {
            return (float)data_.size();
        } else {
            return noisedCount;
        }
    }

    /*
     * this returns the minimum conditional entropy one
     */
    std::tuple<std::shared_ptr<Split>, float> localRNM(int id, float privacyEps) const
    {
        if (id2node_[id]->idxs.empty()) {
            DEBUG_PRINTF("No data_ at leaf %d\n", id);
            return std::make_tuple(nullptr, std::numeric_limits<float>::quiet_NaN());
        }

        std::unordered_map<int, int> labelCount = labelCounts(id);
        float origG = splittingCriterion->calcG(labelCount);

        int total = (int)id2node_[id]->idxs.size();
        float minCondG = INT_MAX;
        std::shared_ptr<Split> bestSplit = nullptr;
        for (size_t i = 0; i < splittingClass.size(); i++) {
            std::unordered_map<int, std::unordered_map<int, int>> splitLabelCounts_ =
                splitLabelCounts(id, splittingClass[i]);
            std::unordered_map<int, int> splitCounts_ =
                splitCounts(id, splittingClass[i]);
            float condG = 0.0;
            for (auto& split2labelCount : splitLabelCounts(id, splittingClass[i])) {
                float innerG = splittingCriterion->calcG(split2labelCount.second);
                if (splitCounts_.find(split2labelCount.first) == splitCounts_.end()) {
//                    printLabelCounts(splitCounts_);
//                    printSplitLabelCounts(splitLabelCounts_);
                    assert(false);
                }
                condG += (float)splitCounts_.at(split2labelCount.first) / total * innerG;
            }

            // get noise for condG based on RNM
            float sensitivity = splittingCriterion->sensitivity(total);
            float condGNoise = privacyNoise_.laplace(sensitivity / privacyEps);
            condG += condGNoise;

            // condG is positive
            if (condG < 0) {
                condG = 0;
            }
            if (condG < minCondG) {
                minCondG = condG;
                bestSplit = splittingClass[i];
            }
        }
        float infoGain = origG - minCondG;
        return std::make_tuple(bestSplit, infoGain);
    }

private:
    float clipCount(float noisedCount) const
    {
        if (noisedCount < 1.0) {
            return 1.0;
        } else if (noisedCount > (float)data_.size()) {
            return (float)data_.size();
        } else {
            return noisedCount;
        }
    }

    std::unordered_map<int, int> splitCounts(int id, const std::shared_ptr<Split>& splitFn) const
    {
        std::unordered_map<int, int> splitCounts;
        for (size_t i = 0; i < id2node_[id]->idxs.size(); i++) {
            int idx = id2node_[id]->idxs[i];
            int split = splitFn->applySplit(data_[idx]);
            if (splitCounts.find(split) == splitCounts.end()) {
                splitCounts.insert({split, 0});
            }
            splitCounts[split]++;
        }
        return splitCounts;
    }

    std::unordered_map<int, std::unordered_map<int, int>> splitLabelCounts(
        int id, const std::shared_ptr<Split>& splitFn) const
    {
        std::unordered_map<int, std::unordered_map<int, int>> result;
        for (size_t i = 0; i < id2node_[id]->idxs.size(); i++) {
            int idx = id2node_[id]->idxs[i];
            int split = splitFn->applySplit(data_[idx]);
            if (result.find(split) == result.end()) {
                result.insert({split, {}});
            }
            if (result[split].find(labels_[idx]) == result[split].end()) {
                result[split].insert({labels_[idx], 0});
            }
            result[split][labels_[idx]]++;
        }
        return result;
    }

    std::unordered_map<int, int> labelCounts(int id) const
    {
        std::unordered_map<int, int> result;
        for (size_t i = 0; i < id2node_[id]->idxs.size(); i++) {
            int idx = id2node_[id]->idxs[i];
            if (result.find(labels_[idx]) == result.end()) {
                result.insert({labels_[idx], 0});
            }
            result[labels_[idx]]++;
        }
        return result;
    }

    int totalCount(int id) const
    {
        return id2node_[id]->idxs.size();
    }

    mutable Noise privacyNoise_;
    const std::vector<std::vector<float>> data_;
    const std::vector<int> labels_;
    std::vector<std::shared_ptr<EntityNode>> id2node_;
    const std::vector<std::shared_ptr<Split>> splittingClass;
    const std::shared_ptr<SplittingCriterion> splittingCriterion;
    const std::shared_ptr<EntityNode> root;
};

#endif // D3T_ENTITY_H
