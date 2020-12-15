/** @file coordinator.h
 *  @brief Central coordinator for distributed decision tree learning. 
 *         Uses results from entity queries to build a decision tree.
 *         Here, entities are just represented as objects. The queries can be
 *         easily extended to RPC calls for a distributed version.
 */
#ifndef D3T_COORDINATOR_H
#define D3T_COORDINATOR_H

#include "entity.h"
#include "split.h"
#include "utils.h"

#include <ctime>
#include <memory>
#include <queue>
#include <unordered_map>

class CoordinatorNode {
public:
    CoordinatorNode(int id, int depth)
        : id(id), depth(depth), isLeaf(true), label(-1)
    {
    }

    int id;
    int depth;
    float weight;

    // if leaf
    bool isLeaf;
    int label;

    // if not leaf
    std::shared_ptr<Split> splitFn;
    std::unordered_map<int, std::shared_ptr<CoordinatorNode>> children;
};

class QueueDataType {
public:
    QueueDataType(float priority,
                  std::shared_ptr<CoordinatorNode> leaf,
                  std::shared_ptr<Split> splitFn)
        : priority(priority), leaf(std::move(leaf)), splitFn(std::move(splitFn))
    {
    }

    float priority;
    std::shared_ptr<CoordinatorNode> leaf;
    std::shared_ptr<Split> splitFn;
};

class WeightLessThan {
public:
    bool operator()(QueueDataType a, QueueDataType b)
    {
        return a.priority <= b.priority;
    }
};

class Coordinator {
public:
    Coordinator(float leafPrivacyFraction,
                int maxNumNodes,
                int maxDepth,
                float eps,
                std::string budgetFn,
                std::string algo,
                int numDatapoints,
                std::vector<Entity> entities,
                std::vector<std::shared_ptr<Split>> splittingClass,
                std::shared_ptr<SplittingCriterion> splittingCriterion)
        : leafPrivacyFraction(leafPrivacyFraction),
          maxNumNodes(maxNumNodes),
          maxDepth(maxDepth),
          eps(eps),
          budgetFn(std::move(budgetFn)),
          algo(std::move(algo)),
          numDataPoints(numDatapoints),
          entities(std::move(entities)),
          splittingClass(std::move(splittingClass)),
          splittingCriterion(std::move(splittingCriterion))
    {
        INFO_PRINTF(
            "Initialized Coordinator(maxNumNodes=%d, eps=%f, budgetFn=%s, "
            "algo=%s, numDatapoints=%d, %zu entities, %zu splitting class, and "
            "criterion)\n",
            maxNumNodes, eps, Coordinator::budgetFn.c_str(),
            Coordinator::algo.c_str(), numDatapoints,
            Coordinator::entities.size(), Coordinator::splittingClass.size());
    }

    // leaf budget for internal nodes
    float leafBudget(float depth)
    {
        assert(depth < maxDepth);
        if (budgetFn == "uniform") {
            return 1. / maxNumNodes;
        }
        else if (budgetFn == "decay") {
            // depth starts at 1
            return 1. / pow(2, depth);
        }
        else if (budgetFn == "harmonic") {
            float multiplier = 0.;
            for (int i = 1; i <= depth; i++) {
                multiplier += 1. / (maxDepth - i + 1.);
            }
            return multiplier / maxDepth;
        }
        else {
            printf("Invalid budgetFn option %s\n", budgetFn.c_str());
            assert(false);
        }
        return -1;
    }

    // (root, numNodes, maxDepth)
    std::tuple<std::shared_ptr<CoordinatorNode>, int, int> train(float alpha)
    {
        std::shared_ptr<Split> splitFnHat;
        float Jhat;
        std::priority_queue<QueueDataType, std::vector<QueueDataType>, WeightLessThan> Q;

        float splitsAlpha = alpha * (1. - leafPrivacyFraction);

        std::shared_ptr<CoordinatorNode> root =
            std::make_shared<CoordinatorNode>(/*id=*/0, /*depth=*/1);
        root->weight = 1.0;
        id2node_.push_back(root);
        float rootAlpha = splitsAlpha * leafBudget(root->depth);
        std::tie(splitFnHat, Jhat) = privateSplit(root, (float)numDataPoints, rootAlpha);
        assert(splitFnHat != nullptr);
        Q.push(QueueDataType(Jhat, root, splitFnHat));

        while ((int)id2node_.size() < maxNumNodes) {
            if (Q.empty())
                break;
            std::shared_ptr<CoordinatorNode> bestLeaf = Q.top().leaf;
            bestLeaf->splitFn = Q.top().splitFn;
            assert(bestLeaf->isLeaf);
            bestLeaf->isLeaf = false;
#if defined(DEBUG) && DEBUG > 1
            int workedTotal = std::round(totalCountAcrossEntities(bestLeaf->id, INT_MAX));
#endif
            DEBUG_PRINTF(
                "Node: %d\tweight: %f\tpriority: %f\tdepth: %d\twith "
                "%d/%d\tSplitFn %d (%s)\n",
                bestLeaf->id, bestLeaf->weight, Q.top().priority,
                bestLeaf->depth, workedTotal, numDataPoints,
                bestLeaf->splitFn->id, bestLeaf->splitFn->toString().c_str());
            Q.pop();

            // tell entities to split the leaf with split function
            for (size_t i = 0; i < entities.size(); i++) {
                entities[i].splitLeafWithFn(bestLeaf->id, bestLeaf->splitFn);
            }

            // for each child, perform a private split
            for (size_t i = 0; i < bestLeaf->splitFn->labels.size(); i++) {
                std::shared_ptr<CoordinatorNode> child = std::make_shared<CoordinatorNode>(
                    (int)id2node_.size(), bestLeaf->depth + 1);
                bestLeaf->children.insert({bestLeaf->splitFn->labels[i], child});
                id2node_.push_back(child);
                if (child->depth >= maxDepth) {
                    continue; // depth of internal node goes up to maxDepth-1
                }

                float leafAlpha = splitsAlpha * leafBudget(child->depth);
                float total = totalCountAcrossEntities(child->id, leafAlpha / 3);
                float weight = total / numDataPoints;
                assert(weight <= 1.0);
                child->weight = weight;

                DEBUG_PRINTF("Split %zu has %d/%d\n", i,
                             (int)std::round(totalCountAcrossEntities(child->id, INT_MAX)),
                             workedTotal);

                if (weight <= eps / maxNumNodes) {
                    DEBUG_PRINTF(
                        "Node %d has weight %f=%f/%d too small, less than %f\n",
                        child->id, weight, total, numDataPoints, eps / maxNumNodes);
                    continue;
                }
                std::tie(splitFnHat, Jhat) =
                    privateSplit(child, total, 2 * leafAlpha / 3);
                if (std::isnan(Jhat)) {
                    DEBUG_PRINTF("Node %d has NaN Jhat\n", child->id);
                    continue;
                }
                // TODO: hardcoded threshold value right now
                if (Jhat < 1e-2) {
                    DEBUG_PRINTF(
                        "Node %d has Jhat %f (id=%d), which is too small\n",
                        child->id, Jhat, splitFnHat->id);
                    continue;
                }
                Q.push(QueueDataType(weight * Jhat, child, splitFnHat));
            }
        }

        // labelling the leaves with the rest of the budget
        float leavesLabelingAlpha = alpha * leafPrivacyFraction;
        int maxAchievedDepth = 1;
        std::queue<std::shared_ptr<CoordinatorNode>> BFS;
        BFS.push(root);
        while (!BFS.empty()) {
            std::shared_ptr<CoordinatorNode> node = BFS.front();
            maxAchievedDepth = std::max(maxAchievedDepth, node->depth);
            BFS.pop();

            if (node->children.empty()) {
                assert(node->isLeaf);
                std::unordered_map<int, float> counts =
                    labelCountsAcrossEntities(node->id, leavesLabelingAlpha);
                float maxCount = 0;
                int bestLabel = -1;
                for (auto& label2count : counts) {
                    if (label2count.second > maxCount) {
                        maxCount = label2count.second;
                        bestLabel = label2count.first;
                    }
                }
                node->label = bestLabel;
            }

            for (auto& split2child : node->children) {
                BFS.push(split2child.second);
            }
        }
        return std::make_tuple(root, id2node_.size(), maxAchievedDepth);
    }

    const float leafPrivacyFraction;
    const int maxNumNodes;
    const int maxDepth;
    const float eps;
    const std::string budgetFn;
    const std::string algo;
    const int numDataPoints;
    std::vector<Entity> entities;
    const std::vector<std::shared_ptr<Split>> splittingClass;
    const std::shared_ptr<SplittingCriterion> splittingCriterion;

private:
    std::unordered_map<int, float> splitCountsAcrossEntities(
        int id, const std::shared_ptr<Split>& splitFn, float privacyEps) const
    {
        std::unordered_map<int, float> splitCounts, tmpSplitCounts;
        for (size_t i = 0; i < entities.size(); i++) {
            tmpSplitCounts = entities[i].getSplitCounts(id, splitFn, privacyEps);

            // aggregating splitCounts
            for (auto& split2count : tmpSplitCounts) {
                if (splitCounts.find(split2count.first) == splitCounts.end()) {
                    splitCounts.insert(split2count);
                    continue;
                }
                splitCounts[split2count.first] += split2count.second;
            }
        }
        return splitCounts;
    }

    std::unordered_map<int, std::unordered_map<int, float>> splitLabelCountsAcrossEntities(
        int id, const std::shared_ptr<Split>& splitFn, float privacyEps) const
    {
        std::unordered_map<int, std::unordered_map<int, float>> splitLabelCount,
            tmpSplitLabelCount;
        for (size_t i = 0; i < entities.size(); i++) {
            tmpSplitLabelCount = entities[i].getSplitLabelCounts(id, splitFn, privacyEps);

            // aggregating splitLabelCount
            for (auto& split2labelCount : tmpSplitLabelCount) {
                if (splitLabelCount.find(split2labelCount.first) ==
                    splitLabelCount.end()) {
                    splitLabelCount.insert({split2labelCount});
                    continue;
                }
                for (auto& label2count : split2labelCount.second) {
                    if (splitLabelCount[split2labelCount.first].find(label2count.first) ==
                        splitLabelCount[split2labelCount.first].end()) {
                        splitLabelCount[split2labelCount.first].insert({label2count});
                        continue;
                    }
                    splitLabelCount[split2labelCount.first][label2count.first] +=
                        label2count.second;
                }
            }
        }
        return splitLabelCount;
    }

    std::unordered_map<int, float> labelCountsAcrossEntities(int id, float privacyEps) const
    {
        std::unordered_map<int, float> labelCount, tmpLabelCount;
        for (size_t i = 0; i < entities.size(); i++) {
            tmpLabelCount = entities[i].getLabelCounts(id, privacyEps);

            // aggregating labelCount
            for (auto& label2count : tmpLabelCount) {
                if (labelCount.find(label2count.first) == labelCount.end()) {
                    labelCount.insert(label2count);
                    continue;
                }
                labelCount[label2count.first] += label2count.second;
            }
        }
        return labelCount;
    }

    float totalCountAcrossEntities(int id, float privacyEps) const
    {
        float totalCount = 0;
        for (size_t i = 0; i < entities.size(); i++) {
            totalCount += entities[i].getTotalCount(id, privacyEps);
        }
        return totalCount;
    }

    /*
     * Calls PrivateSplit and returns an estimate of (best split, J value of best split.
     */
    std::tuple<std::shared_ptr<Split>, float> privateSplit(
        const std::shared_ptr<CoordinatorNode>& leaf, float total, float privacyEps) const
    {
        std::shared_ptr<Split> bestSplit = nullptr;
        float minCondG = INT_MAX;
        if (algo == "singleMachine") {
            assert((int)entities.size() == 1);
            return entities[0].localRNM(leaf->id, privacyEps);
        }

        std::vector<std::shared_ptr<Split>> candidateSplits;
        if (algo == "localRNM") {
            for (size_t i = 0; i < entities.size(); i++) {
                auto res = entities[i].localRNM(leaf->id, privacyEps / 2);
                if (std::get<0>(res) == nullptr) {
                    assert(std::isnan(std::get<1>(res)));
                    continue;
                }
                candidateSplits.push_back(std::get<0>(res));
            }
            // the rest of function has half the budget
            privacyEps = privacyEps / 2;
        }
        else if (algo == "distributedBaseline") {
            candidateSplits = splittingClass;
        }
        else {
            WARNING_PRINTF("Invalid algo %s\n", algo.c_str());
            assert(false);
        }

        // 2/3 of the privacy budget in for loop, and 1/3 of privacy budget for labelCounts
        float eachEps = privacyEps / (3 * candidateSplits.size());
        for (size_t i = 0; i < candidateSplits.size(); i++) {
            std::unordered_map<int, std::unordered_map<int, float>> splitLabelCounts =
                splitLabelCountsAcrossEntities(leaf->id, candidateSplits[i], eachEps);
            std::unordered_map<int, float> splitCounts =
                splitCountsAcrossEntities(leaf->id, candidateSplits[i], eachEps);
            float condG = 0.0;
            for (auto& split2labelCount : splitLabelCounts) {
                condG += splitCounts.at(split2labelCount.first) / total *
                         splittingCriterion->calcG(split2labelCount.second);
            }

            if (std::isnan(condG)) {
                WARNING_PRINTF("condG is nan!\n");
                assert(false);
            }

            if (condG < minCondG) {
                minCondG = condG;
                bestSplit = candidateSplits[i];
            }
        }
        std::unordered_map<int, float> labelCounts =
            labelCountsAcrossEntities(leaf->id, privacyEps / 3);
        float infoGain = splittingCriterion->calcG(labelCounts) - minCondG;
        return std::make_tuple(bestSplit, infoGain);
    }

    std::vector<std::shared_ptr<CoordinatorNode>> id2node_;
};

#endif // D3T_COORDINATOR_H
