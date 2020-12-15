/** @file run_helpers.h
 *  @brief Helper functions including creating entities objects, evaluating a 
 *         decision tree, and a main function that performs the actual 
 *         training task by putting together coordinators and entities.
 *
 */
#ifndef D3T_RUN_HELPERS_H
#define D3T_RUN_HELPERS_H

#include "coordinator.h"
#include "entity.h"
#include "split.h"
#include "utils.h"
#include <thread>
#include <vector>

std::vector<Entity> createEntities(bool turnOffNoise,
                                   int seed,
                                   const std::vector<std::vector<std::vector<float>>>& data,
                                   const std::vector<std::vector<int>>& labels,
                                   const std::vector<std::shared_ptr<Split>>& splittingClass,
                                   std::shared_ptr<SplittingCriterion> splittingCriterion)
{
    assert(data.size() == labels.size());
    std::vector<Entity> result;
    for (size_t i = 0; i < data.size(); i++) {
        result.emplace_back(turnOffNoise, i, seed, data[i], labels[i],
                            splittingClass, splittingCriterion);
    }
    return result;
}

float evaluate(const std::shared_ptr<CoordinatorNode>& root,
                const std::vector<std::vector<float>>& data,
                const std::vector<int>& labels)
{
    int numCorrect = 0;
    for (size_t i = 0; i < data.size(); i++) {
        std::shared_ptr<CoordinatorNode> node = root;
        while (!node->isLeaf) {
            int split = node->splitFn->applySplit(data[i]);
            if (node->children.find(split) == node->children.end()) {
                INFO_PRINTF("%zu has never been encountered before\n", i);
                break;
            }
            node = node->children[split];
        }
        if (node->label == labels[i]) {
            numCorrect++;
        }
    }
    return (float)numCorrect / data.size();
}

class Results {
public:
    Results(float trainAcc,
            float testAcc,
            std::string trainingTime,
            std::string evaluationTime,
            int numNodes,
            int maxAchievedDepth)
        : trainAcc(trainAcc),
          testAcc(testAcc),
          trainingTime(std::move(trainingTime)),
          evaluationTime(std::move(evaluationTime)),
          numNodes(numNodes),
          maxAchievedDepth(maxAchievedDepth)
    {
    }

    float trainAcc;
    float testAcc;
    std::string trainingTime;
    std::string evaluationTime;
    int numNodes;
    int maxAchievedDepth;
};

Results performTest(const std::string& dataset,
                    float trainingFraction,
                    int numEntities,
                    int seed,
                    const std::string& splittingCriterionName,
                    float leafPrivacyFraction,
                    int maxNumNodes,
                    int maxDepth,
                    float epsilon,
                    float alpha,
                    const std::string& budgetFn,
                    const std::string& algo)
{
    std::vector<std::vector<float>> data, testData;
    std::vector<int> labels, testLabels;

    std::string trainPath = "../data/" + dataset + "_train";
    std::string testPath = "../data/" + dataset + "_test";

    int numLabels = parseProtobuf(data, labels, trainPath, seed, trainingFraction);
    assert(data.size() > 0);

    int trainSize = data.size();
    int numCols = data[0].size();
    parseProtobuf(testData, testLabels, testPath, 0, 1.0);
    int testSize = testData.size();
    printf(
        "performTest(dataset=%s, "
        "trainingFraction=%f, "
        "numEntities=%d, "
        "seed=%d, "
        "splittingCriterionName=%s, "
        "leafPrivacyFraction=%f, "
        "maxNumNodes=%d, "
        "maxDepth=%d, "
        "epsilon=%f, "
        "alpha=%f, "
        "budgetFn=%s, "
        "algo=%s) "
        "with %d cols and %d label types\t%d "
        "trainSize\t%d testSize\n",
        dataset.c_str(),
        trainingFraction,
        numEntities,
        seed,
        splittingCriterionName.c_str(),
        leafPrivacyFraction,
        maxNumNodes,
        maxDepth,
        epsilon,
        alpha,
        budgetFn.c_str(),
        algo.c_str(),
        numCols, numLabels,
        trainSize, testSize);

    std::vector<int> partitionSizes;
    if (algo == "singleMachine") {
        partitionSizes = {trainSize};
    }
    else if (algo == "localRNM" || algo == "distributedBaseline") {
        int entitySize = trainSize / numEntities;
        int lastEntitySize = trainSize - (numEntities - 1) * entitySize;
        for (int i = 0; i < numEntities - 1; i++) {
            partitionSizes.push_back(entitySize);
        }
        partitionSizes.push_back(lastEntitySize);
    }
    else {
        WARNING_PRINTF("Invalid algo %s\n", algo.c_str());
        assert(false);
    }

    std::shared_ptr<SplittingCriterion> splittingCriterion;
    if (splittingCriterionName == "entropy") {
        splittingCriterion = std::static_pointer_cast<SplittingCriterion>(
            std::make_shared<Entropy>(numLabels));
    }
    else if (splittingCriterionName == "gini") {
        splittingCriterion = std::static_pointer_cast<SplittingCriterion>(
            std::make_shared<Gini>(numLabels));
    }
    else {
        WARNING_PRINTF("Invalid splitting criterion %s\n",
                       splittingCriterionName.c_str());
        assert(false);
    }

    std::vector<std::shared_ptr<Split>> splittingClass;
    if (dataset == "mnist60k" || dataset == "mnist100k") {
        splittingClass = ImageBlockSplittingClass(28, 28, 4, 4, 3);
    }
    else if (dataset == "adult") {
        splittingClass = AdultSplittingClass(10);
    }
    else if (dataset == "bank") {
        splittingClass = BankSplittingClass();
    }
    else if (dataset == "creditcard") {
        splittingClass = CreditcardSplittingClass();
    }
    else if (dataset == "skin") {
        splittingClass = SkinSplittingClass(32);
    }
    else if (dataset == "kddcup") {
        splittingClass = KDDCupSplittingClass();
    }
    else if (dataset == "ctr") {
        splittingClass = CTRSplittingClass();
    }
    else {
        WARNING_PRINTF("Invalid dataset %s\n", dataset.c_str());
        assert(false);
    }

    std::vector<std::vector<std::vector<float>>> entitiesData;
    std::vector<std::vector<int>> entitiesLabels;
    std::tie(entitiesData, entitiesLabels) = partitionData(data, labels, partitionSizes);
    std::vector<Entity> entities =
        createEntities(floatEq(alpha, -1), seed, entitiesData, entitiesLabels,
                       splittingClass, splittingCriterion);
    Coordinator coordinator(leafPrivacyFraction,
                            maxNumNodes,
                            maxDepth,
                            epsilon,
                            budgetFn,
                            algo,
                            trainSize,
                            entities,
                            splittingClass,
                            splittingCriterion);

    auto start = std::chrono::high_resolution_clock::now();
    std::shared_ptr<CoordinatorNode> root;
    int numNodes, maxAchievedDepth;
    std::tie(root, numNodes, maxAchievedDepth) = coordinator.train(alpha);
    auto end = std::chrono::high_resolution_clock::now();
    std::string trainingTime =
        sec2str(std::chrono::duration_cast<std::chrono::seconds>(end - start).count());

    start = std::chrono::high_resolution_clock::now();
    float trainAcc = evaluate(root, data, labels);
    float testAcc = evaluate(root, testData, testLabels);
    end = std::chrono::high_resolution_clock::now();
    std::string evaluationTime =
        sec2str(std::chrono::duration_cast<std::chrono::seconds>(end - start).count());

    printf(
        "Training acc: %f\tTesting acc: %f\tTraining time: %s\tEvaluation "
        "time: %s\tNum nodes: %d\tMax achieved depth: %d\n",
        trainAcc, testAcc, trainingTime.c_str(), evaluationTime.c_str(),
        numNodes, maxAchievedDepth);
    return Results(trainAcc, testAcc, trainingTime, evaluationTime, numNodes, maxAchievedDepth);
}

#endif // D3T_RUN_HELPERS_H


