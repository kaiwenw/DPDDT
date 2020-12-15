/** @file single_run.cpp
 *  @brief entry point for AWS jobs
 */

#include "run_helpers.h"

int main() { // int argc, char *argv[], char *envp[]) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    std::vector<int> numEntities = {
        4,
    };

    std::vector<std::string> splittingCriterionNames = {
        "entropy",
        //        "gini",
    };

    std::vector<int> maxNumNodes = {512};

    std::vector<int> maxDepths = {
        //40,
        80,
    };

    std::vector<float> epsilons = {0.1};

    // -1 is for no noise!
    std::vector<float> alphas = {
         -1, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    };
    std::vector<std::string> budgetFns = {
        "decay",
        //        "uniform",
        //        "harmonic",
    };
    std::vector<std::string> algos = {
        "singleMachine",
        "localRNM",
        "distributedBaseline",
    };


    const char *dataset_c;
    dataset_c = getenv("DATASET");
    if (dataset_c == NULL) {
        printf("FATAL: environment variable DATASET not set!\n");
        assert(false);
    }
    std::string dataset(dataset_c);
    std::cout << "got dataset = " << dataset << std::endl;

    const char *seed_c;
    seed_c = getenv("SEED");
    if (seed_c == NULL) {
        printf("FATAL: environment variable SEED not set!\n");
        assert(false);
    }
    std::string seed_s(seed_c);
    int seed = std::stoi(seed_s);
    std::cout << "got seed = " << seed << std::endl;

    const char *trainingFraction_c;
    trainingFraction_c = getenv("TRAINING_FRACTION");
    if (trainingFraction_c == NULL) {
        printf("FATAL: environment variable TRAINING_FRACTION not set! Should be float in [0,1]. \n");
        assert(false);
    }
    std::string trainingFraction_s(trainingFraction_c);
    float trainingFraction = std::stof(trainingFraction_s);
    std::cout << "got training fraction = " << trainingFraction << std::endl;

    const char *budgetFn_c;
    budgetFn_c = getenv("BUDGET_FN");
    if (budgetFn_c == NULL) {
        printf("FATAL: environment variable BUDGET_FN not set! Possible values: decay/uniform/harmonic \n");
        assert(false);
    }
    std::string budgetFn(budgetFn_c);
    std::cout << "got budgetFn = " << budgetFn << std::endl;

    const char *leafPrivacyFraction_c;
    leafPrivacyFraction_c = getenv("LEAF_PRIVACY_FRACTION");
    if (leafPrivacyFraction_c == NULL) {
        printf("FATAL: environment variable LEAF_PRIVACY_FRACTION not set! Should be float in [0,1]. \n");
        assert(false);
    }
    std::string leafPrivacyFraction_s(leafPrivacyFraction_c);
    float leafPrivacyFraction = std::stof(leafPrivacyFraction_s);
    std::cout << "got leaf privacy fraction = " << leafPrivacyFraction << std::endl;

    std::string csvPath = "dataset_" + dataset + \
                          "-seed_" + seed_s + \
                          "-trainingFraction_" + trainingFraction_s + \
                          "-budgetFn_" + budgetFn + \
                          "-leafPrivacyFraction_" + leafPrivacyFraction_s + \
                          + ".csv";

    std::ofstream myfile_;
    myfile_.open(csvPath);
    myfile_ << "dataset,"
               "trainingFraction,"
               "numEntities,"
               "seed,"
               "splittingCriterionName,"
               "leafPrivacyFraction,"
               "maxNumNode,"
               "maxDepth,"
               "eps,"
               "alpha,"
               "budgetFn,"
               "algo,"
               "trainAcc,"
               "testAcc,"
               "trainingTime,"
               "evaluationTime,"
               "numNodes,"
               "maxAchievedDepth\n";

    for (int numEntity : numEntities) {
        for (const std::string &splittingCriterionName : splittingCriterionNames) {
            for (int maxNumNode : maxNumNodes) {
                for (int maxDepth : maxDepths) {
                    for (float eps : epsilons) {
                        for (float alpha : alphas) {
                            for (const std::string &algo : algos) {
                                Results r = performTest(
                                        dataset,
                                        trainingFraction,
                                        numEntity,
                                        seed,
                                        splittingCriterionName,
                                        leafPrivacyFraction,
                                        maxNumNode,
                                        maxDepth,
                                        eps,
                                        alpha,
                                        budgetFn,
                                        algo);
                                myfile_ << dataset 
                                        << "," << trainingFraction 
                                        << "," << numEntity
                                        << "," << seed 
                                        << "," << splittingCriterionName
                                        << "," << leafPrivacyFraction 
                                        << "," << maxNumNode 
                                        << "," << maxDepth 
                                        << "," << eps
                                        << "," << alpha 
                                        << "," << budgetFn 
                                        << "," << algo
                                        << "," << r.trainAcc 
                                        << "," << r.testAcc 
                                        << "," << r.trainingTime
                                        << "," << r.evaluationTime
                                        << "," << r.numNodes 
                                        << "," << r.maxAchievedDepth
                                        << std::endl;
                            }
                        }
                    }
                }
            }
        }
    }
}

