#!/usr/bin/env python3
"Submit a job to AWS Batch."

import boto3

batch = boto3.client("batch", region_name="us-east-1")
# m4xlarge has 4 CPU and 16GB. So to maximize usage should be < 4000 - eps
MEMORY = 3800
# OUT_DIRECTORY = "s3://private-decision-trees-kaiwen/adult/"
OUT_DIRECTORY = "s3://private-decision-trees-kaiwen/icml-supp/"

START_IDX = 0
ARRAY_SIZE = 100

# datasets = ["mnist60k"]
# datasets = ["kddcup", "ctr"]
datasets = ["kddcup", "mnist60k", "ctr"]
# datasets = ["bank", "creditcard", "skin"]
# datasets = ["bank", "creditcard", "ctr", "kddcup", "mnist60k", "skin"]
# datasets = ["adult", "bank", "creditcard", "ctr", "kddcup", "mnist60k", "skin"]
training_fractions = [1.0]
# leaf_privacy_fractions = [0.125, 0.25, 0.5, 0.75, 0.875]
leaf_privacy_fractions = [2**-4, 2**-5, 2**-6, 2**-7, 2**-8]
# leaf_privacy_fractions = [0.125, 0.25, 0.5, 0.75, 0.875, 2**-4, 2**-5, 2**-6, 2**-7, 2**-8]
budget_fns = ["decay"]
# budget_fns = ["uniform"]


def float2str(num):
    assert isinstance(num, float)
    return str(num).replace(".", "o")


for dataset in datasets:
    for training_fraction in training_fractions:
        for budget_fn in budget_fns:
            for leaf_privacy_fraction in leaf_privacy_fractions:
                jobName = (
                    "privateDT-dataset_%s-trainingFraction_%s-budgetFn_%s-leafPrivacyFraction_%s"
                    % (
                        dataset,
                        float2str(training_fraction),
                        budget_fn,
                        float2str(leaf_privacy_fraction),
                    )
                )
                response = batch.submit_job(
                    jobName=jobName,
                    jobDefinition="PrivateDT:3",
                    jobQueue="m4xlarge-256",
                    retryStrategy={"attempts": 1,},
                    timeout={"attemptDurationSeconds": 10000000,},
                    arrayProperties={"size": ARRAY_SIZE,},
                    containerOverrides={
                        "command": ["/user/run.sh"],
                        "vcpus": 1,
                        "memory": MEMORY,
                        "environment": [
                            {"name": "OUT_DIRECTORY", "value": OUT_DIRECTORY},
                            {
                                "name": "TRAINING_FRACTION",
                                "value": str(training_fraction),
                            },
                            {
                                "name": "LEAF_PRIVACY_FRACTION",
                                "value": str(leaf_privacy_fraction),
                            },
                            {"name": "DATASET", "value": dataset},
                            {"name": "BUDGET_FN", "value": budget_fn},
                            {"name": "START_INDEX", "value": str(START_IDX)},
                        ],
                    },
                )
                print("Job name %s has ID %s" % (jobName, response["jobId"]))
