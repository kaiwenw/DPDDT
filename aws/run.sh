#!/usr/bin/env bash

export SEED=$AWS_BATCH_JOB_ARRAY_INDEX

cd private-DPDDT/build
./single_run

# Copy output.out to S3
~/.local/bin/aws s3 cp *.csv $OUT_DIRECTORY
