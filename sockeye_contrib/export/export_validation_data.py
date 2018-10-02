# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import argparse
import logging

import mxnet as mx

from sockeye import constants as C
from sockeye import data_io

DATA_SUFFIX = ".data"
LABEL_SUFFIX = ".label"


def main():
    params = argparse.ArgumentParser(description="Export Sockeye validation data in MXNet str->NDArray dict format.")
    params.add_argument("--shard", "-s", required=True,
                        help="Data shard representing non-bucketed validation data.")
    params.add_argument("--output", "-o", required=True,
                        help="Output prefix, will write `OUTPUT%s' and `OUTPUT%s'." % (DATA_SUFFIX, LABEL_SUFFIX))
    args = params.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logging.info("Loading validation data from shard `%s'." % args.shard)
    dataset = data_io.ParallelDataSet.load(args.shard)

    # No bucketing is equivalent to bucket 0 containing everything.
    data = {C.SOURCE_NAME: dataset.source[0], C.TARGET_NAME: dataset.target[0]}
    data_fname = args.output + DATA_SUFFIX
    logging.info("Writing data dict (%s %s, %s %s) to `%s'." % (C.SOURCE_NAME,
                                                                dataset.source[0].shape,
                                                                C.TARGET_NAME,
                                                                dataset.target[0].shape,
                                                                data_fname))
    mx.nd.save(data_fname, data)

    label = {C.TARGET_LABEL_NAME: dataset.label[0]}
    label_dict = args.output + LABEL_SUFFIX
    logging.info("Writing label dict (%s %s) to `%s'." % (C.TARGET_LABEL_NAME, dataset.label[0].shape, label_dict))
    mx.nd.save(label_dict, label)


if __name__ == "__main__":
    main()
