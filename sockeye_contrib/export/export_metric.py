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
import json
import logging

from sockeye import constants as C
from sockeye import training


def main():
    params = argparse.ArgumentParser(description="Export Sockeye metric as MXNet EvalMetric config (JSON file).")
    params.add_argument("--metric", "-m", default=C.PERPLEXITY, choices=[C.ACCURACY, C.PERPLEXITY],
                        help="Name of metric to export. Default: %(default)s.")
    params.add_argument("--output", "-o", required=True,
                        help="Output file (`metric.json' or similar).")
    args = params.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logging.info("Creating metric named `%s'." % args.metric)
    metric = training.EarlyStoppingTrainer._create_eval_metric(args.metric)

    logging.info("Generating EvalMetric config.")
    config = metric.get_config()

    logging.info("Writing config as JSON file `%s'." % args.output)
    with open(args.output, "w") as out:
        json.dump(config, out)


if __name__ == "__main__":
    main()
