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

import mxnet as mx


def main():
    params = argparse.ArgumentParser(description="Load pre-trained MXNet model and score validation data.")
    params.add_argument("--symbol", "-s", required=True,
                        help="MXNet symbol file (named `MODEL-symbol.json' or similar).")
    params.add_argument("--params", "-p", required=True,
                        help="MXNet params file (named `MODEL-CHECKPOINT.params' or similar).")
    params.add_argument("--metric", "-m", required=True,
                        help="MXNet EvalMetric config (JSON file).")
    params.add_argument("--data", "-d", required=True,
                        help="Validation data file (MXNet str->NDArray dict format).")
    params.add_argument("--label", "-l", required=True,
                        help="Validation label file (MXNet str->NDArray dict format).")
    params.add_argument("--batch-size", "-b", type=int, default=64,
                        help="Validation target file. Default: %(default)s.")
    args = params.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logging.info("Loading metric file `%s'." % args.metric)
    with open(args.metric, "r") as inp:
        config = json.load(inp)
    metric = mx.metric.create(**config)

    logging.info("Loading data file `%s'." % args.data)
    data = mx.nd.load(args.data)

    logging.info("Loading label file `%s'." % args.label)
    label = mx.nd.load(args.label)

    # Gluon doesn't distinguish between data and label inputs, so we specify
    # the keys in both for input_names.
    input_names = list(data.keys()) + list(label.keys())
    logging.info("Creating Gluon SymbolBlock from `%s' and `%s' using input names: %s."
                 % (args.symbol, args.params, " ".join(input_names)))
    model = mx.gluon.nn.SymbolBlock.imports(args.symbol, input_names, args.params)

    # When a SymbolBlock is defined with N inputs, forward() accepts N
    # positional arguments in the same order.
    # TODO: Currently one giant batch regardless of what user specifies.
    logging.info("Running forward.")
    preds = model.forward(*data.values(), *label.values())

    # TODO: What is the canonical way to handle metrics with Gluon?
    logging.info("Computing score.")
    labels = mx.nd.reshape(list(label.values())[0], shape=(-1,))
    metric.update(labels, preds)
    results = metric.get_name_value()
    for k, v in results:
        print(k, v)


if __name__ == "__main__":
    main()
