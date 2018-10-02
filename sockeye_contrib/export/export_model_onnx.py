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


def main():
    params = argparse.ArgumentParser(description="Load pre-trained MXNet model and score validation data.")
    params.add_argument("--symbol", "-s", required=True,
                        help="MXNet symbol file (named `MODEL-symbol.json' or similar).")
    params.add_argument("--params", "-p", required=True,
                        help="MXNet params file (named `MODEL-CHECKPOINT.params' or similar).")
    params.add_argument("--data", "-d", required=True,
                        help="Validation data file (MXNet str->NDArray dict format).")
    params.add_argument("--label", "-l", required=True,
                        help="Validation label file (MXNet str->NDArray dict format).")
    params.add_argument("--output", "-o", required=True,
                        help="Output file (`MODEL.onnx' or similar).")
    args = params.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logging.info("Loading data file `%s'." % args.data)
    data = mx.nd.load(args.data)

    logging.info("Loading label file `%s'." % args.label)
    label = mx.nd.load(args.label)

    # TODO: Why does export_model reverse these?
    input_shape = [v.shape for v in reversed(list(data.values()))] + [v.shape for v in label.values()]
    mx.contrib.onnx.export_model(args.symbol, args.params, input_shape, onnx_file_path=args.output)


if __name__ == "__main__":
    main()
