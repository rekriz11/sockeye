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
import nnvm


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
    args = params.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logging.info("Loading symbol file `%s'." % args.symbol)
    symbol = mx.sym.load(args.symbol)

    logging.info("Loading params file `%s'." % args.params)
    save_dict = mx.nd.load(args.params)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(":", 1)
        if tp == "arg":
            arg_params[name] = v
        if tp == "aux":
            aux_params[name] = v

    logging.info("Loading data file `%s'." % args.data)
    data = mx.nd.load(args.data)

    logging.info("Loading label file `%s'." % args.label)
    label = mx.nd.load(args.label)

    sym, params = nnvm.frontend.from_mxnet(symbol, arg_params=arg_params, aux_params=aux_params)


if __name__ == "__main__":
    main()
