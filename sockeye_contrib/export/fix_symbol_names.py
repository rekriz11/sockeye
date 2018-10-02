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

import logging
import sys

MAX_RENAME = 100

def main():

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Fixing duplicate symbol names.")

    names = set()
    for line in sys.stdin:
        if "\"name\":" in line:
            k, v = line.split(":")
            name = v.strip(" \",\n")
            # Repeat name line
            if name in names:
                for i in range(MAX_RENAME):
                    new_name = name + str(i)
                    if new_name in names:
                        if i == MAX_RENAME - 1:
                            raise RuntimeError("Too many renames: %s, %d" % (name, MAX_RENAME))
                        continue
                    logging.info("Rename: %s -> %s" % (name, new_name))
                    names.add(new_name)
                    sys.stdout.write("%s: \"%s\",\n" % (k, new_name))
                    break
            else:
                # New name line
                names.add(name)
                sys.stdout.write(line)
        else:
            # Non-name line
            sys.stdout.write(line)


if __name__ == "__main__":
    main()
