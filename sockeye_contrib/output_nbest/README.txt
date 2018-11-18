# output_nbest.py
# Script contributed by Reno Kriz
# University of Pennsylvania
# 2018

This README describes how to output multiple sentences for each source sentence in your test file, using Sockeye's beam search history.

### 1. Store the beam histories

By default, sockeye does not store the entire beam search history. In order to store this, inference needs to be run with `--output-type beam_store`. E.g.:

```
python3 -m sockeye.translate --models <MODEL_FILEPATH> \
                             --input <SOURCE_FILEPATH> \
                             --output <BEAM_HISTORY_FILEPATH> \
                             --output-type beam_store \
                             --beam-size <BEAM_SIZE>
```

### Generate the graphs

After inference, the graphs can be generated with:

```
python3 sockeye_contrib/print_nbest.py \
-d <BEAM_HISTORY_FILEPATH> \
-o <OUTPUT_FILEPATH>
```

After this, your output file contain one tab-separated line generated sentences for each source sentence in the original test file.
