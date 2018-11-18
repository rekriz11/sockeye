###########################################################################
# Created by Reno Kriz, University of Pennsylvania
# This code can be used to output multiple sentences for each
# source sentence in your test file, using Sockeye's beam search history.
#
### 1. Store the beam histories
#
# By default, sockeye does not store the entire beam search history.
# In order to store this, inference needs to be run with
# `--output-type beam_store`. E.g.:
#
# ```
# python3 -m sockeye.translate --models <MODEL_FILEPATH> \
#                              --input <SOURCE_FILEPATH> \
#                              --output <BEAM_HISTORY_FILEPATH> \
#                              --output-type beam_store \
#                              --beam-size <BEAM_SIZE>
# ```
#
### Generate the graphs
#
# After inference, the graphs can be generated with:
#
# ```
# python3 sockeye_contrib/output_nbest.py \
# -d <BEAM_HISTORY_FILEPATH> \
# -o <OUTPUT_FILEPATH>
# ```
#
# Your output file contain one tab-separated line generated sentences for
# each source sentence in the original test file.
###########################################################################

import argparse
import os
import json
import operator

## Pad token used in sockeye, used to filter out pad tokens from the graph
PAD_TOKEN = "<pad>"
        
## Extracts all partial and complete sentences from beam history
def collect_candidates(input_data, include_pad=False):
    candidates = []

    with open(input_data) as beams:       
        for i, line in enumerate(beams):
            candidate_dicts = []
            ## Each sentence starts with the start token
            candidate_dicts.append({0:[['<s>'], 0]})
            
            beam = json.loads(line)

            for j in range(len(beam["predicted_tokens"])):
                cand_dict = dict()
                for k in range(len(beam["predicted_tokens"][j])):
                    current_token = beam["predicted_tokens"][j][k]
                    ## Filters out padded tokens
                    if not include_pad and current_token == PAD_TOKEN:
                        continue
                    parent_id = beam["parent_ids"][j][k]
                    score = beam["normalized_scores"][j][k]

                    current_sentence = candidate_dicts[j][parent_id][0] + [current_token]
                    cand_dict[k] = [current_sentence, score]

                candidate_dicts.append(cand_dict)
            candidates.append(candidate_dicts)
    return candidates

## Extracts complete sentences, and sorts them by score
def find_completes(candidates):
    sentences = []
    for cands in candidates:
        sents = dict()
        for cand_dict in cands:
            for id,sent in cand_dict.items():
                score = sent[1]
                if sent[0][len(sent[0])-1] == '</s>':
                    sents[" ".join(sent[0][1:-1])] = score

        sorted_sents = sorted(sents.items(), key=operator.itemgetter(1))
        sorted_sents = [s[0] for s in sorted_sents]
        sentences.append(sorted_sents)
    return sentences

## Outputs sentences to file
def output_sentences(sentences, output_file):
    with open(output_file, 'w', encoding='utf8') as f:
        for sents in sentences:
            f.write("\t".join(sents) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate nbest sentences")
    parser.add_argument(
        "-d", "--data", type=str, required=True,
        help="path to the beam search data file")
    parser.add_argument(
        "-o", "--output_file", type=str, required=True,
        help="path to the output file")
    parser.add_argument('--pad', dest='include_pad', action='store_true')
    parser.add_argument('--no-pad', dest='include_pad', action='store_false')
    parser.set_defaults(include_pad=False)
    args = parser.parse_args()

    candidates = collect_candidates(args.data, include_pad=args.include_pad)
    sentences = find_completes(candidates)
    output_sentences(sentences, args.output_file)

if __name__ == "__main__":
    main()
