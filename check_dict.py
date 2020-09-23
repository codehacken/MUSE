"""
Combine pairs for training and test to create a single set.
"""

import os
import json
import io
import argparse

from src.evaluation.word_translation import DIC_EVAL_PATH, COM_DIC_EVAL_PATH

def build_dictionary(lang1, lang2, filename, pairs=[], reverse=False):
    path = os.path.join(COM_DIC_EVAL_PATH, filename)
    assert os.path.isfile(path)
    print(path)
    word_dict = []
    with io.open(path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            assert line == line.lower()
            parts = line.rstrip().split()
            if len(parts) < 2:
                logger.warning("Could not parse line %s (%i)", line, index)
                continue
            if reverse:
                word2, word1 = parts
            else:
                word1, word2 = parts

            if word1 not in word_dict:
                word_dict.append(word1)

            # Build Common Pairs.
            if (word1, word2) not in pairs:
                pairs.append((word1, word2))

    return pairs, word_dict

def check_pairs(tr_pairs_set, te_pair, prt_str, vocab=[], te_vocab=[]):
    cnt = 0
    for tr_pairs in tr_pairs_set:
        for pair in tr_pairs:
            if pair in te_pair or (pair[0] in vocab and pair[0] in te_vocab):
                cnt += 1

    print("Common Pairs {}: {} Total: {}".format(prt_str, cnt, len(te_pair)))

# Main function.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine Training pairs')
    parser.add_argument("--lang1", type=str, default='en', help="Source language")
    parser.add_argument("--lang2", type=str, default='es', help="Target language")
    params = parser.parse_args()

    lang1 = params.lang1
    lang2 = params.lang2
    print("Language 1: %s, Language 2: %s" % (params.lang1, params.lang2))

    # Training Files.
    filename_f = '%s-%s.0-5000.txt' % (params.lang1, params.lang2)
    f_tr_pairs, f_tr_dict = build_dictionary(lang1, lang2, filename_f, pairs=[])

    filename_b = '%s-%s.0-5000.txt' % (params.lang2, params.lang1)
    b_tr_pairs, b_tr_dict = build_dictionary(lang1, lang2, filename_b,  pairs=[],
                                          reverse=True)

    # Testing Files.
    filename_f = '%s-%s.5000-6500.txt' % (params.lang1, params.lang2)
    f_te_pairs, f_te_dict = build_dictionary(lang1, lang2, filename_f,  pairs=[])

    filename_b = '%s-%s.5000-6500.txt' % (params.lang2, params.lang1)
    b_te_pairs, b_te_dict = build_dictionary(lang1, lang2, filename_b, pairs=[], reverse=True)

    # Check if there are similar pairs.
    check_pairs([f_tr_pairs], f_te_pairs, "F->F", f_tr_dict, f_te_dict)
    check_pairs([b_tr_pairs], f_te_pairs, "B->F", b_tr_dict, f_te_dict)
    check_pairs([f_tr_pairs], b_te_pairs, "F->B", f_tr_dict, b_te_dict)
    check_pairs([b_tr_pairs], b_te_pairs, "B->B", b_tr_dict, b_te_dict)
