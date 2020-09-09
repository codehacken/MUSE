"""
Combine pairs for training and test to create a single set.
"""

import os
import json
import io
import argparse

from src.evaluation.word_translation import DIC_EVAL_PATH, COM_DIC_EVAL_PATH

def build_dictionary(lang1, lang2, filename, pairs=[], reverse=False,
                     check_pairs=[]):
    path = os.path.join(DIC_EVAL_PATH, filename)
    assert os.path.isfile(path)
    print(path)
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

            if ((word1, word2) not in pairs) and ((word1, word2) not in check_pairs):
                pairs.append((word1, word2))

    return pairs

def write_pairs(filename, pairs, reverse=False):
    path = os.path.join(COM_DIC_EVAL_PATH, filename)
    with io.open(path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            if reverse:
                word2, word1 = pair
            else:
                word1, word2 = pair
            f.write('%s %s\n' % (word1, word2))

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
    tr_pairs = build_dictionary(lang1, lang2, filename_f)

    filename_b = '%s-%s.0-5000.txt' % (params.lang2, params.lang1)
    tr_pairs = build_dictionary(lang1, lang2, filename_b, pairs=tr_pairs, reverse=True)

    write_pairs(filename_f, tr_pairs)
    write_pairs(filename_b, tr_pairs, reverse=True)

    # Testing Files.
    filename_f = '%s-%s.5000-6500.txt' % (params.lang1, params.lang2)
    te_pairs = build_dictionary(lang1, lang2, filename_f, pairs=[], check_pairs=tr_pairs)

    filename_b = '%s-%s.5000-6500.txt' % (params.lang2, params.lang1)
    te_pairs = build_dictionary(lang1, lang2, filename_b, pairs=te_pairs,
                                reverse=True, check_pairs=tr_pairs)

    write_pairs(filename_f, te_pairs)
    write_pairs(filename_b, te_pairs, reverse=True)
