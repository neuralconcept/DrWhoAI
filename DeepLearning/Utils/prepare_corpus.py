# coding=utf-8
"""
Build vocab with a set max vocab size.
Build token ids given the vocab.
Do get_data.py first.
"""
# from __future__ import unicode_literals, print_function, division

import os
import subprocess
import re
from unidecode import unidecode
from nltk import word_tokenize

from DeepLearning.Utils import CACHE_DIR
from DeepLearning.Utils.data_utils import create_vocabulary, data_to_token_ids
from DeepLearning.Utils.genTrainingData import PERSON_FILENAME, DOCTOR_FILENAME, TRAIN_SUFFIX, DEV_SUFFIX
from DeepLearning.Utils.genTrainingData import PERSON_PATH, DOCTOR_PATH, PERSON_TRAIN_PATH, PERSON_DEV_PATH, DOCTOR_TRAIN_PATH, DOCTOR_DEV_PATH

# TODO: integrate all this in AIBot.py
PERSON_VOCAB_FILENAME = "person.vocab"
DOCTOR_VOCAB_FILENAME = "doctor.vocab"

PERSON_VOCAB_MAX = 20000
DOCTOR_VOCAB_MAX = 20000

IDS_SUFFIX = ".ids"

PERSON_VOCAB_PATH = os.path.join(CACHE_DIR, PERSON_VOCAB_FILENAME)
DOCTOR_VOCAB_PATH = os.path.join(CACHE_DIR, DOCTOR_VOCAB_FILENAME)

PERSON_TRAIN_IDS_PATH = os.path.join(CACHE_DIR, "person" + TRAIN_SUFFIX + IDS_SUFFIX)
PERSON_DEV_IDS_PATH = os.path.join(CACHE_DIR, "person" + DEV_SUFFIX + IDS_SUFFIX)
DOCTOR_TRAIN_IDS_PATH = os.path.join(CACHE_DIR, "doctor" + TRAIN_SUFFIX + IDS_SUFFIX)
DOCTOR_DEV_IDS_PATH = os.path.join(CACHE_DIR, "doctor" + DEV_SUFFIX + IDS_SUFFIX)

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")


def _tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens + lower()."""
    words = []
    for space_separated_fragment in sentence.lower().strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def tokenizer(sentence): # TODO: not working for apostrophes
    sentence = sentence.strip().lower()
    if type(sentence) != unicode:
        sentence = unicode(sentence, encoding='utf-8', errors='replace')
    sentence = unidecode(sentence)
    sentence = sentence.replace("' ", "'")
    return word_tokenize(sentence)


def build_vocab():
    create_vocabulary(PERSON_VOCAB_PATH, PERSON_PATH, PERSON_VOCAB_MAX, tokenizer=tokenizer)
    create_vocabulary(DOCTOR_VOCAB_PATH, DOCTOR_PATH, DOCTOR_VOCAB_MAX, tokenizer=tokenizer)

    print( subprocess.check_output(['wc', '-l', PERSON_VOCAB_PATH]) )
    print( subprocess.check_output(['wc', '-l', DOCTOR_VOCAB_PATH]) )


def build_ids():
    data_to_token_ids(PERSON_TRAIN_PATH, PERSON_TRAIN_IDS_PATH, PERSON_VOCAB_PATH, tokenizer=tokenizer)
    data_to_token_ids(PERSON_DEV_PATH, PERSON_DEV_IDS_PATH, PERSON_VOCAB_PATH, tokenizer=tokenizer)
    data_to_token_ids(DOCTOR_TRAIN_PATH, DOCTOR_TRAIN_IDS_PATH, DOCTOR_VOCAB_PATH, tokenizer=tokenizer)
    data_to_token_ids(DOCTOR_DEV_PATH, DOCTOR_DEV_IDS_PATH, DOCTOR_VOCAB_PATH, tokenizer=tokenizer)

    print( subprocess.check_output(['wc', '-l', PERSON_TRAIN_IDS_PATH]) )
    print( subprocess.check_output(['wc', '-l', PERSON_DEV_IDS_PATH]) )
    print( subprocess.check_output(['wc', '-l', DOCTOR_TRAIN_IDS_PATH]) )
    print( subprocess.check_output(['wc', '-l', DOCTOR_DEV_IDS_PATH]) )

def split_parallel_set(split_size=30000):
    # split the dataset into train and dev sets (i'm using shell scripts)

    subprocess.call(['split', '-l', str(split_size), PERSON_PATH])
    subprocess.call(['mv', 'xaa', PERSON_TRAIN_PATH])
    subprocess.call(['mv', 'xab', PERSON_DEV_PATH])

    subprocess.call(['split', '-l', str(split_size), DOCTOR_PATH])
    subprocess.call(['mv', 'xaa', DOCTOR_TRAIN_PATH])
    subprocess.call(['mv', 'xab', DOCTOR_DEV_PATH])

    print( subprocess.check_output(['wc', '-l', PERSON_TRAIN_PATH]) )
    print( subprocess.check_output(['wc', '-l', PERSON_DEV_PATH]) )
    print( subprocess.check_output(['wc', '-l', DOCTOR_TRAIN_PATH]) )
    print( subprocess.check_output(['wc', '-l', DOCTOR_DEV_PATH]) )



if __name__ == '__main__':
    split_parallel_set()
    build_vocab()
    build_ids()