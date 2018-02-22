"""
60 mins. (4 cores / 16 GB RAM / 60 minutes run-time / 1 GB scratch and output disk space)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import datetime
import os
import gc
import glob
import mmh3
import nltk
import numpy as np
import pandas as pd
import pickle as pkl
import shutil
import spacy
import time
import tensorflow as tf
import re
import string
import sys
from collections import Counter, defaultdict
from hashlib import md5

from fastcache import clru_cache as lru_cache

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk import ToktokTokenizer

from multiprocessing import Pool

from six.moves import range
from six.moves import zip

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelBinarizer

from keras.preprocessing.sequence import pad_sequences

from metrics import rmse
from topk import top_k_selector
from xnn import XNN
from utils import _get_logger, _makedirs, _timestamp


##############################################################################
_makedirs("./log")
logger = _get_logger("./log", "hyperopt-%s.log" % _timestamp())

##############################################################################

RUNNING_MODE = "validation"
# RUNNING_MODE = "submission"
DEBUG = False
DUMP_DATA = True
USE_PREPROCESSED_DATA = True

USE_MULTITHREAD = False
if RUNNING_MODE == "submission":
    N_JOBS = 4
else:
    N_JOBS = 4
NUM_PARTITIONS = 32

DEBUG_SAMPLE_NUM = 200000
LRU_MAXSIZE = 2 ** 16

#######################################
# File
MISSING_VALUE_STRING = "MISSINGVALUE"
DROP_ZERO_PRICE = True
#######################################


# Preprocessing
USE_SPACY_TOKENIZER = False
USE_NLTK_TOKENIZER = False
USE_KAGGLE_TOKENIZER = False
# default: '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
KERAS_TOKENIZER_FILTERS = '\'!"#%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n'
KERAS_TOKENIZER_FILTERS = ""
KERAS_SPLIT = " "

USE_LEMMATIZER = False
USE_STEMMER = False
USE_CLEAN = True

WORDREPLACER_DICT = {
    "bnwt": "brand new with tags",
    "nwt": "new with tags",
    "bnwot": "brand new without tags",
    "nwot": "new without tags",
    "bnip": "brand new in packet",
    "nip": "new in packet",
    "bnib": "brand new in box",
    "nib": "new in box",
    "mib": "mint in box",
    "mwob": "mint without box",
    "mip": "mint in packet",
    "mwop": "mint without packet"
}

BRAND_NAME_PATTERN_LIST = [
    ("nike", "nike"),
    ("pink", "pink"),
    ("apple", "iphone|ipod|ipad|iwatch|apple|mac"),
    ("victoria's secret", "victoria"),
    ("lularoe", "lularoe"),
    ("nintendo", "nintendo"),
    ("lululemon", "lululemon"),
    ("forever 21", "forever\s+21|forever\s+twenty\s+one"),
    ("michael kors", "michael\s+kors"),
    ("american eagle", "american\s+eagle"),
    ("rae dunn", "rae dunn"),
]

# word count |   #word
#    >= 1    |  195523
#    >= 2    |   93637
#    >= 3    |   67498
#    >= 4    |   56265
#    >= 5    |   49356
MAX_NUM_WORDS = 80000
MAX_NUM_BIGRAMS = 50000
MAX_NUM_TRIGRAMS = 50000
MAX_NUM_SUBWORDS = 20000

NUM_TOP_WORDS_NAME = 50
NUM_TOP_WORDS_ITEM_DESC = 50

MAX_CATEGORY_NAME_LEN = 3

EXTRACTED_BIGRAM = True
EXTRACTED_TRIGRAM = True
EXTRACTED_SUBWORD = False
VOCAB_HASHING_TRICK = False

######################

####################################################################
HYPEROPT_MAX_EVALS = 1

param_space_com = {
    "RUNNING_MODE": RUNNING_MODE,
    # size for the attention block
    "MAX_NUM_WORDS": MAX_NUM_WORDS,
    "MAX_NUM_BIGRAMS": MAX_NUM_BIGRAMS,
    "MAX_NUM_TRIGRAMS": MAX_NUM_TRIGRAMS,
    "MAX_NUM_SUBWORDS": MAX_NUM_SUBWORDS,

    "model_dir": "./weights",

    "item_condition_size": 5,
    "shipping_size": 1,
    "num_vars_size": 3,
    # pad_sequences
    "pad_sequences_padding": "post",
    "pad_sequences_truncating": "post",
    # optimization
    "optimizer_clipnorm": 1.,
    "batch_size_train": 512,
    "batch_size_inference": 512*2,
    "shuffle_with_replacement": False,
    # CyclicLR
    "t_mul": 1,
    "snapshot_every_num_cycle": 128,
    "max_snapshot_num": 14,
    "snapshot_every_epoch": 4,  # for t_mult != 1
    "eval_every_num_update": 1000,
    # static param
    "random_seed": 2018,
    "n_folds": 1,
    "validation_ratio": 0.4,
}

param_space_best = {

    #### params for input
    # bigram/trigram/subword
    "use_bigram": True,
    "use_trigram": True,
    "use_subword": False,

    # seq len
    "max_sequence_length_name": 10,
    "max_sequence_length_item_desc": 50,
    "max_sequence_length_category_name": 10,
    "max_sequence_length_item_desc_subword": 45,

    #### params for embed
    "embedding_dim": 250,
    "embedding_dropout": 0.,
    "embedding_mask_zero": False,
    "embedding_mask_zero_subword": False,

    #### params for encode
    "encode_method": "fasttext",
    # cnn
    "cnn_num_filters": 16,
    "cnn_filter_sizes": [2, 3],
    "cnn_timedistributed": False,
    # rnn
    "rnn_num_units": 16,
    "rnn_cell_type": "gru",
    #### params for attend
    "attend_method": "ave",

    #### params for predict
    # deep
    "enable_deep": True,
    # fm
    "enable_fm_first_order": True,
    "enable_fm_second_order": True,
    "enable_fm_higher_order": False,
    # fc block
    "fc_type": "fc",
    "fc_dim": 64,
    "fc_dropout": 0.,

    #### params for optimization
    "optimizer_type": "nadam",  # "nadam",  # ""lazyadam", "nadam"
    "max_lr_exp": 0.005,
    "lr_decay_each_epoch_exp": 0.9,
    "lr_jump_exp": True,
    "max_lr_cosine": 0.005,
    "base_lr": 0.00001,  # minimum lr
    "lr_decay_each_epoch_cosine": 0.5,
    "lr_jump_rate": 1.,
    "snapshot_before_restarts": 4,
    "beta1": 0.975,
    "beta2": 0.999,
    "schedule_decay": 0.004,
    # "lr_schedule": "exponential_decay",
    "lr_schedule": "cosine_decay_restarts",
    "epoch": 4,
    # CyclicLR
    "num_cycle_each_epoch": 8,

    #### params ensemble
    "enable_snapshot_ensemble": True,
    "n_runs": 2,

}
param_space_best.update(param_space_com)
if RUNNING_MODE == "submission":
    EXTRACTED_BIGRAM = param_space_best["use_bigram"]
    EXTRACTED_SUBWORD = param_space_best["use_subword"]

param_space_hyperopt = param_space_best

int_params = [
    "max_sequence_length_name",
    "max_sequence_length_item_desc",
    "max_sequence_length_item_desc_subword",
    "max_sequence_length_category_name",
    "embedding_dim", "embedding_dim",
    "cnn_num_filters", "rnn_num_units", "fc_dim",
    "epoch", "n_runs",
    "num_cycle_each_epoch", "t_mul", "snapshot_every_num_cycle",
]
int_params = set(int_params)

if DEBUG:
    param_space_hyperopt["num_cycle_each_epoch"] = param_space_best["num_cycle_each_epoch"] = 2
    param_space_hyperopt["snapshot_every_num_cycle"] = param_space_best["snapshot_every_num_cycle"] = 1
    param_space_hyperopt["batch_size_train"] = param_space_best["batch_size_train"] = 512
    param_space_hyperopt["batch_size_inference"] = param_space_best["batch_size_inference"] = 512


####################################################################################################
########################################### NLP ####################################################
####################################################################################################
def mmh3_hash_function(x):
    return mmh3.hash(x, 42, signed=True)


def md5_hash_function(x):
    return int(md5(x.encode()).hexdigest(), 16)


@lru_cache(LRU_MAXSIZE)
def hashing_trick(string, n, hash_function="mmh3"):
    if hash_function == "mmh3":
        hash_function = mmh3_hash_function
    elif hash_function == "md5":
        hash_function = md5_hash_function
    i = (hash_function(string) % n) + 1
    return i


# 5.67 µs ± 78.2 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
@lru_cache(LRU_MAXSIZE)
def get_subword_for_word_all(word, n1=3, n2=6):
    z = []
    z_append = z.append
    word = "*" + word + "*"
    l = len(word)
    z_append(word)
    for k in range(n1, n2 + 1):
        for i in range(l - k + 1):
            z_append(word[i:i + k])
    return z


@lru_cache(LRU_MAXSIZE)
def get_subword_for_word_all0(word, n1=3, n2=6):
    z = []
    z_append = z.append
    word = "*" + word + "*"
    l = len(word)
    z_append(word)
    if l > n1:
        n2 = min(n2, l - 1)
        for i in range(l - n1 + 1):
            for k in range(n1, n2 + 1):
                if 2 * i + n2 < l:
                    z_append(word[i:(i + k)])
                    if i == 0:
                        z_append(word[-(i + k + 1):])
                    else:
                        z_append(word[-(i + k + 1):-i])
                else:
                    if 2 * i + k < l:
                        z_append(word[i:(i + k)])
                        z_append(word[-(i + k + 1):-i])
                    elif 2 * (i - 1) + n2 < l:
                        z_append(word[i:(i + k)])
    return z


# 3.44 µs ± 101 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
@lru_cache(LRU_MAXSIZE)
def get_subword_for_word0(word, n1=4, n2=5, include_self=False):
    """only extract the prefix and suffix"""
    l = len(word)
    n1 = min(n1, l)
    n2 = min(n2, l)
    z1 = [word[:k] for k in range(n1, n2 + 1)]
    z2 = [word[-k:] for k in range(n1, n2 + 1)]
    z = z1 + z2
    if include_self:
        z.append(word)
    return z


# 2.49 µs ± 104 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
@lru_cache(LRU_MAXSIZE)
def get_subword_for_word(word, n1=3, n2=6, include_self=False):
    """only extract the prefix and suffix"""
    z = []
    if len(word) >= n1:
        word = "*" + word + "*"
        l = len(word)
        n1 = min(n1, l)
        n2 = min(n2, l)
        # bind method outside of loop to reduce overhead
        # https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/feature_extraction/text.py#L144
        z_append = z.append
        if include_self:
            z_append(word)
        for k in range(n1, n2 + 1):
            z_append(word[:k])
            z_append(word[-k:])
    return z


# 564 µs ± 14.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
def get_subword_for_list0(input_list, n1=4, n2=5):
    subword_lst = [get_subword_for_word(w, n1, n2) for w in input_list]
    subword_lst = [w for ws in subword_lst for w in ws]
    return subword_lst


# 505 µs ± 15 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
def get_subword_for_list(input_list, n1=4, n2=5):
    subwords = []
    subwords_extend = subwords.extend
    for w in input_list:
        subwords_extend(get_subword_for_word(w, n1, n2))
    return subwords


@lru_cache(LRU_MAXSIZE)
def get_subword_for_text(text, n1=4, n2=5):
    return get_subword_for_list(text.split(" "), n1, n2)


stopwords = [
    "and",
    "the",
    "for",
    "a",
    "in",
    "to",
    "is",
    # "s",
    "of",
    "i",
    "on",
    "it",
    "you",
    "your",
    "are",
    "this",
    "my",
]
stopwords = set(stopwords)


# spacy model
class SpacyTokenizer(object):
    def __init__(self):
        self.nlp = spacy.load("en", disable=["parser", "tagger", "ner"])

    def tokenize(self, text):
        tokens = [tok.lower_ for tok in self.nlp(text)]
        # tokens = get_valid_words(tokens)
        return tokens


LEMMATIZER = nltk.stem.wordnet.WordNetLemmatizer()
STEMMER = nltk.stem.snowball.EnglishStemmer()
TOKTOKTOKENIZER = ToktokTokenizer()


# SPACYTOKENIZER = SpacyTokenizer()


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def get_valid_words(sentence):
    res = [w.strip() for w in sentence]
    return [w for w in res if w]


@lru_cache(LRU_MAXSIZE)
def stem_word(word):
    return STEMMER.stem(word)


@lru_cache(LRU_MAXSIZE)
def lemmatize_word(word, pos=wordnet.NOUN):
    return LEMMATIZER.lemmatize(word, pos)


def stem_sentence(sentence):
    return [stem_word(w) for w in get_valid_words(sentence)]


def lemmatize_sentence(sentence):
    res = []
    sentence_ = get_valid_words(sentence)
    for word, pos in pos_tag(sentence_):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatize_word(word, pos=wordnet_pos))
    return res


def stem_lemmatize_sentence(sentence):
    return [stem_word(word) for word in lemmatize_sentence(sentence)]


TRANSLATE_MAP = maketrans(KERAS_TOKENIZER_FILTERS, KERAS_SPLIT * len(KERAS_TOKENIZER_FILTERS))


def get_tokenizer():
    if USE_LEMMATIZER and USE_STEMMER:
        return stem_lemmatize_sentence
    elif USE_LEMMATIZER:
        return lemmatize_sentence
    elif USE_STEMMER:
        return stem_sentence
    else:
        return get_valid_words


tokenizer = get_tokenizer()


#
# https://stackoverflow.com/questions/21883108/fast-optimize-n-gram-implementations-in-python
# @lru_cache(LRU_MAXSIZE)
# 40.1 µs ± 918 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
def get_ngrams0(words, ngram_value):
    # # return list
    ngrams = [" ".join(ngram) for ngram in zip(*[words[i:] for i in range(ngram_value)])]
    # return generator (10x faster)
    # ngrams = (" ".join(ngram) for ngram in zip(*[words[i:] for i in range(ngram_value)]))
    return ngrams


# 36.2 µs ± 1.04 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
def get_ngrams(words, ngram_value):
    tokens = []
    tokens_append = tokens.append
    for i in range(ngram_value):
        tokens_append(words[i:])
    ngrams = []
    ngrams_append = ngrams.append
    space_join = " ".join
    for ngram in zip(*tokens):
        ngrams_append(space_join(ngram))
    return ngrams


def get_bigrams(words):
    return get_ngrams(words, 2)


def get_trigrams(words):
    return get_ngrams(words, 3)


@lru_cache(LRU_MAXSIZE)
# 68.8 µs ± 1.86 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
def get_ngrams_range(text, ngram_range):
    unigrams = text.split(" ")
    ngrams = []
    ngrams_extend = ngrams.extend
    for i in range(ngram_range[0], ngram_range[1] + 1):
        ngrams_extend(get_ngrams(unigrams, i))
    return ngrams


# 69.6 µs ± 1.45 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
def get_ngrams_range0(text, ngram_range):
    unigrams = text.split(" ")
    res = []
    for i in ngram_range:
        res += get_ngrams(unigrams, i)
    res += unigrams
    return res


@lru_cache(LRU_MAXSIZE)
def stem(s):
    return STEMMER.stem(s)


tags = re.compile(r'<.+?>')
whitespace = re.compile(r'\s+')
non_letter = re.compile(r'\W+')


@lru_cache(LRU_MAXSIZE)
def clean_text(text):
    # text = text.lower()
    text = non_letter.sub(' ', text)

    tokens = []

    for t in text.split():
        # if len(t) <= 2 and not t.isdigit():
        #     continue
        if t in stopwords:
            continue
        t = stem(t)
        tokens.append(t)

    text = ' '.join(tokens)

    text = whitespace.sub(' ', text)
    text = text.strip()
    return text.split(" ")


@lru_cache(LRU_MAXSIZE)
def tokenize(text):
    if USE_NLTK_TOKENIZER:
        # words = get_valid_words(word_tokenize(text))
        # words = get_valid_words(wordpunct_tokenize(text))
        words = get_valid_words(TOKTOKTOKENIZER.tokenize(text))
    elif USE_SPACY_TOKENIZER:
        words = get_valid_words(SPACYTOKENIZER.tokenize(text))
    elif USE_KAGGLE_TOKENIZER:
        words = clean_text(text)
    else:
        words = tokenizer(text.translate(TRANSLATE_MAP).split(KERAS_SPLIT))
    return words


@lru_cache(LRU_MAXSIZE)
def tokenize_with_subword(text, n1=4, n2=5):
    words = tokenize(text)
    subwords = get_subword_for_list(words, n1, n2)
    return words + subwords


######################################################################
# --------------------------- Processor ---------------------------
## base class
## Most of the processings can be casted into the "pattern-replace" framework
class BaseReplacer:
    def __init__(self, pattern_replace_pair_list=[]):
        self.pattern_replace_pair_list = pattern_replace_pair_list


## deal with word replacement
# 1st solution in CrowdFlower
class WordReplacer(BaseReplacer):
    def __init__(self, replace_dict):
        self.replace_dict = replace_dict
        self.pattern_replace_pair_list = []
        for k, v in self.replace_dict.items():
            # pattern = r"(?<=\W|^)%s(?=\W|$)" % k
            pattern = k
            replace = v
            self.pattern_replace_pair_list.append((pattern, replace))


class MerCariCleaner(BaseReplacer):
    """https://stackoverflow.com/questions/7317043/regex-not-operator
    """

    def __init__(self):
        self.pattern_replace_pair_list = [
            # # remove filters
            # (r'[-!\'\"#&()\*\+,-/:;<=＝>?@\[\\\]^_`{|}~\t\n]+', r""),
            # remove punctuation ".", e.g.,
            (r"(?<!\d)\.(?!\d+)", r" "),
            # iphone 6/6s -> iphone 6 / 6s
            # iphone 6:6s -> iphone 6 : 6s
            (r"(\W+)", r" \1 "),
            # # non
            # (r"[^A-Za-z0-9]+", r" "),
            # 6s -> 6 s
            # 32gb -> 32 gb
            # 4oz -> 4 oz
            # 4pcs -> 4 pcs
            (r"(\d+)([a-zA-Z])", r"\1 \2"),
            # iphone5 -> iphone 5
            # xbox360 -> xbox 360
            # only split those with chars length > 3
            (r"([a-zA-Z]{3,})(\d+)", r"\1 \2"),
        ]


###########################################
def df_lower(df):
    return df.str.lower()


def df_contains(df, pat):
    return df.str.contains(pat).astype(int)


def df_len(df):
    return df.str.len().astype(float)


def df_num_tokens(df):
    return df.str.split().apply(len).astype(float)


def df_in(df, col1, col2):
    def _in(x):
        return x[col1] in x[col2]

    return df.apply(_in, 1).astype(int)


def df_brand_in_name(df):
    return df_in(df, "brand_name", "name")


def df_category1_in_name(df):
    return df_in(df, "category_name1", "name")


def df_category2_in_name(df):
    return df_in(df, "category_name2", "name")


def df_category3_in_name(df):
    return df_in(df, "category_name3", "name")


def df_brand_in_desc(df):
    return df_in(df, "brand_name", "item_desc")


def df_category1_in_desc(df):
    return df_in(df, "category_name1", "item_desc")


def df_category2_in_desc(df):
    return df_in(df, "category_name2", "item_desc")


def df_category3_in_desc(df):
    return df_in(df, "category_name3", "item_desc")


def df_clean(df):
    for pat, repl in MerCariCleaner().pattern_replace_pair_list:
        df = df.str.replace(pat, repl)
    # for pat, repl in WordReplacer(WORDREPLACER_DICT).pattern_replace_pair_list:
    #     df = df.str.replace(pat, repl)
    return df


def df_tokenize(df):
    return df.apply(tokenize)


def df_tokenize_with_subword(df):
    return df.apply(tokenize_with_subword)


def df_get_bigram(df):
    return df.apply(get_bigrams)


def df_get_trigram(df):
    return df.apply(get_trigrams)


def df_get_subword(df):
    return df.apply(get_subword_for_list)


def parallelize_df_func(df, func, num_partitions=NUM_PARTITIONS, n_jobs=N_JOBS):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(n_jobs)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


######################################################################

def load_train_data():
    types_dict_train = {
        'train_id': 'int32',
        'item_condition_id': 'int32',
        'price': 'float32',
        'shipping': 'int8',
        'name': 'str',
        'brand_name': 'str',
        'item_desc': 'str',
        'category_name': 'str',
    }
    df = pd.read_csv('../input/train.tsv', delimiter='\t', low_memory=True, dtype=types_dict_train)
    df.rename(columns={"train_id": "id"}, inplace=True)
    df.rename(columns={"item_description": "item_desc"}, inplace=True)
    if DROP_ZERO_PRICE:
        df = df[df.price > 0].copy()
    price = np.log1p(df.price.values)
    df.drop("price", axis=1, inplace=True)
    df["price"] = price
    df["is_train"] = 1
    df["missing_brand_name"] = df["brand_name"].isnull().astype(int)
    df["missing_category_name"] = df["category_name"].isnull().astype(int)
    missing_ind = np.logical_or(df["item_desc"].isnull(),
                                df["item_desc"].str.lower().str.contains("no\s+description\s+yet"))
    df["missing_item_desc"] = missing_ind.astype(int)
    df["item_desc"][missing_ind] = df["name"][missing_ind]
    gc.collect()
    if DEBUG:
        return df.head(DEBUG_SAMPLE_NUM)
    else:
        return df


def load_test_data(chunksize=350000*2):
    types_dict_test = {
        'test_id': 'int32',
        'item_condition_id': 'int32',
        'shipping': 'int8',
        'name': 'str',
        'brand_name': 'str',
        'item_description': 'str',
        'category_name': 'str',
    }
    chunks = pd.read_csv('../input/test.tsv', delimiter='\t',
                         low_memory=True, dtype=types_dict_test,
                         chunksize=chunksize)
    for df in chunks:
        df.rename(columns={"test_id": "id"}, inplace=True)
        df.rename(columns={"item_description": "item_desc"}, inplace=True)
        df["missing_brand_name"] = df["brand_name"].isnull().astype(int)
        df["missing_category_name"] = df["category_name"].isnull().astype(int)
        missing_ind = np.logical_or(df["item_desc"].isnull(),
                                    df["item_desc"].str.lower().str.contains("no\s+description\s+yet"))
        df["missing_item_desc"] = missing_ind.astype(int)
        df["item_desc"][missing_ind] = df["name"][missing_ind]
        yield df


@lru_cache(1024)
def split_category_name(row):
    grps = row.split("/")
    if len(grps) > MAX_CATEGORY_NAME_LEN:
        grps = grps[:MAX_CATEGORY_NAME_LEN]
    else:
        grps += [MISSING_VALUE_STRING.lower()] * (MAX_CATEGORY_NAME_LEN - len(grps))
    return tuple(grps)


"""
https://stackoverflow.com/questions/3172173/most-efficient-way-to-calculate-frequency-of-values-in-a-python-list

| approach       | american-english, |      big.txt, | time w.r.t. defaultdict |
|                |     time, seconds | time, seconds |                         |
|----------------+-------------------+---------------+-------------------------|
| Counter        |             0.451 |         3.367 |                     3.6 |
| setdefault     |             0.348 |         2.320 |                     2.5 |
| list           |             0.277 |         1.822 |                       2 |
| try/except     |             0.158 |         1.068 |                     1.2 |
| defaultdict    |             0.141 |         0.925 |                       1 |
| numpy          |             0.012 |         0.076 |                   0.082 |
| S.Mark's ext.  |             0.003 |         0.019 |                   0.021 |
| ext. in Cython |             0.001 |         0.008 |                  0.0086 |

code: https://gist.github.com/347000
"""


def get_word_index0(words, max_num, prefix):
    word_counts = defaultdict(int)
    for ws in words:
        for w in ws:
            word_counts[w] += 1
    wcounts = list(word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)
    sorted_voc = [wc[0] for wc in wcounts[:(max_num - 1)]]
    word_index = dict(zip(sorted_voc, range(2, max_num)))
    return word_index


def get_word_index1(words, max_num, prefix):
    word_counts = Counter([w for ws in words for w in ws])
    # """
    wcounts = list(word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)
    sorted_voc = [wc[0] for wc in wcounts]
    del wcounts
    gc.collect()
    # only keep MAX_NUM_WORDS
    sorted_voc = sorted_voc[:(max_num - 1)]
    # note that index 0 is reserved, never assigned to an existing word
    word_index = dict(zip(sorted_voc, range(2, max_num)))
    return word_index


def get_word_index(words, max_num, prefix):
    word_counts = Counter([w for ws in words for w in ws])
    sorted_voc = [w for w, c in word_counts.most_common(max_num - 1)]
    word_index = dict(zip(sorted_voc, range(2, max_num)))
    return word_index


def get_word_index(words, max_num, prefix):
    sorted_voc = top_k_selector.topKFrequent(words, max_num - 1)
    word_index = dict(zip(sorted_voc, range(2, max_num)))
    return word_index


class MyLabelEncoder(object):
    """safely handle unknown label"""

    def __init__(self):
        self.mapper = {}

    def fit(self, X):
        uniq_X = np.unique(X)
        # reserve 0 for unknown
        self.mapper = dict(zip(uniq_X, range(1, len(uniq_X) + 1)))
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _map(self, x):
        return self.mapper.get(x, 0)

    def transform(self, X):
        return list(map(self._map, X))


class MyStandardScaler(object):
    def __init__(self, identity=False, epsilon=1e-8):
        self.identity = identity
        self.mean_ = 0.
        self.scale_ = 1.
        self.epsilon = epsilon

    def fit(self, X):
        if not self.identity:
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
        else:
            self.epsilon = 0.

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        return (X - self.mean_) / (self.scale_ + self.epsilon)

    def inverse_transform(self, X):
        return X * (self.scale_ + self.epsilon) + self.mean_


def preprocess(df, word_index=None, bigram_index=None,
               trigram_index=None, subword_index=None,
               label_encoder=None):
    start_time = time.time()

    #### fill na
    df.fillna(MISSING_VALUE_STRING, inplace=True)
    gc.collect()

    #### to lower case
    df["name"] = df.name.str.lower()
    df["brand_name"] = df.brand_name.str.lower()
    df["category_name"] = df.category_name.str.lower()
    df["item_desc"] = df.item_desc.str.lower()
    gc.collect()
    print("[%.5f] Done df_lower" % (time.time() - start_time))

    #### split category name
    for i, cat in enumerate(zip(*df.category_name.apply(split_category_name))):
        df["category_name%d" % (i + 1)] = cat
        gc.collect()

    #### regex based cleaning
    if USE_CLEAN:
        df["name"] = parallelize_df_func(df["name"], df_clean)
        df["item_desc"] = parallelize_df_func(df["item_desc"], df_clean)
        # df["category_name"] = parallelize_df_func(df["category_name"], df_clean)
        print("[%.5f] Done df_clean" % (time.time() - start_time))
        gc.collect()

    #### tokenize
    # print("   Fitting tokenizer...")
    df["seq_name"] = parallelize_df_func(df["name"], df_tokenize)
    df["seq_item_desc"] = parallelize_df_func(df["item_desc"], df_tokenize)
    # df["seq_brand_name"] = parallelize_df_func(df["brand_name"], df_tokenize)
    # df["seq_category_name"] = parallelize_df_func(df["category_name"], df_tokenize)
    gc.collect()
    print("[%.5f] Done df_tokenize" % (time.time() - start_time))
    df.drop(["name"], axis=1, inplace=True)
    df.drop(["item_desc"], axis=1, inplace=True)
    gc.collect()
    if USE_MULTITHREAD:
        if EXTRACTED_BIGRAM:
            df["seq_bigram_item_desc"] = parallelize_df_func(df["seq_item_desc"], df_get_bigram)
            print("[%.5f] Done df_get_bigram" % (time.time() - start_time))
        if EXTRACTED_TRIGRAM:
            df["seq_trigram_item_desc"] = parallelize_df_func(df["seq_item_desc"], df_get_trigram)
            print("[%.5f] Done df_get_trigram" % (time.time() - start_time))
        if EXTRACTED_SUBWORD:
            df["seq_subword_item_desc"] = parallelize_df_func(df["seq_item_desc"], df_get_subword)
            print("[%.5f] Done df_get_subword" % (time.time() - start_time))
    else:
        if EXTRACTED_BIGRAM:
            df["seq_bigram_item_desc"] = df_get_bigram(df["seq_item_desc"])
            print("[%.5f] Done df_get_bigram" % (time.time() - start_time))
        if EXTRACTED_TRIGRAM:
            df["seq_trigram_item_desc"] = df_get_trigram(df["seq_item_desc"])
            print("[%.5f] Done df_get_trigram" % (time.time() - start_time))
        if EXTRACTED_SUBWORD:
            df["seq_subword_item_desc"] = df_get_subword(df["seq_item_desc"])
            print("[%.5f] Done df_get_subword" % (time.time() - start_time))
    if not VOCAB_HASHING_TRICK:
        if word_index is None:
            ##### word_index
            words = df.seq_name.tolist() + \
                    df.seq_item_desc.tolist()
                    # df.seq_category_name.tolist()
            word_index = get_word_index(words, MAX_NUM_WORDS, "word")
            del words
            gc.collect()
        if EXTRACTED_BIGRAM:
            if bigram_index is None:
                bigrams = df.seq_bigram_item_desc.tolist()
                bigram_index = get_word_index(bigrams, MAX_NUM_BIGRAMS, "bigram")
                del bigrams
                gc.collect()
        if EXTRACTED_TRIGRAM:
            if trigram_index is None:
                trigrams = df.seq_trigram_item_desc.tolist()
                trigram_index = get_word_index(trigrams, MAX_NUM_TRIGRAMS, "trigram")
                del trigrams
                gc.collect()
        if EXTRACTED_SUBWORD:
            if subword_index is None:
                subwords = df.seq_subword_item_desc.tolist()
                subword_index = get_word_index(subwords, MAX_NUM_SUBWORDS, "subword")
                del subwords
                gc.collect()
        print("[%.5f] Done building vocab" % (time.time() - start_time))

        # faster
        # v = range(10000)
        # k = [str(i) for i in v]
        # vocab = dict(zip(k, v))
        # %timeit word2ind(word_lst, vocab)
        # 4.06 µs ± 63.8 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        def word2ind0(word_lst, vocab):
            vect = []
            for w in word_lst:
                if w in vocab:
                    vect.append(vocab[w])
            return vect

        # 4.46 µs ± 77.9 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        def word2ind1(word_lst, vocab):
            vect = [vocab[w] for w in word_lst if w in vocab]
            return vect

        # 13.3 µs ± 99.7 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        def word2ind2(word_lst, vocab):
            vect = []
            for w in word_lst:
                i = vocab.get(w)
                if i is not None:
                    vect.append(i)
            return vect

        # 14.6 µs ± 114 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        def word2ind3(word_lst, vocab):
            return [vocab.get(w, 1) for w in word_lst]

        word2ind = word2ind0

        def wordlist2ind0(word_list_lst, vocab):
            if len(word_list_lst) == 0:
                vect = [[]]
            else:
                vect = []
                for word_list in word_list_lst:
                    vect_ = []
                    for w in word_list:
                        if w in vocab:
                            vect_.append(vocab[w])
                    vect.append(vect_)
            return vect

        wordlist2ind = wordlist2ind0

        def word_lst_to_sequences(word_lst):
            return word2ind(word_lst, word_index)

        def df_word_lst_to_sequences(df):
            return df.apply(word_lst_to_sequences)

        def bigram_lst_to_sequences(word_lst):
            return word2ind(word_lst, bigram_index)

        def df_bigram_lst_to_sequences(df):
            return df.apply(bigram_lst_to_sequences)

        def trigram_lst_to_sequences(word_lst):
            return word2ind(word_lst, trigram_index)

        def df_trigram_lst_to_sequences(df):
            return df.apply(trigram_lst_to_sequences)

        def subword_lst_to_sequences(word_lst):
            return word2ind(word_lst, subword_index)

        def df_subword_lst_to_sequences(df):
            return df.apply(subword_lst_to_sequences)

        # print("   Transforming text to seq...")
        if USE_MULTITHREAD:
            df["seq_name"] = parallelize_df_func(df["seq_name"], df_word_lst_to_sequences)
            df["seq_item_desc"] = parallelize_df_func(df["seq_item_desc"], df_word_lst_to_sequences)
            # df["seq_category_name"] = parallelize_df_func(df["seq_category_name"], df_word_lst_to_sequences)
            print("[%.5f] Done df_word_lst_to_sequences" % (time.time() - start_time))
            if EXTRACTED_BIGRAM:
                df["seq_bigram_item_desc"] = parallelize_df_func(df["seq_bigram_item_desc"],
                                                                 df_bigram_lst_to_sequences)
                print("[%.5f] Done df_bigram_lst_to_sequences" % (time.time() - start_time))
            if EXTRACTED_TRIGRAM:
                df["seq_trigram_item_desc"] = parallelize_df_func(df["seq_trigram_item_desc"],
                                                                  df_trigram_lst_to_sequences)
                print("[%.5f] Done df_trigram_lst_to_sequences" % (time.time() - start_time))
            if EXTRACTED_SUBWORD:
                df["seq_subword_item_desc"] = parallelize_df_func(df["seq_subword_item_desc"],
                                                                  df_subword_lst_to_sequences)
                print("[%.5f] Done df_subword_lst_to_sequences" % (time.time() - start_time))
        else:
            df["seq_name"] = df_word_lst_to_sequences(df["seq_name"])
            df["seq_item_desc"] = df_word_lst_to_sequences(df["seq_item_desc"])
            # df["seq_category_name"] = df_word_lst_to_sequences(df["seq_category_name"])
            print("[%.5f] Done df_word_lst_to_sequences" % (time.time() - start_time))
            if EXTRACTED_BIGRAM:
                df["seq_bigram_item_desc"] = df_bigram_lst_to_sequences(df["seq_bigram_item_desc"])
                print("[%.5f] Done df_bigram_lst_to_sequences" % (time.time() - start_time))
            if EXTRACTED_TRIGRAM:
                df["seq_trigram_item_desc"] = df_trigram_lst_to_sequences(df["seq_trigram_item_desc"])
                print("[%.5f] Done df_trigram_lst_to_sequences" % (time.time() - start_time))
            if EXTRACTED_SUBWORD:
                df["seq_subword_item_desc"] = df_subword_lst_to_sequences(df["seq_subword_item_desc"])
                print("[%.5f] Done df_subword_lst_to_sequences" % (time.time() - start_time))

    else:
        def word_lst_to_sequences_hash(word_lst):
            vect = [hashing_trick(w, MAX_NUM_WORDS) for w in word_lst]
            return vect

        def df_word_lst_to_sequences_hash(df):
            return df.apply(word_lst_to_sequences_hash)

        def bigram_lst_to_sequences_hash(word_lst):
            vect = [hashing_trick(w, MAX_NUM_BIGRAMS) for w in word_lst]
            return vect

        def df_bigram_lst_to_sequences_hash(df):
            return df.apply(bigram_lst_to_sequences_hash)

        def trigram_lst_to_sequences_hash(word_lst):
            vect = [hashing_trick(w, MAX_NUM_TRIGRAMS) for w in word_lst]
            return vect

        def df_trigram_lst_to_sequences_hash(df):
            return df.apply(trigram_lst_to_sequences_hash)

        def subword_lst_to_sequences_hash(word_lst):
            vect = [hashing_trick(w, MAX_NUM_SUBWORDS) for w in word_lst]
            return vect

        def df_subword_lst_to_sequences_hash(df):
            return df.apply(subword_lst_to_sequences_hash)

        # print("   Transforming text to seq...")
        if USE_MULTITHREAD:
            df["seq_name"] = parallelize_df_func(df["seq_name"], df_word_lst_to_sequences_hash)
            df["seq_item_desc"] = parallelize_df_func(df["seq_item_desc"], df_word_lst_to_sequences_hash)
            # df["seq_category_name"] = parallelize_df_func(df["seq_category_name"], df_word_lst_to_sequences_hash)
            gc.collect()
            print("[%.5f] Done df_word_lst_to_sequences_hash" % (time.time() - start_time))
            if EXTRACTED_BIGRAM:
                df["seq_bigram_item_desc"] = parallelize_df_func(df["seq_bigram_item_desc"],
                                                                 df_bigram_lst_to_sequences_hash)
                print("[%.5f] Done df_bigram_lst_to_sequences_hash" % (time.time() - start_time))
            if EXTRACTED_TRIGRAM:
                df["seq_trigram_item_desc"] = parallelize_df_func(df["seq_trigram_item_desc"],
                                                                  df_trigram_lst_to_sequences_hash)
                print("[%.5f] Done df_trigram_lst_to_sequences_hash" % (time.time() - start_time))
            if EXTRACTED_SUBWORD:
                df["seq_subword_item_desc"] = parallelize_df_func(df["seq_subword_item_desc"],
                                                                  df_subword_lst_to_sequences_hash)
                print("[%.5f] Done df_subword_lst_to_sequences_hash" % (time.time() - start_time))
        else:
            df["seq_name"] = df_word_lst_to_sequences_hash(df["seq_name"])
            df["seq_item_desc"] = df_word_lst_to_sequences_hash(df["seq_item_desc"])
            # df["seq_category_name"] = df_word_lst_to_sequences_hash(df["seq_category_name"])
            gc.collect()
            print("[%.5f] Done df_word_lst_to_sequences_hash" % (time.time() - start_time))
            if EXTRACTED_BIGRAM:
                df["seq_bigram_item_desc"] = df_bigram_lst_to_sequences_hash(df["seq_bigram_item_desc"])
                print("[%.5f] Done df_bigram_lst_to_sequences_hash" % (time.time() - start_time))
            if EXTRACTED_TRIGRAM:
                df["seq_trigram_item_desc"] = df_trigram_lst_to_sequences_hash(df["seq_trigram_item_desc"])
                print("[%.5f] Done df_trigram_lst_to_sequences_hash" % (time.time() - start_time))
            if EXTRACTED_SUBWORD:
                df["seq_subword_item_desc"] = df_subword_lst_to_sequences_hash(df["seq_subword_item_desc"])
                print("[%.5f] Done df_subword_lst_to_sequences_hash" % (time.time() - start_time))

    print("[%.5f] Done tokenize data" % (time.time() - start_time))

    if RUNNING_MODE != "submission":
        print('Average name sequence length: {}'.format(df["seq_name"].apply(len).mean()))
        print('Average item_desc sequence length: {}'.format(df["seq_item_desc"].apply(len).mean()))
        # print('Average brand_name sequence length: {}'.format(df["seq_brand_name"].apply(len).mean()))
        # print('Average category_name sequence length: {}'.format(df["seq_category_name"].apply(len).mean()))
        if EXTRACTED_SUBWORD:
            print('Average item_desc subword sequence length: {}'.format(
                df["seq_subword_item_desc"].apply(len).mean()))

    #### convert categorical variables
    if label_encoder is None:
        label_encoder = {}
        label_encoder["brand_name"] = MyLabelEncoder()
        df["brand_name_cat"] = label_encoder["brand_name"].fit_transform(df["brand_name"])
        label_encoder["category_name"] = MyLabelEncoder()
        df["category_name_cat"] = label_encoder["category_name"].fit_transform(df["category_name"])
        df.drop("brand_name", axis=1, inplace=True)
        df.drop("category_name", axis=1, inplace=True)
        gc.collect()
        for i in range(MAX_CATEGORY_NAME_LEN):
            label_encoder["category_name%d" % (i + 1)] = MyLabelEncoder()
            df["category_name%d_cat" % (i + 1)] = label_encoder["category_name%d" % (i + 1)].fit_transform(
                df["category_name%d" % (i + 1)])
            df.drop("category_name%d" % (i + 1), axis=1, inplace=True)
    else:
        df["brand_name_cat"] = label_encoder["brand_name"].transform(df["brand_name"])
        df["category_name_cat"] = label_encoder["category_name"].transform(df["category_name"])
        df.drop("brand_name", axis=1, inplace=True)
        df.drop("category_name", axis=1, inplace=True)
        gc.collect()
        for i in range(MAX_CATEGORY_NAME_LEN):
            df["category_name%d_cat" % (i + 1)] = label_encoder["category_name%d" % (i + 1)].transform(
                df["category_name%d" % (i + 1)])
            df.drop("category_name%d" % (i + 1), axis=1, inplace=True)
    print("[%.5f] Done Handling categorical variables" % (time.time() - start_time))


    if DUMP_DATA and RUNNING_MODE != "submission":
        try:
            with open(pkl_file, "wb") as f:
                pkl.dump(df, f)
        except:
            pass

    return df, word_index, bigram_index, trigram_index, subword_index, label_encoder


feat_cols = [
    "missing_brand_name", "missing_category_name", "missing_item_desc",
]
NUM_VARS_DIM = len(feat_cols)


def get_xnn_data(dataset, lbs, params):
    start_time = time.time()

    if lbs is None:
        lbs = []
        lb = LabelBinarizer(sparse_output=True)
        item_condition_array = lb.fit_transform(dataset.item_condition_id).toarray()
        lbs.append(lb)

    else:
        lb = lbs[0]
        item_condition_array = lb.transform(dataset.item_condition_id).toarray()


    num_vars = dataset[feat_cols].values

    X = {}

    X['seq_name'] = pad_sequences(dataset.seq_name, maxlen=params["max_sequence_length_name"],
                                        padding=params["pad_sequences_padding"],
                                        truncating=params["pad_sequences_truncating"])
    X["sequence_length_name"] = params["max_sequence_length_name"] * np.ones(dataset.shape[0])

    X['seq_item_desc'] = pad_sequences(dataset.seq_item_desc, maxlen=params["max_sequence_length_item_desc"],
                                             padding=params["pad_sequences_padding"],
                                             truncating=params["pad_sequences_truncating"])
    X["sequence_length_item_desc"] = params["max_sequence_length_item_desc"] * np.ones(dataset.shape[0])

    X['seq_bigram_item_desc'] = pad_sequences(dataset.seq_bigram_item_desc,
                                                    maxlen=params["max_sequence_length_item_desc"],
                                                    padding=params["pad_sequences_padding"],
                                                    truncating=params["pad_sequences_truncating"]) if params[
        "use_bigram"] else None

    X['seq_trigram_item_desc'] = pad_sequences(dataset.seq_trigram_item_desc,
                                                     maxlen=params["max_sequence_length_item_desc"],
                                                     padding=params["pad_sequences_padding"],
                                                     truncating=params["pad_sequences_truncating"]) if params[
        "use_trigram"] else None

    X['seq_subword_item_desc'] = pad_sequences(dataset.seq_subword_item_desc,
                                                     maxlen=params["max_sequence_length_item_desc_subword"],
                                                     padding=params["pad_sequences_padding"],
                                                     truncating=params["pad_sequences_truncating"]) if params[
        "use_subword"] else None
    X["sequence_length_item_desc_subword"] = params["max_sequence_length_item_desc_subword"] * np.ones(dataset.shape[0])

    X.update({
        'brand_name': dataset.brand_name_cat.values.reshape((-1, 1)),
        # 'category_name': dataset.category_name_cat.values.reshape((-1, 1)),
        'category_name1': dataset.category_name1_cat.values.reshape((-1, 1)),
        'category_name2': dataset.category_name2_cat.values.reshape((-1, 1)),
        'category_name3': dataset.category_name3_cat.values.reshape((-1, 1)),
        'item_condition_id': dataset.item_condition_id.values.reshape((-1, 1)),
        'item_condition': item_condition_array,
        'num_vars': num_vars,
        'shipping': dataset.shipping.values.reshape((-1, 1)),

    })

    print("[%.5f] Done get_xnn_data." % (time.time() - start_time))
    return X, lbs, params



########################
# MODEL TRAINING
########################
def get_training_params(train_size, batch_size, params):
    params["num_update_each_epoch"] = int(train_size / float(batch_size))

    # # cyclic lr
    params["m_mul"] = np.power(params["lr_decay_each_epoch_cosine"], 1. / params["num_cycle_each_epoch"])
    params["m_mul_exp"] = np.power(params["lr_decay_each_epoch_exp"], 1. / params["num_cycle_each_epoch"])
    if params["t_mul"] == 1:
        tmp = int(params["num_update_each_epoch"] / params["num_cycle_each_epoch"])
    else:
        tmp = int(params["num_update_each_epoch"] / params["snapshot_every_epoch"] * (1. - params["t_mul"]) / (
                1. - np.power(params["t_mul"], params["num_cycle_each_epoch"] / params["snapshot_every_epoch"])))
    params["first_decay_steps"] = max([tmp, 1])
    params["snapshot_every_num_cycle"] = params["num_cycle_each_epoch"] // params["snapshot_every_epoch"]
    params["snapshot_every_num_cycle"] = max(params["snapshot_every_num_cycle"], 1)

    # cnn
    if params["cnn_timedistributed"]:
        params["cnn_num_filters"] = params["embedding_dim"]

    # text dim after the encode step
    if params["encode_method"] == "fasttext":
        encode_text_dim = params["embedding_dim"]
    elif params["encode_method"] == "textcnn":
        encode_text_dim = params["cnn_num_filters"] * len(params["cnn_filter_sizes"])
    elif params["encode_method"] in ["textrnn", "textbirnn"]:
        encode_text_dim = params["rnn_num_units"]
    elif params["encode_method"] == "fasttext+textcnn":
        encode_text_dim = params["embedding_dim"] + params["cnn_num_filters"] * len(
            params["cnn_filter_sizes"])
    elif params["encode_method"] in ["fasttext+textrnn", "fasttext+textbirnn"]:
        encode_text_dim = params["embedding_dim"] + params["rnn_num_units"]
    elif params["encode_method"] in ["fasttext+textcnn+textrnn", "fasttext+textcnn+textbirnn"]:
        encode_text_dim = params["embedding_dim"] + params["cnn_num_filters"] * len(
            params["cnn_filter_sizes"]) + params["rnn_num_units"]
    params["encode_text_dim"] = encode_text_dim

    return params


def cross_validation_hyperopt(dfTrain, params, target_scaler):
    params = ModelParamSpace()._convert_int_param(params)
    _print_param_dict(params)

    # level1, valid index
    level1Ratio, validRatio = 0.6, 0.4
    num_train = dfTrain.shape[0]
    level1Size = int(level1Ratio * num_train)
    indices = np.arange(num_train)
    np.random.seed(params["random_seed"])
    np.random.shuffle(indices)
    level1Ind, validInd = indices[:level1Size], indices[level1Size:]
    y_level1, y_valid = dfTrain.price.values[level1Ind].reshape((-1, 1)), dfTrain.price.values[validInd].reshape(
        (-1, 1))
    y_valid_inv = target_scaler.inverse_transform(y_valid)

    X_level1, lbs, params = get_xnn_data(dfTrain.iloc[level1Ind], lbs=None, params=params)
    X_valid, lbs, _ = get_xnn_data(dfTrain.iloc[validInd], lbs=lbs, params=params)

    params = get_training_params(train_size=len(level1Ind), batch_size=params["batch_size_train"],
                                 params=params)
    model = XNN(params, target_scaler, logger)
    model.fit(X_level1, y_level1, validation_data=(X_valid, y_valid))
    y_valid_tf = model.predict(X_valid, mode="raw")
    y_valid_tf_inv = target_scaler.inverse_transform(y_valid_tf)
    for j in reversed(range(y_valid_tf.shape[1])):
        rmsle = rmse(y_valid_inv, y_valid_tf_inv[:, j, np.newaxis])
        logger.info("valid-rmsle (tf of last %d): %.5f" % (y_valid_tf.shape[1] - j, rmsle))
        y_valid_tf_inv_ = np.mean(y_valid_tf_inv[:, j:], axis=1, keepdims=True)
        rmsle = rmse(y_valid_inv, y_valid_tf_inv_)
        logger.info(
            "valid-rmsle (tf snapshot ensemble with mean of last %d): %.5f" % (y_valid_tf.shape[1] - j, rmsle))

    stacking_model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=4)
    stacking_model.fit(y_valid_tf, y_valid)
    logger.info(stacking_model.intercept_)
    logger.info(stacking_model.coef_)
    y_valid_stack = stacking_model.predict(y_valid_tf).reshape((-1, 1))
    y_valid_stack_inv = target_scaler.inverse_transform(y_valid_stack)
    rmsle = rmse(y_valid_inv, y_valid_stack_inv)
    logger.info("rmsle (stack): %.5f" % rmsle)

    rmsle_mean = rmsle
    rmsle_std = 0
    logger.info("RMSLE")
    logger.info("      Mean: %.6f" % rmsle_mean)
    logger.info("      Std: %.6f" % rmsle_std)
    ret = {
        "loss": rmsle_mean,
        "attachments": {
            "std": rmsle_std,
        },
        "status": STATUS_OK,
    }
    return ret


def submission(params):
    params = ModelParamSpace()._convert_int_param(params)
    _print_param_dict(params)
    start_time = time.time()

    dfTrain = load_train_data()
    target_scaler = MyStandardScaler()
    dfTrain["price"] = target_scaler.fit_transform(dfTrain["price"].values.reshape(-1, 1))
    dfTrain, word_index, bigram_index, trigram_index, subword_index, label_encoder = preprocess(
        dfTrain)

    X_train, lbs_tf, params = get_xnn_data(dfTrain, lbs=None, params=params)
    y_train = dfTrain.price.values.reshape((-1, 1))

    params["MAX_NUM_BRANDS"] = dfTrain["brand_name_cat"].max() + 1
    params["MAX_NUM_CATEGORIES"] = dfTrain["category_name_cat"].max() + 1
    params["MAX_NUM_CATEGORIES_LST"] = [0] * MAX_CATEGORY_NAME_LEN
    for i in range(MAX_CATEGORY_NAME_LEN):
        params["MAX_NUM_CATEGORIES_LST"][i] = dfTrain["category_name%d_cat" % (i + 1)].max() + 1
    params["MAX_NUM_CONDITIONS"] = dfTrain["item_condition_id"].max()
    params["MAX_NUM_SHIPPINGS"] = 2
    params["NUM_VARS_DIM"] = NUM_VARS_DIM

    del dfTrain
    gc.collect()
    print('[%.5f] Finished loading data' % (time.time() - start_time))

    params = get_training_params(train_size=len(y_train), batch_size=params["batch_size_train"], params=params)
    model = XNN(params, target_scaler, logger)
    model.fit(X_train, y_train)
    del X_train
    del y_train
    gc.collect()
    print('[%.5f] Finished training tf' % (time.time() - start_time))

    y_test = []
    id_test = []
    for dfTest in load_test_data(chunksize=350000*2):
        dfTest, _, _, _, _, _ = preprocess(dfTest, word_index, bigram_index, trigram_index, subword_index, label_encoder)
        X_test, lbs_tf, _ = get_xnn_data(dfTest, lbs=lbs_tf, params=params)

        y_test_ = model.predict(X_test, mode="weight")
        y_test.append(y_test_)
        id_test.append(dfTest.id.values.reshape((-1, 1)))

    y_test = np.vstack(y_test)
    id_test = np.vstack(id_test)
    y_test = np.expm1(target_scaler.inverse_transform(y_test))
    y_test = y_test.flatten()
    id_test = id_test.flatten()
    id_test = id_test.astype(int)
    y_test[y_test < 0.0] = 0.0
    submission = pd.DataFrame({"test_id": id_test, "price": y_test})
    submission.to_csv("sample_submission.csv", index=False)
    print('[%.5f] Finished prediction' % (time.time() - start_time))


# -------------------------------------- fasttext ---------------------------------------------
def _print_param_dict(d, prefix="      ", incr_prefix="      "):
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            logger.info("%s%s:" % (prefix, k))
            _print_param_dict(v, prefix + incr_prefix, incr_prefix)
        else:
            logger.info("%s%s: %s" % (prefix, k, v))


class ModelParamSpace:
    def __init__(self):
        pass

    def _convert_int_param(self, param_dict):
        if isinstance(param_dict, dict):
            for k, v in param_dict.items():
                if k in int_params:
                    param_dict[k] = v if v is None else int(v)
                elif isinstance(v, list) or isinstance(v, tuple):
                    for i in range(len(v)):
                        self._convert_int_param(v[i])
                elif isinstance(v, dict):
                    self._convert_int_param(v)
        return param_dict


if RUNNING_MODE == "validation":
    load_data_success = False
    pkl_file = "../input/dfTrain_bigram_[MAX_NUM_WORDS_%d]_[MAX_NUM_BIGRAMS_%d]_[VOCAB_HASHING_TRICK_%s].pkl" % (
        MAX_NUM_WORDS, MAX_NUM_BIGRAMS, str(VOCAB_HASHING_TRICK))
    if USE_PREPROCESSED_DATA:
        try:
            with open(pkl_file, "rb") as f:
                dfTrain = pkl.load(f)
            if DEBUG:
                dfTrain = dfTrain.head(DEBUG_SAMPLE_NUM)
            load_data_success = True
        except:
            pass
    if not load_data_success:
        dfTrain = load_train_data()
        dfTrain, word_index, bigram_index, trigram_index, subword_index, label_encoder = preprocess(dfTrain)
    target_scaler = MyStandardScaler()
    dfTrain["price"] = target_scaler.fit_transform(dfTrain["price"].values.reshape(-1, 1))

    param_space_hyperopt["MAX_NUM_BRANDS"] = dfTrain["brand_name_cat"].max() + 1
    param_space_hyperopt["MAX_NUM_CATEGORIES"] = dfTrain["category_name_cat"].max() + 1
    param_space_hyperopt["MAX_NUM_CATEGORIES_LST"] = [0] * MAX_CATEGORY_NAME_LEN
    for i in range(MAX_CATEGORY_NAME_LEN):
        param_space_hyperopt["MAX_NUM_CATEGORIES_LST"][i] = dfTrain["category_name%d_cat" % (i + 1)].max() + 1
    param_space_hyperopt["MAX_NUM_CONDITIONS"] = dfTrain["item_condition_id"].max()
    param_space_hyperopt["MAX_NUM_SHIPPINGS"] = 2
    param_space_hyperopt["NUM_VARS_DIM"] = NUM_VARS_DIM

    start_time = time.time()
    trials = Trials()
    obj = lambda param: cross_validation_hyperopt(dfTrain, param, target_scaler)
    best = fmin(obj, param_space_hyperopt, tpe.suggest, HYPEROPT_MAX_EVALS, trials)
    best_params = space_eval(param_space_hyperopt, best)
    best_params = ModelParamSpace()._convert_int_param(best_params)
    trial_rmsles = np.asarray(trials.losses(), dtype=float)
    best_ind = np.argmin(trial_rmsles)
    best_rmse_mean = trial_rmsles[best_ind]
    best_rmse_std = trials.trial_attachments(trials.trials[best_ind])["std"]
    logger.info("-" * 50)
    logger.info("Best RMSLE")
    logger.info("      Mean: %.6f" % best_rmse_mean)
    logger.info("      Std: %.6f" % best_rmse_std)
    logger.info("Best param")
    _print_param_dict(best_params)
    logger.info("time: %.5f" % (time.time() - start_time))

else:
    submission(param_space_best)