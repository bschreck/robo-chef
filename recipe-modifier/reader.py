"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import time

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile
import string


def _read_words(filename):
  replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
  #separate recipes with <eor>
  #separate phrases with <eop>
  with gfile.GFile(filename, "r") as f:
    return f.read().translate(replace_punctuation).replace("\t"," <eop> ").replace("\n", " <eor> ").split()

def _read_max_phrase_num_and_len(filename):
  with gfile.GFile(filename, "r") as f:
    max_phrase_num = int(f.readline().split()[1])
    max_phrase_len = int(f.readline().split()[1])
  return max_phrase_num, max_phrase_len




def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: -x[1])

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id, unknown_word_id,buckets):
  #buckets = [(max number of phrases, max number of words per phrase)]
  #last bucket needs to be max of both values
  data = _read_words(filename)
  recipes = {}
  for bucket in buckets:
      recipes[bucket] = []
  phrase_i = 0
  word_i = 0
  # current_recipe = np.zeros((max_phrase_num+1,max_phrase_len+1), dtype=np.int32)
  # current_recipe.fill(-1)
  current_recipe = [[]]
  max_current_num_words = 0

  for word in data:
    #each recipe can be at maximum max_phrase_num+1 phrases long,
    #with the 1 added for the <eor> word
    #each phrase can at maximum max_phrase_len+1 words long,
    #with the 1 added for the <eop> word
    if word in word_to_id:
        current_recipe[-1].append(word_to_id[word])
    else:
        current_recipe[-1].append(unknown_word_id)
    if word == "<eop>":
      if word_i > max_current_num_words:
        max_current_num_words = word_i
      #make a new phrase
      current_recipe.append([])
      word_i = 0
      phrase_i += 1
    elif word == "<eor>":
      for bucket in buckets:
        if phrase_i <= bucket[0] and max_current_num_words <= bucket[1]:
          current_bucket = bucket
          break
      for phrase in current_recipe:
        if len(phrase) < current_bucket[1]:
          phrase.extend([-1]*(current_bucket[1]-len(phrase)))
      if len(current_recipe) < current_bucket[0]:
        current_recipe.extend([[-1]*current_bucket[1] for _ in xrange(current_bucket[0]-len(current_recipe))])
      recipes[current_bucket].append(current_recipe)
      phrase_i = 0
      word_i = 0
      max_current_num_words = 0
      current_recipe = [[]]
    else:
      word_i += 1
  return recipes

def ptb_raw_data(data_path=None):

  train_path = os.path.join(data_path, "recipes_train.txt")
  valid_path = os.path.join(data_path, "recipes_valid.txt")
  test_path = os.path.join(data_path, "recipes_test.txt")
  max_phrases_path = os.path.join(data_path, "max_phrases.txt")

  max_phrase_num, max_phrase_len = _read_max_phrase_num_and_len(max_phrases_path)
  buckets=[(10,15), (15,20), (20,25), (25,30), (30,35)]
  buckets.append((max_phrase_num, max_phrase_len))

  word_to_id = _build_vocab(train_path)
  unknown_word_id = len(word_to_id)
  train_data = _file_to_word_ids(train_path, word_to_id, unknown_word_id, buckets)
  valid_data = _file_to_word_ids(valid_path, word_to_id,unknown_word_id,  buckets)
  test_data = _file_to_word_ids(test_path, word_to_id, unknown_word_id, buckets)
  vocabulary = unknown_word_id + 1
  return train_data, valid_data, test_data, vocabulary, max_phrase_num, max_phrase_len,buckets

def _draw_buckets(fraction_each_bucket, batch_size):
    bucket_sel = np.random.random(size=batch_size)
    num_each_bucket = collections.defaultdict(int)
    for bucket_draw in bucket_sel:
        for f,b in fraction_each_bucket:
            if bucket_draw < f:
                num_each_bucket[buckets[b]] += 1
                break
    return num_each_bucket
def ptb_iterator(raw_data, batch_size):
    buckets = sorted(raw_data.keys(), key=lambda x:x[0])
    total_each_bucket = np.array([len(raw_data[b]) for b in buckets])
    data_len = sum(total_each_bucket)
    fraction_each_bucket = total_each_bucket / data_len
    fraction_each_bucket = [[f,i] for i,f in enumerate(fraction_each_bucket)]
    fraction_each_bucket = sorted(fraction_each_bucket, key=lambda x:x[0])
    for i,f in enumerate(fraction_each_bucket[1:]):
        fraction_each_bucket[i+1][0] = f[0] + fraction_each_bucket[i][0]
    batch_len = data_len // batch_size
    bucket_i = collections.defaultdict(int)
    for i in range(batch_len):
        num_each_bucket = _draw_buckets(fraction_each_bucket, batch_size)
        current_data = {}
        for bucket in num_each_bucket:
            new_bucket_i = bucket_i[bucket] + num_each_bucket[bucket]
            current_data[bucket] = raw_data[bucket][bucket_i[bucket]:new_bucket_i]
            bucket_i[bucket] = new_bucket_i
        yield current_data

if __name__ == '__main__':
    train, valid,test,vocab,max_phrase_num,max_phrase_len,buckets = ptb_raw_data(data_path = '.')
    for i,batch in enumerate(ptb_iterator(train, 10)):
        pass
