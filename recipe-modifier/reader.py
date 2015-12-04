"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division

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
import generate_refinements as gen
from itertools import chain


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




def build_vocab(filename):
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

def recipe_raw_data(word_to_id, initial_buckets, data_path=None, corpus='train'):
    if corpus == 'train':
        filename = "recipes_train.txt"
    elif corpus == 'val':
        filename = "recipes_valid.txt"
    else:
        filename = "recipes_test.txt"
    path = os.path.join(data_path, filename)
    max_phrases_path = os.path.join(data_path, "max_phrases.txt")

    max_phrase_num, max_phrase_len = _read_max_phrase_num_and_len(max_phrases_path)
    buckets=initial_buckets
    buckets.append((max_phrase_num, max_phrase_len))

    unknown_word_id = len(word_to_id)
    data = _file_to_word_ids(path, word_to_id, unknown_word_id, buckets)
    vocabulary = unknown_word_id + 1
    return data, vocabulary, max_phrase_num, max_phrase_len,buckets

def _draw_buckets(fraction_each_bucket, batch_size, buckets):
    bucket_sel = np.random.random()
    for f,b in fraction_each_bucket:
        if bucket_sel < f:
            return buckets[b]
    return buckets[b]
def recipe_iterator(raw_data, batch_size, all_buckets=False):
    for bucket in raw_data:
        np.random.shuffle(raw_data[bucket])
    buckets = sorted(raw_data.keys(), key=lambda x:x[0])

    total_each_bucket = np.array([len(raw_data[b]) for b in buckets])
    data_len = sum(total_each_bucket)
    if not all_buckets:
        fraction_each_bucket = total_each_bucket / data_len
        fraction_each_bucket = [[f,i] for i,f in enumerate(fraction_each_bucket)]
        fraction_each_bucket = sorted(fraction_each_bucket, key=lambda x:x[0])
        for i,f in enumerate(fraction_each_bucket[1:]):
            fraction_each_bucket[i+1][0] = f[0] + fraction_each_bucket[i][0]

    batch_len = data_len // batch_size
    bucket_i = collections.defaultdict(int)

    num_batches = len(buckets) if all_buckets else 1
    for i in xrange(batch_len):
        bucketed_batches = {}
        for batch in xrange(num_batches):
            if all_buckets:
                bucket = buckets[batch]
            else:
                bucket = _draw_buckets(fraction_each_bucket, batch_size, buckets)
            new_bucket_i = bucket_i[bucket] + batch_size
            current_data = raw_data[bucket][bucket_i[bucket]:new_bucket_i]
            bucket_i[bucket] = new_bucket_i
            if all_buckets:
                bucketed_batches[bucket] = current_data
            else:
                yield bucket, current_data
        if all_buckets:
            yield bucketed_batches

def generate_labeled_batch(batch_recipe_list):
    labels, refinements, recipes = zip(*[gen.removals(recipe) for recipe in batch_recipe_list])
    labels = np.array(list(chain.from_iterable(i for i in labels)))
    refinements = np.array(list(chain.from_iterable(i for i in refinements)))
    recipes = list(chain.from_iterable(i for i in recipes))
    l = len(recipes[0])
    for recipe in recipes:
        if len(recipe) != l:
            print 'wtf'
    sys.exit()
    expanded_recipes = np.empty((len(recipes), 2*len(recipes[0]), len(recipes[0][0])))
    print expanded_recipes.shape
    for i,recipe in enumerate(recipes):
        blank_line = [-1]*len(recipe[0])
        print len(recipe)
        expanped_recipe =  np.array([recipe[j//2] if j%2==0 else blank_line for j in xrange(2*len(recipe))])
        print expanped_recipe.shape
        expanded_recipes[i,:,:] = np.array([recipe[j//2] if j%2==0 else blank_line for j in xrange(2*len(recipe))])
    #print recipes
    print expanded_recipes.shape
    sys.exit()
    num_in_bucket = len(labels)
    shuffle = range(num_in_bucket)
    np.random.shuffle(shuffle)
    labels = labels[shuffle]
    refinements = refinements[shuffle]
    recipes = recipes[shuffle]
    return labels, refinements, recipes

def batch_iterator(word_to_id, pre_generation_batch_size, corpus, initial_buckets, all_buckets=False):
    #loops continuously over corpus
    data,vocab,max_phrase_num,max_phrase_len, buckets= recipe_raw_data(word_to_id, initial_buckets, data_path = '.', corpus=corpus)
    pre_generation_batch_size = 10
    start = True
    while True:
        if start:
            yield buckets
            start = False
        recipes = recipe_iterator(data, pre_generation_batch_size, all_buckets=all_buckets)
        for i,batch in enumerate(recipes):
            if all_buckets:
                data = batch
                labeled_data = {}
                for bucket_id,bucket in enumerate(buckets):
                    labels, refinements, recipes = generate_labeled_batch(data[bucket])
                    labeled_data[bucket_id] = (labels, refinements, recipes)
                yield labeled_data
            else:
                bucket, data = batch
                bucket_id = buckets.index(bucket)
                labels, refinements, recipes = generate_labeled_batch(data)
                yield bucket_id, labels, refinements, recipes

word_to_id = build_vocab('./recipes_train.txt')
buckets = [(10,15), (15,20), (20,25), (25,30), (30,35)]
b = batch_iterator(word_to_id, 10,'train', buckets)
b.next()
print b.next()
# print batch_iterator(word_to_id, 10,'train').next()[0]
#print batch_iterator(word_to_id, 10,'train', all_buckets=True).next().values()[4][0]
