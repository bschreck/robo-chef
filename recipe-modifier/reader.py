"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division

import collections
import os
import sys
import time
import copy

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile
import string
import generate_refinements as gen
from itertools import chain
FLAGS = tf.app.flags.FLAGS

#TODO: TRY DIFFERENT WAYS TO SPLIT SENTENCES (maybe don't replace all punctuation?)
#IDEALLY PHRASES WILL ALREADY BY SPLIT AT THIS POINT thought
#see if we need GO symbol?"
#also should we make a symbol for digits?

_EOP = '<eop>'
_EOR = '<eor>'

PAD_ID = 0
UNK_ID = 1

def _read_words(filename):
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    #separate recipes with <eor>
    #separate phrases with <eop>
    with gfile.GFile(filename, "r") as f:
        return f.read().translate(replace_punctuation).replace("\t"," <eop> ").replace("\n", " <eor> ").split()

def _read_lines(filename):
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    #separate recipes with <eor>
    #separate phrases with <eop>
    with gfile.GFile(filename, "r") as f:
        for line in f:
            no_punc = line.translate(replace_punctuation)
            by_tab = no_punc.split('\t')
            label = by_tab[0]
            refinement = by_tab[1]
            recipe = by_tab[2:]
            yield label, refinement, recipe

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
  words = list(words)
  words.insert(0,'_UNK')
  words.insert(0,'_PAD')
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id

def _get_bucket(num_words, num_phrases, buckets):
    #num_words and num_phrases must be less than or equal to the max in the bucket +1,
    #we're accounting for the <eop> at the end of each phrase
    #and the <eor> as its own phrase at the end of the recipe
    for bucket in buckets:
        if num_phrases <= bucket[0]+1 and num_words <= bucket[1]+1:
            return bucket
    raise ValueError, "num_words %s or num_phrases %s too long"%(num_words, num_phrases)

def _file_to_word_ids(filename, word_to_id, unknown_word_id,buckets):
    #buckets = [(max number of phrases, max number of words per phrase)]
    #last bucket needs to be max of both values
    recipes = {}
    refinements = {}
    labels = {}
    for bucket in buckets:
        recipes[bucket] = []
        labels[bucket] = []
        refinements[bucket] = []
    for line in _read_lines(filename):
        label, refinement, recipe = line
        label = int(label)
        if label < 0:
            #insertion
            label = 2*(-label-1) + 1
        else:
            #replacement
            label = 2*(label-1)
        refinement = [word_to_id[w] if w in word_to_id else unknown_word_id for w in refinement.split()]
        refinement.append(word_to_id['<eop>'])
        max_current_num_words = len(refinement)

        recipe = [phrase.split() for phrase in recipe]
        for i,phrase in enumerate(recipe[:-1]):
            recipe[i] = [word_to_id[w] if w in word_to_id else unknown_word_id for w in phrase]
            recipe[i].append(word_to_id['<eop>'])
            if len(recipe[i]) > max_current_num_words:
                max_current_num_words = len(recipe[i])
        recipe[-1].append(word_to_id['<eor>'])
        current_num_phrases = len(recipe)
        bucket = _get_bucket(max_current_num_words, current_num_phrases, buckets)

        nprecipe = np.zeros((bucket[0]+1,bucket[1]+1), dtype=np.int32)
        nprecipe.fill(PAD_ID)
        nprefinement = np.zeros(bucket[1]+1, dtype=np.int32)
        nprefinement.fill(PAD_ID)
        nprefinement[:len(refinement)] = refinement

        for i,phrase in enumerate(recipe):
            nprecipe[i,:len(phrase)] = np.array(phrase)

        recipes[bucket].append(nprecipe)
        labels[bucket].append(label)
        refinements[bucket].append(nprefinement)
    for bucket in buckets:
        labels[bucket] = np.array(labels[bucket], dtype=np.int32)
        refinements[bucket] = np.array(refinements[bucket], dtype=np.int32)
        recipes[bucket] = np.array(recipes[bucket], dtype=np.int32)
    return labels, refinements, recipes

def recipe_raw_data(word_to_id, initial_buckets, corpus='train'):
    if corpus == 'train':
        filename = os.path.join(FLAGS.data_dir, FLAGS.train_corpus)
    elif corpus == 'val':
        filename = os.path.join(FLAGS.data_dir, FLAGS.val_corpus)
    else:
        filename = os.path.join(FLAGS.data_dir, FLAGS.test_corpus)
    max_phrases_path = os.path.join(FLAGS.data_dir, FLAGS.max_phrases_file)

    max_phrase_num, max_phrase_len = _read_max_phrase_num_and_len(max_phrases_path)
    buckets=copy.copy(initial_buckets)
    buckets.append((max_phrase_num, max_phrase_len))

    labels, refinements, recipes = _file_to_word_ids(filename, word_to_id, UNK_ID, buckets)
    vocabulary = len(word_to_id)
    return labels, refinements, recipes, vocabulary, max_phrase_num, max_phrase_len,buckets

def _draw_buckets(fraction_each_bucket, batch_size, buckets):
    bucket_sel = np.random.random()
    for f,b in fraction_each_bucket:
        if bucket_sel < f:
            return buckets[b]
    return buckets[b]
def recipe_iterator(labels, refinements, recipes, batch_size, all_buckets=False):
    for bucket in labels:
        shuffled_indices = range(len(labels[bucket]))
        np.random.shuffle(shuffled_indices)
        labels[bucket] = labels[bucket][shuffled_indices]
        refinements[bucket] = refinements[bucket][shuffled_indices]
        recipes[bucket] = recipes[bucket][shuffled_indices]
    buckets = sorted(labels.keys(), key=lambda x:x[0])

    total_each_bucket = np.array([len(labels[b]) for b in buckets])
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
        labels_bucketed_batches = {}
        refinements_bucketed_batches = {}
        recipes_bucketed_batches = {}
        for batch in xrange(num_batches):
            if all_buckets:
                bucket_id = batch
                bucket = buckets[bucket_id]
            else:
                bucket = _draw_buckets(fraction_each_bucket, batch_size, buckets)
            new_bucket_i = bucket_i[bucket] + batch_size
            current_recipes = recipes[bucket][bucket_i[bucket]:new_bucket_i, :,:]
            current_labels = labels[bucket][bucket_i[bucket]:new_bucket_i]
            current_refinements = refinements[bucket][bucket_i[bucket]:new_bucket_i,:]
            bucket_i[bucket] = new_bucket_i
            if all_buckets:
                labels_bucketed_batches[bucket_id] = current_labels
                refinements_bucketed_batches[bucket_id] = current_refinements
                recipes_bucketed_batches[bucket_id] = current_recipes
            else:
                yield bucket, current_labels, current_refinements, current_recipes
        if all_buckets:
            yield labels_bucketed_batches, refinements_bucketed_batches, recipes_bucketed_batches

def _add_blanks_to_recipe(recipe):
    #recipe in shape (num_phrases, num_words_per_phrase, batch_size)
    #add a blank to each line except the last
    expanded_recipe = np.zeros((2*recipe.shape[0] - 1, recipe.shape[1], recipe.shape[2]), dtype=np.int32)
    expanded_recipe.fill(PAD_ID)
    indices = range(recipe.shape[0]-1)
    indices = [2*i for i in indices]
    indices.append(indices[-1]+1)
    expanded_recipe[indices,:,:] = recipe
    return expanded_recipe

def _index_to_ohe(indices, desired_size):
    #indices is shape (batch_size,)
    #output is shape (desired_size, batch_size)
    output = np.zeros((desired_size, indices.shape[0]))
    for batch,i in enumerate(indices):
        output[i, batch] = 1
    return output

def batch_iterator(word_to_id, batch_size, corpus, initial_buckets, all_buckets=False):
    #expand recipes to include blanks between each line

    #loops continuously over corpus
    labels, refinements, recipes,vocab,max_phrase_num,max_phrase_len, buckets= recipe_raw_data(word_to_id, initial_buckets, corpus=corpus)
    start = True
    while True:
        if start:
            yield buckets
            start = False
        processed = recipe_iterator(labels, refinements, recipes, batch_size, all_buckets=all_buckets)
        for i,batch in enumerate(processed):
            if all_buckets:
                labels, refinements, recipes = batch
                for bucket in refinements:
                    refinements[bucket] = np.rollaxis(refinements[bucket], 1,0)
                    recipes[bucket] = _add_blanks_to_recipe(np.rollaxis(np.rollaxis(recipes[bucket],1,0),2,1))
                    labels[bucket] = _index_to_ohe(labels[bucket], recipes[bucket].shape[0])
                yield labels, refinements, recipes
            else:
                bucket, labels, refinements, recipes= batch
                bucket_id = buckets.index(bucket)
                #change dimensions so batch is last
                refinements = np.rollaxis(refinements, 1,0)
                recipes = np.rollaxis(np.rollaxis(recipes,1,0),2,1)
                recipes_with_blanks = _add_blanks_to_recipe(recipes)
                labels  = _index_to_ohe(labels, recipes_with_blanks.shape[0])
                yield bucket_id, labels, refinements, recipes_with_blanks

# word_to_id = build_vocab(TRAIN_CORPUS)
# init_buckets = [(10,15), (15,20), (20,25), (25,30), (30,35)]

# labels, refinements, recipes, vocabulary, max_phrase_num, max_phrase_len,buckets = recipe_raw_data(word_to_id, buckets, data_path='.', corpus='train')
# print recipes[(15,20)][0:5]
# print len(recipes[(15,20)])

# b = batch_iterator(word_to_id, 10,'train', init_buckets, all_buckets=False)
# b.next()
# first_batch = b.next()
# print 'label shape:', first_batch[1].shape
# print 'refinement shape:', first_batch[2].shape
# print 'recipe shape:', first_batch[3].shape


# b = batch_iterator(word_to_id, 10,'train', init_buckets, all_buckets=True)
# init_buckets.append((39,68))
# b.next()
# first_batch = b.next()
# for bucket_id,bucket in enumerate(init_buckets):
    # print bucket_id, bucket
    # print 'label shape:', first_batch[0][bucket_id].shape
    # print 'refinement shape:', first_batch[1][bucket_id].shape
    # print 'recipe shape:', first_batch[2][bucket_id].shape

# print batch_iterator(word_to_id, 10,'train').next()[0]
#print batch_iterator(word_to_id, 10,'train', all_buckets=True).next().values()[4][0]
