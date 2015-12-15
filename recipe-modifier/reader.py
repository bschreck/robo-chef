"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division

import collections
import os
import sys
import time
import copy
import cPickle as pickle
import util as util

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile
import string
import generate_refinements as gen
from itertools import chain
FLAGS = tf.app.flags.FLAGS

#see if we need GO symbol?"
#also should we make a symbol for digits?


PAD_ID = 0
UNK_ID = 1
_EOP_ID = 2
_EOR_ID = 3

_EOP = '<eor>'
_EOR = '<eop>'

_IRS = '<INTER_SEGMENT_SPACE>'

# def _read_words(filename):
    # #separate recipes with <eor>
    # #separate phrases with <eop>
    # with gfile.GFile(filename, "r") as f:
        # return util.phrase2words(f.read().replace("\t"," <eop> ").replace("\n", " <eor> "))

def _read_lines(filename):
    #separate recipes with <eor>
    #separate phrases with <eop>
    with gfile.GFile(filename, "r") as f:
        for i,line in enumerate(f):
            by_tab = line.split('\t')
            if len(by_tab) > 2:
                label = by_tab[0].strip()
                refinement = by_tab[1].strip().split()
                recipe = [segment.strip().split() for segment in by_tab[2:]]
                yield label, refinement, recipe

def _read_max_phrase_num_and_len():
    max_phrases_path = os.path.join(FLAGS.data_dir, FLAGS.max_phrases_file)
    with gfile.GFile(max_phrases_path, "r") as f:
        max_phrase_num = int(f.readline().split()[1])
        max_phrase_len = int(f.readline().split()[1])
    return max_phrase_num, max_phrase_len




def build_vocab():
    #similar to func in pck_to_txt but builds a word 2 int mapping
    vocab_file = FLAGS.vocab_dict_file#'vocab_dict.p'
    vocab_set_file = FLAGS.vocab_file#'vocab.p'
    if os.path.isfile(vocab_file):
        return pickle.load(open(vocab_file, 'rb'))
    else:
        data = pickle.load(open(vocab_set_file, 'rb'))
        words = list(data)
        # counter = collections.Counter(data)
        # count_pairs = sorted(counter.items(), key=lambda x: -x[1])

        # words, _ = list(zip(*count_pairs))
        # words = list(words)
        words.insert(0,_IRS)
        words.insert(0,'<eop>')
        words.insert(0,'<eor>')
        words.insert(0,'_UNK')
        words.insert(0,'_PAD')
        word_to_id = dict(zip(words, range(len(words))))
        pickle.dump(word_to_id, open(vocab_file,'wb'))
    return word_to_id

def _get_bucket(num_words, num_phrases, buckets):
    #num_words and num_phrases must be less than or equal to the max in the bucket
    #we're accounting for the <eop> at the end of each phrase
    #and the <eor> as its own phrase at the end of the recipe
    for bucket in buckets:
        if num_phrases <= bucket[0] and num_words <= bucket[1]:
            return bucket
    raise ValueError, "num_words %s or num_phrases %s too long"%(num_words, num_phrases)
# def _add_blanks_to_recipe(recipe):
    # #recipe in shape (num_phrases, num_words_per_phrase, batch_size)
    # #add a blank to each line except the last
    # expanded_recipe = np.zeros((2*recipe.shape[0] - 1, recipe.shape[1], recipe.shape[2]), dtype=np.int32)
    # expanded_recipe.fill(PAD_ID)
    # indices = range(recipe.shape[0]-1)
    # indices = [2*i for i in indices]
    # indices.append(indices[-1]+1)
    # expanded_recipe[indices,:,:] = recipe
    # return expanded_recipe

def refinement_str_list_to_rnn_format(refinement, word_to_id, unknown_word_id,bucket):
    refinement = [word_to_id[w] if w in word_to_id else unknown_word_id for w in refinement]
    refinement.append(word_to_id['<eop>'])
    nprefinement = np.zeros(bucket[1], dtype=np.int32)
    nprefinement.fill(PAD_ID)
    nprefinement[:len(refinement)] = refinement
    return nprefinement

def recipe_str_list_to_rnn_format(recipe, word_to_id, unknown_word_id, buckets, max_current_num_words):
    for i,phrase in enumerate(recipe[:-1]):
        recipe[i] = [word_to_id[w] if w in word_to_id else unknown_word_id for w in phrase]
        recipe[i].append(word_to_id['<eop>'])
        if len(recipe[i]) > max_current_num_words:
            max_current_num_words = len(recipe[i])
    #this actually doesn't do anything
    #recipe[-1].append(word_to_id['<eor>'])

    current_num_phrases = len(recipe)
    #get bucket accounting for blanks every other phrase (for insertion)
    bucket = _get_bucket(max_current_num_words, 2*current_num_phrases, buckets)

    nprecipe = np.zeros((bucket[0],bucket[1]), dtype=np.int32)
    nprecipe.fill(PAD_ID)
    #fill nprecipe with ndarray versions of each phrase,
    #every other line, not including the very last phrase
    #which is just an <eor> symbol
    for i,phrase in enumerate(recipe[:-1]):
        nprecipe[2*i+1,:len(phrase)] = np.array(phrase)
    return nprecipe,bucket
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
            #insertion comes before each line
            label = 2*(-label-1)
        else:
            #replacement comes at every line
            label = 2*(label-1) + 1
        max_current_num_words = len(refinement)+1# plus 1 for <eop> at end
        nprecipe,bucket = recipe_str_list_to_rnn_format(recipe,
                                            word_to_id, unknown_word_id,buckets, max_current_num_words)

        nprefinement = refinement_str_list_to_rnn_format(refinement,
                                            word_to_id, unknown_word_id,bucket)

        recipes[bucket].append(nprecipe)
        labels[bucket].append(label)
        refinements[bucket].append(nprefinement)
    for bucket in buckets:
        labels[bucket] = np.array(labels[bucket], dtype=np.int32)
        refinements[bucket] = np.array(refinements[bucket], dtype=np.int32)
        recipes[bucket] = np.array(recipes[bucket], dtype=np.int32)
    return labels, refinements, recipes

def getDataFiles(corpus, directory = None):
    if not directory:
        directory = FLAGS.data_dir
    files = [os.path.join(directory,f) for f in os.listdir(directory) if (
                                    os.path.isfile(os.path.join(directory,f)) and
                                    os.path.join(directory,f).endswith('.txt') and
                                    os.path.join(directory,f).find(corpus) > -1)]
    return files

def recipe_raw_data(word_to_id, buckets, filename):
    labels, refinements, recipes = _file_to_word_ids(filename, word_to_id, UNK_ID, buckets)
    return labels, refinements, recipes

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
        print 'fraction_each_bucket:', fraction_each_bucket
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



def _index_to_ohe(indices, desired_size):
    #indices is shape (batch_size,)
    #output is shape (desired_size, batch_size)
    output = np.zeros((desired_size, indices.shape[0]))
    for batch,i in enumerate(indices):
        output[i, batch] = 1
    return output

def _read_file_no_buckets(filename):
    for label_str, refinement, recipe in _read_lines(filename):
        label = int(label_str)
        if label > 0:
            refinement_indx = (abs(label) - 1)*2 + 1
        else:
            refinement_indx = (abs(label) - 1)*2
        padded_recipe = [[_IRS]]
        for seg in recipe:
            padded_recipe.append(seg)
            padded_recipe.append([_IRS])
        assert refinement_indx < len(padded_recipe), 'index conversion is screwed up'
        yield padded_recipe, refinement, refinement_indx

def bucketless_recipe_iterator(word_to_id, corpus):
    unknown_word_id = word_to_id['_UNK']

    #loops continuously over corpus
    for filename in getDataFiles(corpus):
        for recipe, refinement, refinement_indx in _read_file_no_buckets(filename):
            recipe_word_ids = []
            for seg in recipe:
                recipe_word_ids.append([word_to_id[w] if w in word_to_id else unknown_word_id for w in seg])
            refinement_word_ids = [word_to_id[w] if w in word_to_id else unknown_word_id for w in refinement]



def batch_iterator(word_to_id, batch_size, corpus, initial_buckets, all_buckets=False):
    #expand recipes to include blanks between each line

    max_phrase_num, max_phrase_len = _read_max_phrase_num_and_len()
    buckets=copy.copy(initial_buckets)
    buckets.append((2*max_phrase_num, max_phrase_len))


    start = True
    while True:
        if start:
            yield buckets
            start = False
        #loops continuously over corpus
        for filename in getDataFiles(corpus):
            labels, refinements, recipes = recipe_raw_data(word_to_id, buckets, filename)

            processed = recipe_iterator(labels, refinements, recipes, batch_size, all_buckets=all_buckets)
            for i,batch in enumerate(processed):
                if all_buckets:
                    labels, refinements, recipes = batch
                    for bucket in refinements:
                        refinements[bucket] = np.rollaxis(refinements[bucket], 1,0)
                        recipes[bucket] = np.rollaxis(np.rollaxis(recipes[bucket],1,0),2,1)
                        labels[bucket] = _index_to_ohe(labels[bucket], recipes[bucket].shape[0])
                    yield labels, refinements, recipes
                else:
                    bucket, labels, refinements, recipes= batch
                    bucket_id = buckets.index(bucket)
                    #change dimensions so batch is last
                    refinements = np.rollaxis(refinements, 1,0)
                    recipes = np.rollaxis(np.rollaxis(recipes,1,0),2,1)
                    labels  = _index_to_ohe(labels, recipes.shape[0])
                    if refinements.shape[-1] != FLAGS.batch_size or \
                            recipes.shape[-1] != FLAGS.batch_size or \
                            labels.shape[-1] != FLAGS.batch_size:
                                print "INCORRECT BATCH SIZE:"
                                print "refinements:", refinements.shape
                                print "recipes:", recipes.shape
                                print "labels:", labels.shape
                                continue
                    yield bucket_id, labels, refinements, recipes
def phrases2int(phrases,word_to_id, unknown_word_id,buckets,max_current_num_words):
    #TODO: do something very similar to _file_to_word_ids,
    #still need to do splitting, still need to add blanks, still need to find bucket
    split = [[w for w in phrase.split(' ') if len(w)>0] for phrase in phrases]
    #not exactly
    int_phrases,bucket = str_list_to_rnn_format(split,
                        word_to_id, unknown_word_id,buckets,0)
    return int_phrases

def getLabeledFiles(directory):
    label_files = [os.path.join(directory,f) for f in os.listdir(directory) if (
                                    os.path.isfile(os.path.join(directory,f)) and
                                    os.path.join(directory,f).endswith('.p') and
                                    os.path.join(directory,f).find('label') > -1)]
    data_files = [os.path.join(directory,f) for f in os.listdir(directory) if (
                                    os.path.isfile(os.path.join(directory,f)) and
                                    os.path.join(directory,f).endswith('.p') and
                                    os.path.join(directory,f).find('label') == -1)]

    matches = []
    for l in label_files:
        for d in data_files:
            data_file_ending = d.split('/')[-1].split('test')[-1].split('.p')[0]
            label_ending = l.split('/')[-1].split('test')[-1].split('_label')[0]
            if label_ending == data_file_ending:
                matches.append((l,d))
                break
    return matches
def readLabelFile(filename):
    with open(filename, 'rb') as f:
        recipes = pickle.load(f)
        return recipes
def readData(label_file, data_file):
    labels = readLabelFile(label_file)
    with open(data_file, 'rb') as f:
        recipes = pickle.load(f)
        for recipe in recipes:
            if recipe in labels:
                yield recipes[recipe]['instructions'], recipes[recipe]['reviews'], labels[recipe]

def bucket_id_from_bucket(bucket,buckets):
    for j,_bucket in enumerate(buckets):
        if bucket == _bucket:
            return j


def end2end_iterator(word_to_id, buckets, label_directory):
    filenames = getLabeledFiles(label_directory)
    for label_file, data_file in filenames:
        for instructions, reviews, labels in readData(label_file, data_file):
            split_instructions = [[w for w in phrase.split(' ') if len(w)>0] for phrase in instructions]
            too_long = False
            for p in split_instructions:
                if len(p) + 1> buckets[-1][1]:
                    too_long = True
                    break
            if too_long:
                continue
            split_instructions.append([])#for <eor> token
            if len(split_instructions) > buckets[-1][0]:
                continue

            for i,review in enumerate(reviews):
                for j,refinement in enumerate(review):
                    split_refinement = [w for w in refinement.split(' ') if len(w)>0]
                    max_current_num_words = len(split_refinement) + 1
                    if max_current_num_words > buckets[-1][1]:
                        continue

                    label_tuple = labels[i][j]
                    if not label_tuple:
                        label = None
                    else:
                        r_type = label_tuple[1]
                        r_index = label_tuple[0]
                        if r_type == 'm':
                            label = 2*r_index + 1
                        elif r_type == 'i':
                            label = 2*r_index
                        else:
                            raise ValueError, "r_type must be m or i, not %s"%r_type

                    nprecipe, bucket = recipe_str_list_to_rnn_format(copy.copy(split_instructions), word_to_id, UNK_ID, buckets, max_current_num_words)
                    nprecipe = np.expand_dims(nprecipe,axis=2)

                    nprefinement = refinement_str_list_to_rnn_format(split_refinement, word_to_id, UNK_ID,bucket)
                    nprefinement = np.expand_dims(nprefinement,axis=1)

                    bucket_id = bucket_id_from_bucket(bucket, buckets)

                    if label:
                        label  = _index_to_ohe(np.expand_dims(label, axis=0), nprecipe.shape[0])
                    else:
                        label = np.zeros((nprecipe.shape[0],1))
                    yield refinement, nprecipe, nprefinement, label, bucket_id

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
