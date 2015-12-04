# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Binary for training translation models and decoding from them.
Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.
Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.
See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/pdf/1412.2007v2.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.rnn.translate import data_utils
from tensorflow.models.rnn.translate import seq2seq_model
from tensorflow.python.platform import gfile


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("pre_generation_batch_size", 64,
                            "Batch size to use before generating refinements.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_initial_buckets = [(10,15), (15,20), (20,25), (25,30), (30,35)]


# def read_data(source_path, target_path, max_size=None):
  # """Read data from source and target files and put into buckets.
  # Args:
    # source_path: path to the files with token-ids for the source language.
    # target_path: path to the file with token-ids for the target language;
      # it must be aligned with the source file: n-th line contains the desired
      # output for n-th line from the source_path.
    # max_size: maximum number of lines to read, all other will be ignored;
      # if 0 or None, data files will be read completely (no limit).
  # Returns:
    # data_set: a list of length len(_buckets); data_set[n] contains a list of
      # (source, target) pairs read from the provided data files that fit
      # into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      # len(target) < _buckets[n][1]; source and target are lists of token-ids.
  # """
  # data_set = [[] for _ in _buckets]
  # with gfile.GFile(source_path, mode="r") as source_file:
    # with gfile.GFile(target_path, mode="r") as target_file:
      # source, target = source_file.readline(), target_file.readline()
      # counter = 0
      # while source and target and (not max_size or counter < max_size):
        # counter += 1
        # if counter % 100000 == 0:
          # print("  reading data line %d" % counter)
          # sys.stdout.flush()
        # source_ids = [int(x) for x in source.split()]
        # target_ids = [int(x) for x in target.split()]
        # target_ids.append(data_utils.EOS_ID)
        # for bucket_id, (source_size, target_size) in enumerate(_buckets):
          # if len(source_ids) < source_size and len(target_ids) < target_size:
            # data_set[bucket_id].append([source_ids, target_ids])
            # break
        # source, target = source_file.readline(), target_file.readline()
  # return data_set


def create_model(session, buckets, forward_only):
  """Create translation model and initialize or load parameters in session."""
  model = net_defn.RecipeNet(
      FLAGS.vocab_size, buckets,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      use_lstm=True, forward_only=forward_only)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.variables.initialize_all_variables())
  return model


def train():
  # Prepare WMT data.
  print("Preparing WMT data in %s" % FLAGS.data_dir)
  word_to_id = reader.build_vocab(os.path.join(FLAGS.data_dir, "recipes_train.txt"))

  with tf.Session() as sess:


    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = reader.batch_iterator(FLAGS.pre_generation_batch_size, 'val', _buckets, all_buckets=True)
    train_set = reader.batch_iterator(FLAGS.pre_generation_batch_size, 'train', _buckets, all_buckets=False)
    #First thing both return is the buckets, everything after is data
    buckets = train_set.next()
    dev_set.next()

    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, buckets, False)

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      # Get a batch and make a step.
      start_time = time.time()
      bucket_id, target_weights, encoder_inputs, decoder_inputs = train_set.next()

      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        target_weights, encoder_inputs, decoder_inputs= dev_set.next()
        for bucket_id in target_weights:
          _, eval_loss, _ = model.step(sess, encoder_inputs[bucket_id], decoder_inputs[bucket_id],
                                       target_weights[bucket_id], bucket_id, True)
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()


def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, buckets, True)
    model.batch_size = 1  # We decode one sentence at a time.

    #REDO:
    # # Load vocabularies.
    # en_vocab_path = os.path.join(FLAGS.data_dir,
                                 # "vocab%d.en" % FLAGS.en_vocab_size)
    # fr_vocab_path = os.path.join(FLAGS.data_dir,
                                 # "vocab%d.fr" % FLAGS.fr_vocab_size)
    # en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
    # _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(sentence, en_vocab)
      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print(" ".join([rev_fr_vocab[output] for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()



def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
