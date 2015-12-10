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
import util

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile
import reader


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 600, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_input_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("num_output_layers", 1, "Number of layers in the model.")

tf.app.flags.DEFINE_string("data_dir", "/local/robotChef/recipe-modifier", "Data directory")
tf.app.flags.DEFINE_string("train_corpus", "recipes_train.txt", "Training data file")
tf.app.flags.DEFINE_string("val_corpus", "recipes_valid.txt", "Validation data file")
tf.app.flags.DEFINE_string("test_corpus", "recipes_test.txt", "Test data file")
tf.app.flags.DEFINE_string("max_phrases_file", "max_phrases.txt", "File that contains max_num_phrases and max_len_phrase")
tf.app.flags.DEFINE_string("train_dir", "/local/robotChef/recipe-modifier/train", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS
import net_defn

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_initial_buckets = [(10,15), (15,20), (20,25), (25,30), (30,35)]



def create_model(session, vocab_size, buckets, forward_only):
  """Create translation model and initialize or load parameters in session."""
  model = net_defn.RecipeNet(
      vocab_size, buckets,
      FLAGS.size, FLAGS.num_input_layers, FLAGS.num_output_layers,
      FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      use_lstm=True, forward_only=forward_only)
  summary_op = tf.merge_all_summaries()
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
  word_to_id = reader.build_vocab()
  vocab_size = len(word_to_id)

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=False)) as sess, tf.device('/cpu:0'):


    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = reader.batch_iterator(word_to_id, FLAGS.batch_size, 'val', _initial_buckets, all_buckets=True)
    train_set = reader.batch_iterator(word_to_id, FLAGS.batch_size, 'train', _initial_buckets, all_buckets=False)
    #First thing both return is the buckets, everything after is data
    buckets = train_set.next()
    dev_set.next()
    print("BUCKETS:", buckets)

    # Create model.
    print("Creating %d input layers and %d output layers of %d units." % (FLAGS.num_input_layers, FLAGS.num_output_layers, FLAGS.size))
    model = create_model(sess, vocab_size, buckets, False)
    print ("created model")

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      # Get a batch and make a step.
      start_time = time.time()
      bucket_id, target_weights, refinement_segment, recipe_segments = train_set.next()
      print("got first batch")
      print("bucket_id:", bucket_id)
      print("target_weights:", target_weights.shape)
      print("recipe_segments:", recipe_segments.shape)
      print("target_weights:", target_weights)
      print("recipe_segments:", recipe_segments)
      print("recipe_segments:", refinement_segment)

      _, step_loss, _ = model.step(sess, refinement_segment, recipe_segments,
                                   target_weights, bucket_id, False)
      print("got first loss", step_loss)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1
      sys.exit()

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
        target_weights, refinement_segment, recipe_segments = dev_set.next()
        for bucket_id in target_weights:
          _, eval_loss, _ = model.step(sess, refinement_segment[bucket_id], recipe_segments[bucket_id],
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
  if FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
