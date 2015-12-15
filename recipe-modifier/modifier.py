from __future__ import absolute_import
# from __future__ import division
from __future__ import print_function

import time

import tensorflow.python.platform

import numpy as np
import tensorflow as tf
import sys,os
import math

from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
from tensorflow.python.platform import gfile
import cPickle as pickle

import reader as reader


tf.app.flags.DEFINE_string("train_dir", '/local/robotChef/recipe-modifier/train', "Training directory where checkpoints should be saved.")
#tf.app.flags.DEFINE_string("data_dir", '/local/robotChef/recipe-modifier/dataset', "Data directory.")
tf.app.flags.DEFINE_string("data_dir", '/local/robotChef/recipe-modifier/full_sentence_dataset', "Data directory.")
tf.app.flags.DEFINE_string("vocab_file", '/local/robotChef/recipe-modifier/vocab.p', "vocab file.")
tf.app.flags.DEFINE_string("vocab_dict_file", '/local/robotChef/recipe-modifier/vocab_dict.p', "vocab dict file.")
tf.app.flags.DEFINE_string("model_path", '/local/robotChef/recipe-modifier/train', "Path for trained model.")

tf.app.flags.DEFINE_string("max_phrases_file", "max_phrases.txt", "File that contains max_num_phrases and max_len_phrase")
tf.app.flags.DEFINE_integer("batch_size", 4,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("steps_per_summary", 100,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("steps_per_lr_decay", 200,
                            "Batch size to use during training.")

FLAGS = tf.app.flags.FLAGS

#FOR FULL SENTENCES
_initial_buckets = [(20,15), (30,20), (40,25), (50,30), (60,35)]

#FOR PHRASES
#_initial_buckets = [(10,15), (16,25), (22,30), (28,40), (34,50)]




class RecipeNet(object):
    def __init__(self, is_training, config,batch_size=FLAGS.batch_size,do_train=True):
        self._batch_size = batch_size


        encoder_size = config.encoder_hidden_size
        vocab_size = config.vocab_size
        self.max_phrase_num = config.buckets[-1][0]
        self.max_sequence_length = config.buckets[-1][0]
        self.max_phrase_len = config.buckets[-1][1]
        self.buckets = config.buckets
        self._lr_decay = config.lr_decay
        self.max_grad_norm = config.max_grad_norm
        self.global_step = tf.Variable(0, trainable=False)
        self.init_scale = config.init_scale
        self.do_train = do_train


        self._input_refinement = tf.placeholder(tf.int32, [self._batch_size, self.max_phrase_len])
        self._input_refinement = []
        for i in xrange(self.max_phrase_len):
            self._input_refinement.append(tf.placeholder(tf.int32, shape=[self._batch_size],
                                                    name="refinement_{0}".format(i)))

        self._target = []
        self._input_recipe_segments = []
        for i in xrange(self.max_sequence_length):
            self._target.append(tf.placeholder(tf.int32, shape=[self._batch_size],
                                                    name="target_{0}".format(i)))
            self._input_recipe_segments.append([])
            for j in xrange(self.max_phrase_len):
                self._input_recipe_segments[-1].append(tf.placeholder(tf.int32, shape=[self._batch_size],
                                                    name="recipe_segment{0}/{1}".format(i,j)))

        #ENCODER (1st LSTM Layer)

        encoder_lstm_cell = rnn_cell.BasicLSTMCell(encoder_size, forget_bias=0.0)
        if is_training and config.keep_prob < 1:
            encoder_lstm_cell = rnn_cell.DropoutWrapper(encoder_lstm_cell, output_keep_prob=config.keep_prob)
        self.encoder = rnn_cell.MultiRNNCell([encoder_lstm_cell] * config.num_layers)

        self._initial_encoder_state = self.encoder.zero_state(self._batch_size, tf.float32)#tf.ones([self._batch_size, config.num_layers * encoder_lstm_cell.state_size])
        self._embedding_size = config.num_layers * int(encoder_lstm_cell.state_size)
        with tf.device('/cpu:0'):
            self._embedding_matrix = tf.get_variable("embedding_matrix", [vocab_size, self._embedding_size])
            tf.histogram_summary('embedding_matrix', self._embedding_matrix)

        #RECIPE PROCESSOR (2nd LSTM Layer)
        recipe_processor_size = config.recipe_processor_hidden_size

        with tf.variable_scope("recipe_processor_cell"):
            recipe_processor_lstm_cell = rnn_cell.BasicLSTMCell(recipe_processor_size, forget_bias=0.0)
            if is_training and config.keep_prob < 1:
                recipe_processor_lstm_cell = rnn_cell.DropoutWrapper(recipe_processor_lstm_cell, output_keep_prob=config.keep_prob)
                self.recipe_processor = rnn_cell.MultiRNNCell([recipe_processor_lstm_cell] * config.num_layers)
            self._initial_recipe_processor_state = self.recipe_processor.zero_state(self._batch_size, tf.float32)#tf.ones([self._batch_size, recipe_processor_size])

        #FINAL REDUCTION TO DISTRIBUTION OVER INDICES
        self.index_predictor_W = weight_variable([recipe_processor_size, 2])
        tf.histogram_summary('index_predictor_w', self.index_predictor_W)

        self.index_predictor_b = bias_variable([2])
        tf.histogram_summary('index_predictor_b', self.index_predictor_b)
        self._lr = tf.Variable(float(config.learning_rate), trainable=False)
        tf.scalar_summary('lr', self._lr)

        self.learning_rate_decay_op = self._lr.assign(
            self._lr * self._lr_decay)

        #BUILD MODEL
        self.outputs, self.losses, self.costs = self.model_with_buckets()
        #CALC GRADIENTS
        if not self.do_train:
            self.calc_gradients()
        self.saver = tf.train.Saver(tf.all_variables())

    def model_with_buckets(self):
        #build an rnn model for each bucket, since tensor flow can't deal with variable length sequences
        #variables are shared across the different buckets, and if you only ask for the outputs
        #up to a certain bucket, then additional computation won't be done past teh steps in that bucket

        #UNCOMMENT FOR ATTENTION
        all_inputs = self._input_refinement+ [seg for seg in self._input_recipe_segments] + self._target
        costs = []
        losses = []
        outputs = []
        with tf.op_scope(all_inputs, None, "model_with_buckets"):
            for j in xrange(len(self.buckets)):
                if j > 0:
                    outside_reuse = True
                else:
                    outside_reuse = None
                with tf.variable_scope("bucket_model_outside",reuse=outside_reuse):
                    phrase_num = self.buckets[j][0]
                    phrase_len = self.buckets[j][1]
                    bucket_refinement_inputs = [self._input_refinement[i]
                                                for i in xrange(phrase_len)]
                    bucket_recipe_segments_inputs = []
                    bucket_target = []
                    bucket_weights = []
                forward_attention_weights = []
                #backward_attention_weights = []
                for i in xrange(phrase_num):
                    with tf.variable_scope("bucket_model_outside",reuse=outside_reuse):
                        bucket_target.append(self._target[i])
                        bucket_weights.append(tf.constant(1, dtype=np.float32, shape=[self._batch_size]))
                    # if j > 0 and i < self.buckets[j-1][0]:
                        # with tf.variable_scope("attention", reuse=True):
                            # forward_weight = tf.get_variable("forward_attention_weight%d"%(i))
                            #backward_weight = tf.get_variable("backward_attention_weight%d"%(i))
                    # else:
                        # with tf.variable_scope("attention", reuse=None):
                            # forward_weight = tf.get_variable("forward_attention_weight%d"%(i), [self._batch_size], dtype=tf.float32)
                            #backward_weight = tf.get_variable("backward_attention_weight%d"%(i), [self._batch_size], dtype=tf.float32)
                    # forward_attention_weights.append(forward_weight)
                    #backward_attention_weights.append(backward_weight)
                    with tf.variable_scope("bucket_model_outside",reuse=outside_reuse):
                        bucket_recipe_segments_inputs.append([])
                        for k in xrange(phrase_len):
                            bucket_recipe_segments_inputs[-1].append(self._input_recipe_segments[i][k])

                with tf.variable_scope("bucket_model_outside",reuse=outside_reuse):
                    bucket_logits = self.build_rnn_model(bucket_refinement_inputs, bucket_recipe_segments_inputs)
                                                        #forward_attention_weights)#, backward_attention_weights)
                    outputs.append([tf.nn.softmax(bucket_logit) for bucket_logit in bucket_logits])
                    loss = seq2seq.sequence_loss_by_example(bucket_logits,
                                                    bucket_target,
                                                    bucket_weights,2)
                    losses.append(loss)
                    costs.append(tf.reduce_sum(loss))
                    tf.histogram_summary("cost_bucket_%d"%j, costs[-1])
        # for i,f in enumerate(forward_attention_weights):
            # tf.histogram_summary("forward_attention_weight%d"%i, f)
            #tf.histogram_summary("backward_attention_weight%d"%i, backward_attention_weights[i])
        return outputs, losses, costs

    def get_encoded_segment(self, segment, reuse):
        inputs = tf.split(0, len(segment), tf.nn.embedding_lookup(self._embedding_matrix, tf.pack(segment)))
        inputs = [tf.squeeze(input_, [0]) for input_ in inputs]
        with tf.variable_scope("encoder", reuse=reuse):
            encoder_outputs, encoder_states = rnn.rnn(self.encoder, inputs, initial_state=self._initial_encoder_state)
        return encoder_outputs[-1]

    def build_rnn_model(self, refinement, recipe_segments):#), forward_attention_weights):#, backward_attention_weights):
        #UNCOMMENT THESE COMMENTS TO RUN ATTENTION MODEL
        encoded_refinement = self.get_encoded_segment(refinement, None)

        encoded_recipe_segments = []
        for i,segment in enumerate(recipe_segments):
            encoded_recipe_segments.append(self.get_encoded_segment(segment, True))

        inputs = [tf.concat(1,[encoded_refinement, seg]) for seg in encoded_recipe_segments]
        #backward_inputs = [tf.concat(1,[encoded_refinement, seg]) for seg in list(reversed(encoded_recipe_segments))]

        with tf.variable_scope("recipe_processor") as scope:
            recipe_processor_outputs, recipe_processor_states = rnn.rnn(self.recipe_processor, inputs, initial_state=self._initial_recipe_processor_state)
            #scope.reuse_variables()
            #backward_recipe_processor_outputs, backward_recipe_processor_states = rnn.rnn(self.recipe_processor, inputs, initial_state=self._initial_recipe_processor_state)

        logits_per_index = []
        for i,output in enumerate(recipe_processor_outputs):
            #forward_logit = tf.mul(tf.transpose(tf.pack([forward_attention_weights[i], forward_attention_weights[i]])),
            #                                tf.nn.xw_plus_b(output, self.index_predictor_W, self.index_predictor_b))
            forward_logit = tf.nn.xw_plus_b(output, self.index_predictor_W, self.index_predictor_b)
            #backward_logit = tf.mul(tf.transpose(tf.pack([backward_attention_weights[i], backward_attention_weights[i]])),
            #                            tf.nn.xw_plus_b(backward_recipe_processor_outputs[-(i+1)], self.index_predictor_W, self.index_predictor_b))
            #logits_per_index.append(forward_logit + backward_logit)
            logits_per_index.append(forward_logit)
        return logits_per_index

    def calc_gradients(self):
        self.gradient_norms = []
        self.updates = []
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        params = tf.trainable_variables()
        for b in xrange(len(self.buckets)):
            gradients = tf.gradients(self.costs[b],params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                self.max_grad_norm)
            self.gradient_norms.append(norm)
            self.updates.append(optimizer.apply_gradients(
                zip(clipped_gradients,params),global_step=self.global_step))


    def step(self, session, refinement, recipe_segments, target, bucket_id, forward_only,
                        summary_op, run_summary):
        phrase_num, phrase_len = self.buckets[bucket_id]
        max_phrase_num, max_phrase_len = self.buckets[-1]
        if len(refinement) != phrase_len:
            raise ValueError("Refinement length must be equal to the one in bucket,"
                           " %d != %d." % (len(encoder_inputs), phrase_len))
        if len(recipe_segments) != phrase_num:
            raise ValueError("Recipe length must be equal to the one in bucket,"
                           " %d != %d." % (len(recipe_segments), phrase_num))
        if len(recipe_segments[0]) != phrase_len:
            raise ValueError("Recipe segment length must be equal to the one in bucket,"
                           " %d != %d." % (len(recipe_segments[0]), phrase_len))

        if len(target) != phrase_num:
            raise ValueError("Target length must be equal to the one in bucket,"
                           " %d != %d." % (len(target_weights), phrase_num))
        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(max_phrase_len):
            if l < phrase_len:
                input_feed[self._input_refinement[l].name] = refinement[l]
            else:
                input_feed[self._input_refinement[l].name] = np.zeros(self._batch_size)
        for l in xrange(max_phrase_num):
            if l < phrase_num:
                input_feed[self._target[l].name] = target[l]
            else:
                input_feed[self._target[l].name] = np.zeros(self._batch_size)
            for k in xrange(max_phrase_len):
                if k < phrase_len and l < phrase_num:
                    input_feed[self._input_recipe_segments[l][k].name] = recipe_segments[l][k]
                else:
                    input_feed[self._input_recipe_segments[l][k].name] = np.zeros(self._batch_size)


        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                         self.gradient_norms[bucket_id],  # Gradient norm.
                         self.costs[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.costs[bucket_id]]  # Loss for this batch.
            for l in xrange(phrase_num):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])
        if run_summary:
            output_feed.insert(0,summary_op)
        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            if run_summary:
                return outputs[0], outputs[2], outputs[3], None # Gradient norm, loss, no outputs.
            else:
                return None, outputs[1], outputs[2], None # Gradient norm, loss, no outputs.
        else:
            if run_summary:
                return outputs[0], None, outputs[1], outputs[2:]  # No gradient norm, loss, outputs.
            else:
                return None, None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))


    @property
    def cost(self):
        return self._cost


    @property
    def lr(self):
        return self._lr
    @property
    def lr_decay(self):
        return self._lr_decay


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class Config(object):
    """Configuration parameters."""
    init_scale = 0.1
    learning_rate = 0.5
    max_grad_norm = 5
    num_layers = 1
    encoder_hidden_size = 200
    recipe_processor_hidden_size = 2*encoder_hidden_size  # must be 2x the size of encoder_hidden_size
    max_epoch = 4
    max_max_epoch = 8
    keep_prob = 0.8
    lr_decay = 0.99

    def __init__(self, vocab_size, buckets):
        self.vocab_size = vocab_size
        self.buckets = buckets

def create_model(session, config, is_training,batch_size=FLAGS.batch_size,do_train=True):
    """Create translation model and initialize or load parameters in session."""

    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    with tf.variable_scope("model", initializer=initializer):
        model = RecipeNet(is_training, config, batch_size=batch_size,do_train=do_train)

    summary_op = tf.merge_all_summaries()
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    sys.stdout.flush()
    if is_training and ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        sys.exit()
        session.run(tf.initialize_all_variables())

    return model, summary_op




def train():
    word_to_id = reader.build_vocab()
    vocab_size = len(word_to_id)

    dev_set = reader.batch_iterator(word_to_id, FLAGS.batch_size, 'val', _initial_buckets, all_buckets=True)
    train_set = reader.batch_iterator(word_to_id, FLAGS.batch_size, 'train', _initial_buckets, all_buckets=False)

    #First thing both return is the buckets, everything after is data
    buckets = train_set.next()
    dev_set.next()
    #trim last buckets so we don't run into memory issues
    buckets = _initial_buckets[:4]
    #buckets = _initial_buckets


    config = Config(vocab_size, buckets)

    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        model,summary_op = create_model(sess, config, True)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                            graph_def=sess.graph_def)
        # This is the training loop.
        step_time, cost = 0.0, 0.0
        current_step = 0
        previous_costs = []
        while True:
            # Get a batch and make a step.
            start_time = time.time()
            bucket_id, target_weights, refinement_segment, recipe_segments = train_set.next()
            if bucket_id > 3:
                continue

            #for testing with small random set:
            # bucket_id = 0
            # target_weights = np.random.randint(2,size=(2,FLAGS.batch_size))
            # refinement_segment = np.random.randint(50,size=(2,FLAGS.batch_size))
            # recipe_segments = [np.random.randint(50,size=(2,FLAGS.batch_size)),np.random.randint(50,size=(2,FLAGS.batch_size))]

            run_summary = False
            if current_step % FLAGS.steps_per_summary == 0:
                run_summary = True
            summary_str, _, step_cost, _ = model.step(sess, refinement_segment, recipe_segments,
                                         target_weights, bucket_id, False, summary_op, run_summary)

            if current_step % FLAGS.steps_per_summary == 0:
                summary_writer.add_summary(summary_str, current_step)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            cost += step_cost / FLAGS.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(cost) if cost < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (model.global_step.eval(), model.lr.eval(),
                                 step_time, perplexity))
                sys.stdout.flush()
                # Decrease learning rate if no improvement was seen over last 3 times.
                if current_step % FLAGS.steps_per_lr_decay == 0:
                    if len(previous_costs) > 2 and cost > max(previous_costs[-3:]):
                        sess.run(model.learning_rate_decay_op)
                    previous_costs.append(cost)

                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, cost = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                target_weights, refinement_segment, recipe_segments = dev_set.next()
                for bucket_id in target_weights:
                    if bucket_id > 3:
                        continue
                    summary_str, _, eval_cost, _ = model.step(sess, refinement_segment[bucket_id], recipe_segments[bucket_id],
                                                 target_weights[bucket_id], bucket_id, True, summary_op, False)
                    eval_ppx = math.exp(eval_cost) if eval_cost < 300 else float('inf')
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()



def do_eval():

    word_to_id = reader.build_vocab()
    vocab_size = len(word_to_id)

    buckets = _initial_buckets[:4]
    label_directory = '/local/robotChef/recipe-modifier/end2endtest'
    iterator = reader.end2end_iterator(word_to_id, buckets, label_directory)


    config = Config(vocab_size, buckets)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model,summary_op = create_model(sess, config, True, batch_size=1, do_train=False)

        #perplexities = []
        predicted_indices = []
        tp = {}
        fp = {}
        tn = {}
        fn = {}
        current_indices = {}
        total_examples = 0
        lm_scores = pickle.load(open('lm_scores.p','rb'))
        for step, data in enumerate(iterator):
            print('step',step)
            raw_refinement, nprecipe, nprefinement, label,bucket_id = data

            lm_score = lm_scores[raw_refinement]

            _, _, cost, output_logits = model.step(sess, nprefinement, nprecipe,
                                                 label, bucket_id, True, summary_op, False)
            print('got step')

            #here multiply lm_score by each argmax of each output_logit
            #get top_1,top_2,top_3 score
            outputs = [lm_score*logit[0,1] for logit in output_logits]
            calc_scores(outputs,label,tp,fp,tn,fn, current_indices)
        return tp,fp,tn,fn,current_indices
def calc_scores(outputs,label,tp_scores, fp_scores,tn_scores, fn_scores,current_indices):

    for threshold in np.arange(0,1,.0001):
        if threshold not in tp_scores:
            tp_scores[threshold] = [0,0,0]
        if threshold not in fp_scores:
            fp_scores[threshold] = [0,0,0]
        if threshold not in tn_scores:
            tn_scores[threshold] = [0,0,0]
        if threshold not in fn_scores:
            fn_scores[threshold] = [0,0,0]
        if threshold not in current_indices:
            current_indices[threshold] = []

        max_scores = [-np.inf]*3
        max_scores_indices = [None,None,None]
        for j,output in enumerate(outputs):
            if output > threshold:
                for i,score in enumerate(max_scores):
                    if output > score:
                        max_scores[i] = output
                        max_scores_indices[i] = j
                        break
        label_index = np.where(label[:,0]==1)[0]
        if len(label_index) > 0:
            label_index = label_index[0]
            #contains a 1, meaning there is a refinement
            if label_index == max_scores_indices[0]:
                tp_scores[threshold][0] += 1
                tp_scores[threshold][1] += 1
                tp_scores[threshold][2] += 1
            elif label_index == max_scores_indices[1]:
                fn_scores[threshold][0] += 1
                tp_scores[threshold][1] += 1
                tp_scores[threshold][2] += 1
            elif label_index == max_scores_indices[2]:
                fn_scores[threshold][0] += 1
                fn_scores[threshold][1] += 1
                tp_scores[threshold][2] += 1
            else:
                fn_scores[threshold][0] += 1
                fn_scores[threshold][1] += 1
                fn_scores[threshold][2] += 1
        else:
            #correctly predicted that this is not a refinement
            if max_scores_indices[0] == None and max_scores_indices[1] == None and max_scores_indices[2] == None:
                tn_scores[threshold][0] += 1
                tn_scores[threshold][1] += 1
                tn_scores[threshold][2] += 1
            else:
                fp_scores[threshold][0] += 1
                fp_scores[threshold][1] += 1
                fp_scores[threshold][2] += 1

        current_indices[threshold].append([m for m in max_scores_indices if m])


def main(unused_args):
    #train()
    tp,fp,tn,fn,indices = do_eval()
    print("TP")
    print(tp)
    print("FP")
    print(fp)
    print("TP")
    print(tn)
    print("FN")
    print(fn)
    print("INDICES")
    print(indices)
    info = [tp,fp,tn,fn,indices]
    pickle.dump(info, open('save_data.p','wb'))


if __name__ == "__main__":
  tf.app.run()
