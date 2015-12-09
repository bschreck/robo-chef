from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import seq2seq
#make sure it works with buckets defined in reader.py ((num_phrases, num_words_per_phrase))

#Write label creation file, which removes random phrases from recipes,
#optionally modifies them, and if it modifies it optionally puts the
#original back in the same location

#Redo class below to create embeddings for each phrase in recipe
#Pass refinement embedding as first decoder input
#Pass recipe phrase embeddings as rest of decoder inputs (with blanks in between each phrase)
#Also reverse order of phrase embeddings and do a separate decoding (or maybe there is a ready made attention function for this)
#Project output to vector of length 2*(number of phrases), and input that into softmax to determine loss



class RecipeNet(object):
    """Sequence-to-sequence model with attention and for multiple buckets.
    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
    or into the seq2seq library for complete model implementation.
    This class also allows to use GRU cells in addition to LSTM cells, and
    sampled softmax to handle large output vocabulary size. A single-layer
    version of this model, but with bi-directional encoder, was presented in
      http://arxiv.org/abs/1409.0473
    and sampled softmax is described in Section 3 of the following paper.
      http://arxiv.org/pdf/1412.2007v2.pdf
    """
    def __init__(self, vocab_size, buckets, size,
                 num_input_layers, num_output_layers, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor, use_lstm=False,
                 forward_only=False):
        """Create the model.
        Args:
          vocab_size: size of the vocabulary.
          buckets: a list of pairs (I, O), where I specifies maximum input length
            that will be processed in that bucket, and O specifies maximum output
            length. Training instances that have inputs longer than I or outputs
            longer than O will be pushed to the next bucket and padded accordingly.
            We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
          size: number of units in each layer of the model.
          num_layers: number of layers in the model.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          use_lstm: if true, we use LSTM cells instead of GRU cells.
          num_samples: number of samples for sampled softmax.
          forward_only: if set, we do not construct the backward pass in the model.
        """
        self.vocab_size = vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None

        # Create the internal multi-layer cell for our RNN.
        single_input_cell = rnn_cell.GRUCell(size)
        if use_lstm:
            single_input_cell = rnn_cell.BasicLSTMCell(size)

        #output cell projects to a weight distribution of size max phrase_num (last bucket)
        single_output_cell = rnn_cell.LSTMCell(size, size,2*self.buckets[-1][0])
        input_cell = single_input_cell
        output_cell = single_output_cell
        if num_input_layers > 1:
            input_cell = rnn_cell.MultiRNNCell([single_input_cell] * num_input_layers)
        if num_output_layers > 1:
            output_cell = rnn_cell.MultiRNNCell([single_output_cell] * num_output_layers)
        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs):
            return self.multi_layer_embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, input_cell, output_cell, vocab_size)
        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][1]+1):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="encoder{0}".format(i)))
        #target weights is 2*len(max_phrase_num) + 1 (+1 for end of recipe phrase)

        for i in xrange(2*buckets[-1][0]+1):
            self.target_weights.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="weight{0}".format(i)))
            self.decoder_inputs.append([])
            for j in xrange(buckets[-1][1]+1):
                self.decoder_inputs[-1].append(tf.placeholder(tf.int32, shape=[None],
                                                    name="decoder{0},{1}".format(i,j)))

        # Training outputs and losses.
        if forward_only:
            self.outputs, self.losses = self.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y))
        else:
            self.outputs, self.losses = self.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y))
        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in xrange(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))
        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             bucket_id, forward_only):
        """Run a step of the model feeding the given inputs.
        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.
          forward_only: whether to do the backward step or only forward.
        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        Raises:
          ValueError: if length of enconder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        phrase_num, phrase_len = self.buckets[bucket_id]
        if len(encoder_inputs) != phrase_len:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                           " %d != %d." % (len(encoder_inputs), phrase_len))
        if len(decoder_inputs) != 2*phrase_num+1:
            raise ValueError("Decoder outer length must be equal to the one in bucket,"
                           " %d != 2*%d." % (len(decoder_inputs), 2*phrase_num+1))
        if len(decoder_inputs[0]) != phrase_len:
            raise ValueError("Decoder inner length must be equal to the one in bucket,"
                           " %d != %d." % (len(decoder_inputs), phrase_len))

        if len(target_weights) != 2*phrase_num:
            raise ValueError("Weights length must be equal to the one in bucket,"
                           " %d != 2*%d." % (len(target_weights), 2*phrase_num))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(phrase_len):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(phrase_num):
            input_feed[self.target_weights[l].name] = target_weights[l]
            for k in xrange(phrase_len):
                input_feed[self.decoder_inputs[l][k].name] = decoder_inputs[l][k]

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                         self.gradient_norms[bucket_id],  # Gradient norm.
                         self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for l in xrange(phrase_num):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])
        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

    def multi_layer_embedding_attention_seq2seq(self,
            encoder_inputs, decoder_inputs, input_cell, output_cell, embedding_size, dtype=tf.float32, scope=None):
        with tf.variable_scope(scope or "multi_layer_embedding_attention_seq2seq"):
            with tf.variable_scope("sublevel_embedding_attention_seq2seq"):
                # Encoder.
                encoder_cell = rnn_cell.EmbeddingWrapper(input_cell, embedding_size)
                encoder_outputs, encoder_states = rnn.rnn(
                    encoder_cell, encoder_inputs, dtype=dtype)

                decoder_embedding_cells = []
                decoder_embedding_outputs = []
                decoder_embedding_states = []
                for decoder_input in decoder_inputs:
                    tf.get_variable_scope().reuse_variables()
                    decoder_embedding_cell = rnn_cell.EmbeddingWrapper(input_cell, embedding_size)
                    decoder_embedding_cells.append(decoder_embedding_cell)
                    decoder_embedding_output,decoder_embedding_state = rnn.rnn(
                            decoder_embedding_cell, decoder_input, dtype=dtype)
                    decoder_embedding_outputs.append(decoder_embedding_output)
                    decoder_embedding_states.append(decoder_embedding_state)

            #TODO: do this for encoder cell and each decoder embedding cell
            # # First calculate a concatenation of encoder outputs to put attention on.
            # top_states = [tf.reshape(e, [-1, 1, cell.output_size])
                          # for e in encoder_outputs]
            # attention_states = tf.concat(1, top_states)

            return seq2seq.rnn_decoder([tf.concat(1, [d[-1], encoder_states[-1]]) for d in decoder_embedding_states], tf.zeros_like(encoder_states[-1]),output_cell)
            # return seq2seq.rnn_decoder([d[-1] for d in decoder_embedding_states], encoder_states[-1],output_cell)

    def model_with_buckets(self, encoder_inputs, decoder_inputs, targets,
                   buckets, seq2seq_func,name=None):
            #encoder_inputs size= (phrase_len, batch_size)
            #decoder_inputs size = (phrase_num, phrase_len, batch_size)
            #targets size = (phrase, batch_size)
        """Create a sequence-to-sequence model with support for bucketing.
        The seq2seq argument is a function that defines a sequence-to-sequence model,
        e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))
        Args:
          encoder_inputs: a list of Tensors to feed the encoder; first seq2seq input.
          decoder_inputs: a list of Tensors to feed the decoder; second seq2seq input.
          targets: a list of 1D batch-sized int32-Tensors (desired output sequence).
          weights: list of 1D batch-sized float-Tensors to weight the targets.
          buckets: a list of pairs of (input size, output size) for each bucket.
          num_decoder_symbols: integer, number of decoder symbols (output classes).
          seq2seq: a sequence-to-sequence model function; it takes 2 input that
            agree with encoder_inputs and decoder_inputs, and returns a pair
            consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
          softmax_loss_function: function (inputs-batch, labels-batch) -> loss-batch
            to be used instead of the standard softmax (the default if this is None).
          name: optional name for this operation, defaults to "model_with_buckets".
        Returns:
          outputs: The outputs for each bucket. Its j'th element consists of a list
            of 2D Tensors of shape [batch_size x num_decoder_symbols] (j'th outputs).
          losses: List of scalar Tensors, representing losses for each bucket.
        Raises:
          ValueError: if length of encoder_inputsut, targets, or weights is smaller
            than the largest (last) bucket.
        """
        if len(encoder_inputs) < buckets[-1][1]+1:
            raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                           "st bucket (%d)." % (len(encoder_inputs), buckets[-1][1]+1))
        if len(decoder_inputs) < 2*buckets[-1][0]+1:
            raise ValueError("Length of decoder_inputs (%d) must be at least twice that of last"
                           "bucket (%d)." % (len(decoder_inputs), buckets[-1][0]+1))
        if len(decoder_inputs[0]) < buckets[-1][1]+1:
            raise ValueError("dim[1] of decoder_inputs  (%d) must be at least that of last"
                           "bucket (%d)." % (len(decoder_inputs), buckets[-1][0]+1))
        if len(targets) < 2*buckets[-1][0]:
            raise ValueError("Length of targets (%d) must be at least twice that of last"
                           "bucket (%d)." % (len(targets), buckets[-1][0]))

        num_targets = 2*buckets[-1][0]
        all_inputs = encoder_inputs + [d for d in decoder_inputs] + targets
        losses = []
        outputs = []
        with tf.op_scope(all_inputs, name, "model_with_buckets"):
            for j in xrange(len(buckets)):
                if j > 0:
                    tf.get_variable_scope().reuse_variables()
                bucket_encoder_inputs = [encoder_inputs[i]
                                         for i in xrange(buckets[j][1]+1)]
                bucket_decoder_inputs = []
                for i in xrange(2*buckets[j][0]+1):
                    bucket_decoder_inputs.append([])
                    for k in xrange(buckets[j][1]+1):
                        bucket_decoder_inputs[-1].append(decoder_inputs[i][k])
                bucket_outputs, _ = seq2seq_func(bucket_encoder_inputs,
                                            bucket_decoder_inputs)
                outputs.append(bucket_outputs)

                output_seq_len = 2*buckets[j][0]
                bucket_targets = [targets[i] for i in xrange(output_seq_len)]
                bucket_targets.extend([tf.constant(0,dtype=np.int32,shape=[FLAGS.batch_size]) for i in xrange(num_targets-output_seq_len)])

                #TODO: eventually learn these as parameters:
                bucket_weights = [tf.constant(1, dtype=np.float32, shape=[FLAGS.batch_size]) for i in xrange(output_seq_len)]
                #TODO: also extend with constant zeros like in targets

                #TODO: is this function fundamentally different than what we want?
                losses.append(seq2seq.sequence_loss(
                    outputs[-1], bucket_targets, bucket_weights, num_targets))

        return outputs, losses
