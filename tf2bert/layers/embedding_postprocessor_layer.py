from collections import Sequence

import tensorflow as tf


class EmbeddingPostprocessorLayer(tf.keras.layers.Layer):

    def __init__(self, shape,
                 use_token_type=False,
                 token_type_vocab_size=16,
                 use_position_embeddings=True,
                 max_position_embeddings=512,
                 token_type_embeddings_initializer='glorot_uniform',
                 position_embeddings_initializer='glorot_uniform',
                 layer_norm_beta_initializer='zeros',
                 layer_norm_gamma_initializer='ones',
                 dropout_prob=0.1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = shape
        self.use_token_type = use_token_type
        self.token_type_vocab_size = token_type_vocab_size
        self.use_position_embeddings = use_position_embeddings

        seq_length = shape[0]
        width = shape[1]

        if use_token_type:
            self.token_type_table = self.add_weight(
                'token_type_table',
                shape=[token_type_vocab_size, width],
                initializer=token_type_embeddings_initializer)

        tf.debugging.assert_less_equal(seq_length, max_position_embeddings)

        self.position_embeddings = self.add_weight(
            'full_position_embeddings',
            shape=[max_position_embeddings, width],
            initializer=position_embeddings_initializer)
        # Since the position embedding table is a learned variable, we create it
        # using a (long) sequence length `max_position_embeddings`. The actual
        # sequence length might be shorter than this, for faster training of
        # tasks that do not have long sequences.
        #
        # So `full_position_embeddings` is effectively an embedding table
        # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
        # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
        # perform a slice.
        # self.position_embeddings = tf.slice(self.full_position_embeddings, [0, 0], [seq_length, -1])
        self.dropout = tf.keras.layers.Dropout(dropout_prob)
        self.layer_norm = tf.keras.layers.LayerNormalization(beta_initializer=layer_norm_beta_initializer,
                                                             gamma_initializer=layer_norm_gamma_initializer)

    def call(self, inputs, mask=None, **kwargs):
        input_tensor = inputs
        token_type_ids = None
        if isinstance(inputs, Sequence):
            input_tensor = inputs[0]
            token_type_ids = inputs[1]

        output = input_tensor
        seq_length = self.shape[0]
        width = self.shape[1]
        if self.use_token_type:
            # This vocab will be small so we always do one-hot here, since it is always
            # faster for a small vocabulary.
            flat_token_type_ids = tf.reshape(token_type_ids, [-1])
            one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.token_type_vocab_size)
            token_type_embeddings = tf.matmul(one_hot_ids, self.token_type_table)
            token_type_embeddings = tf.reshape(token_type_embeddings, [-1, seq_length, width])
            output = tf.add(output, token_type_embeddings, name="add_token_type_embeddings")

        if self.use_position_embeddings:
            # Only the last two dimensions are relevant (`seq_length` and `width`), so
            # we broadcast among the first dimensions, which is typically just
            # the batch size.
            position_embeddings = tf.expand_dims(self.position_embeddings, 0)
            output = tf.add(output, position_embeddings, name="add_position_embeddings")
        output = self.dropout(output)
        output = self.layer_norm(output)
        return output

    def compute_mask(self, inputs, mask=None):
        return mask
