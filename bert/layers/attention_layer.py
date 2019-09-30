import math
from collections import Sequence

import tensorflow as tf

from bert import ckpt_initializer


class AttentionLayer(tf.keras.layers.Layer):

    @staticmethod
    def transpose_for_scores(input_tensor, num_attention_heads,
                             seq_length, width, name):
        output_tensor = tf.reshape(
            input_tensor, [-1, seq_length, num_attention_heads, width], name="%s_reshape" % name)

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    def __init__(self, from_shape, to_shape,
                 num_attention_heads=1,
                 size_per_head=512,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 attention_probs_dropout_prob=0.0,
                 do_return_2d_tensor=False,
                 ckpt=None,
                 ckpt_prefix=None,
                 *args, **kwargs):
        """
        :param from_shape: [from_seq_length, from_width]
        :param to_shape: [to_seq_length, to_width]
        :param num_attention_heads: int. Number of attention heads.
        :param size_per_head: int. Size of each attention head.
        :param query_act: (optional) Activation function for the query transform.
        :param key_act: (optional) Activation function for the key transform.
        :param value_act: (optional) Activation function for the value transform.
        :param attention_probs_dropout_prob: (optional) float. Dropout probability of the attention probabilities.
        :param do_return_2d_tensor: bool. If True, the output will be of shape [batch_size * from_seq_length,
         num_attention_heads * size_per_head]. If False, the output will be of shape [batch_size, from_seq_length,
         num_attention_heads * size_per_head].
        """
        super().__init__(*args, **kwargs)
        self.from_shape = from_shape
        self.to_shape = to_shape
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.do_return_2d_tensor = do_return_2d_tensor

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`

        # `query_layer` = [B*F, N*H]
        self.query_layer = \
            tf.keras.layers.Dense(
                self.num_attention_heads * self.size_per_head, query_act,
                name="query",
                kernel_initializer=ckpt_initializer(ckpt, '%s/query/kernel' % ckpt_prefix,
                                                    'glorot_uniform'),
                bias_initializer=ckpt_initializer(ckpt, '%s/query/bias' % ckpt_prefix, 'zeros')
            )
        # `key_layer` = [B*T, N*H]
        self.key_layer = \
            tf.keras.layers.Dense(
                self.num_attention_heads * self.size_per_head, key_act,
                name="key",
                kernel_initializer=ckpt_initializer(ckpt, '%s/key/kernel' % ckpt_prefix,
                                                    'glorot_uniform'),
                bias_initializer=ckpt_initializer(ckpt, '%s/key/bias' % ckpt_prefix, 'zeros')
            )
        # `value_layer` = [B*T, N*H]
        self.value_layer = \
            tf.keras.layers.Dense(
                self.num_attention_heads * self.size_per_head, value_act,
                name="value",
                kernel_initializer=ckpt_initializer(ckpt, '%s/value/kernel' % ckpt_prefix,
                                                    'glorot_uniform'),
                bias_initializer=ckpt_initializer(ckpt, '%s/value/bias' % ckpt_prefix, 'zeros')
            )

        self.attention_probs_dropout_layer = tf.keras.layers.Dropout(attention_probs_dropout_prob)

    def call(self, inputs, mask=None, **kwargs):
        from_inputs = inputs
        to_inputs = inputs
        attention_mask = self.prepare_mask(mask)
        if isinstance(inputs, Sequence):
            from_inputs = inputs[0]
            to_inputs = inputs[1]

        from_seq_length = self.from_shape[0]
        to_seq_length = self.to_shape[0]

        from_tensor_2d = tf.reshape(from_inputs, [-1, self.from_shape[1]], name="from_reshape")
        to_tensor_2d = tf.reshape(to_inputs, [-1, self.to_shape[1]], name="to_reshape")

        # `query_layer` = [B, N, F, H]
        query_layer = self.query_layer(from_tensor_2d, **kwargs)
        query_layer = self.transpose_for_scores(query_layer, self.num_attention_heads,
                                                from_seq_length, self.size_per_head, name="query_transpose")

        # `key_layer` = [B, N, T, H]
        key_layer = self.key_layer(to_tensor_2d, **kwargs)
        key_layer = self.transpose_for_scores(key_layer, self.num_attention_heads,
                                              to_seq_length, self.size_per_head, name="key_transpose")

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # `attention_scores` = [B, N, F, T]
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(self.size_per_head)))

        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_probs_dropout_layer(attention_probs)

        # `value_layer` = [B, N, T, H]
        value_layer = self.value_layer(to_tensor_2d, **kwargs)
        value_layer = self.transpose_for_scores(value_layer, self.num_attention_heads,
                                                to_seq_length, self.size_per_head, name="value_transpose")

        # `context_layer` = [B, N, F, H]
        context_layer = tf.matmul(attention_probs, value_layer)

        # `context_layer` = [B, F, N, H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

        if self.do_return_2d_tensor:
            # `context_layer` = [B*F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [-1, self.num_attention_heads * self.size_per_head], name="r2d")
        else:
            # `context_layer` = [B, F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [-1, from_seq_length, self.num_attention_heads * self.size_per_head], name="r3d")

        return context_layer

    def prepare_mask(self, mask):
        if mask is None:
            return None
        from_mask = mask
        to_mask = mask
        if isinstance(mask, Sequence):
            from_mask = mask[0]
            to_mask = mask[1]
        from_seq_length = from_mask.shape[1]
        to_seq_length = to_mask.shape[1]
        to_mask = tf.cast(tf.reshape(to_mask, [-1, 1, to_seq_length]), tf.float32)
        # We don't assume that `from_tensor` is a mask (although it could be). We
        # don't actually care if we attend *from* padding tokens (only *to* padding)
        # tokens so we create a tensor of all ones.
        #
        return tf.tile(to_mask, [1, from_seq_length, 1])
