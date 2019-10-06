import tensorflow as tf

from tf2bert import ckpt_initializer
from tf2bert.layers import gelu
from tf2bert.layers.attention_layer import AttentionLayer


class TransformerSingleLayer(tf.keras.layers.Layer):
    def __init__(self,
                 shape,
                 attention_head_size,
                 hidden_size=768,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 intermediate_act_fn=gelu,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 ckpt=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_layer = AttentionLayer(
            shape, shape, num_attention_heads, attention_head_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            do_return_2d_tensor=True, ckpt=ckpt, ckpt_prefix='bert/encoder/%s/attention/self' % self.name)

        self.attention_output_dense = \
            tf.keras.layers.Dense(
                hidden_size,
                kernel_initializer=ckpt_initializer(ckpt, 'bert/encoder/%s/attention/output/dense/kernel' % self.name,
                                                    'glorot_uniform'),
                bias_initializer=ckpt_initializer(ckpt, 'bert/encoder/%s/attention/output/dense/bias' % self.name,
                                                  'zeros'),
            )
        self.attention_output_dropout = tf.keras.layers.Dropout(hidden_dropout_prob)
        self.attention_output_layer_norm = \
            tf.keras.layers.LayerNormalization(
                beta_initializer=ckpt_initializer(ckpt, 'bert/encoder/%s/attention/output/LayerNorm/beta' % self.name,
                                                  'zeros'),
                gamma_initializer=ckpt_initializer(ckpt, 'bert/encoder/%s/attention/output/LayerNorm/gamma' % self.name,
                                                   'ones'),
            )
        # The activation is only applied to the "intermediate" hidden layer.
        self.intermediate_layer = \
            tf.keras.layers.Dense(
                intermediate_size, activation=intermediate_act_fn,
                kernel_initializer=ckpt_initializer(ckpt, 'bert/encoder/%s/intermediate/dense/kernel' % self.name,
                                                    'glorot_uniform'),
                bias_initializer=ckpt_initializer(ckpt, 'bert/encoder/%s/intermediate/dense/bias' % self.name,
                                                  'zeros'),
            )
        self.output_layer = \
            tf.keras.layers.Dense(
                hidden_size,
                kernel_initializer=ckpt_initializer(ckpt, 'bert/encoder/%s/output/dense/kernel' % self.name,
                                                    'glorot_uniform'),
                bias_initializer=ckpt_initializer(ckpt, 'bert/encoder/%s/output/dense/bias' % self.name,
                                                  'zeros'),
            )
        self.output_layer_dropout = tf.keras.layers.Dropout(hidden_dropout_prob)
        self.output_layer_norm = tf.keras.layers.LayerNormalization(
            beta_initializer=ckpt_initializer(ckpt, 'bert/encoder/%s/output/LayerNorm/beta' % self.name, 'zeros'),
            gamma_initializer=ckpt_initializer(ckpt, 'bert/encoder/%s/output/LayerNorm/gamma' % self.name, 'ones')
        )

    def call(self, inputs, mask=None, **kwargs):
        attention_head = self.attention_layer(inputs, mask=mask, **kwargs)
        attention_output = attention_head
        attention_output = self.attention_output_dense(attention_output)
        attention_output = self.attention_output_dropout(attention_output)
        attention_output = self.attention_output_layer_norm(
            tf.add(attention_output, inputs, name="add_attention_output"))
        intermediate_output = self.intermediate_layer(attention_output)
        layer_output = self.output_layer(intermediate_output)
        layer_output = self.output_layer_dropout(layer_output)
        layer_output = self.output_layer_norm(tf.add(layer_output, attention_output, name="add_layer_output"))
        return layer_output


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self,
                 shape,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 intermediate_act_fn=gelu,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 do_return_all_layers=False,
                 ckpt=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = shape
        self.hidden_size = hidden_size
        self.do_return_all_layers = do_return_all_layers

        attention_head_size = int(hidden_size / num_attention_heads)

        self.layers = [TransformerSingleLayer(shape, attention_head_size, hidden_size,
                                              num_attention_heads, intermediate_size, intermediate_act_fn,
                                              hidden_dropout_prob, attention_probs_dropout_prob, name="layer_%d" % i,
                                              ckpt=ckpt)
                       for i in range(num_hidden_layers)]

    def call(self, inputs, mask=None, **kwargs):
        input_tensor = inputs
        shape = input_tensor.shape
        input_width = shape[2]

        # The Transformer performs sum residuals on all layers so the input needs
        # to be the same as the hidden size.
        if input_width != self.hidden_size:
            raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                             (input_width, self.hidden_size))

        # We keep the representation as a 2D tensor to avoid re-shaping it back and
        # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
        # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
        # help the optimizer.
        prev_output = tf.reshape(input_tensor, [-1, input_width])

        all_layer_outputs = []
        for layer in self.layers:
            layer_output = layer(prev_output, mask=mask, **kwargs)
            prev_output = layer_output
            all_layer_outputs.append(layer_output)

        if self.do_return_all_layers:
            final_outputs = []
            for layer_output in all_layer_outputs:
                final_output = tf.reshape(layer_output, [-1, self.shape[0], self.hidden_size])
                final_outputs.append(final_output)
            return final_outputs
        else:
            final_output = tf.reshape(prev_output, [-1, self.shape[0], self.hidden_size])
            return final_output

    def compute_mask(self, inputs, mask=None):
        return mask