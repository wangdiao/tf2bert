import copy

import tensorflow as tf

from bert import ckpt_initializer
from bert.bert_config import BertConfig
from bert.layers.embedding_postprocessor_layer import EmbeddingPostprocessorLayer
from bert.layers.transformer_layer import TransformerLayer


class BertModel(tf.keras.Model):
    def __init__(self,
                 config: BertConfig,
                 input_shape,
                 input_mask=None,
                 ckpt=None,
                 sequence_output=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_output = sequence_output
        config = copy.deepcopy(config)
        if self.trainable:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        batch_size = input_shape[0]
        seq_length = input_shape[1]
        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        self.embedding = tf.keras.layers.Embedding(
            config.vocab_size, config.hidden_size, mask_zero=True,
            embeddings_initializer=ckpt_initializer(ckpt, 'bert/embeddings/word_embeddings'))
        embedding_input_shape = [batch_size, seq_length, config.hidden_size]
        self.embedding_postprocessor = \
            EmbeddingPostprocessorLayer(
                embedding_input_shape,
                use_token_type=False,
                token_type_vocab_size=config.type_vocab_size, use_position_embeddings=True,
                max_position_embeddings=config.max_position_embeddings,
                token_type_embeddings_initializer=ckpt_initializer(ckpt, 'bert/embeddings/token_type_embeddings',
                                                                   'glorot_uniform'),
                position_embeddings_initializer=ckpt_initializer(ckpt, 'bert/embeddings/position_embeddings',
                                                                 'glorot_uniform'),
                dropout_prob=config.hidden_dropout_prob,
                layer_norm_beta_initializer=ckpt_initializer(ckpt, 'bert/embeddings/LayerNorm/beta', 'zeros'),
                layer_norm_gamma_initializer=ckpt_initializer(ckpt, 'bert/embeddings/LayerNorm/gamma', 'ones')
            )
        attention_mask = self.create_attention_mask_from_input_mask(input_shape, input_mask, seq_length)
        self.all_encoder_layers = \
            TransformerLayer(embedding_input_shape, attention_mask=attention_mask, hidden_size=config.hidden_size,
                             num_hidden_layers=config.num_hidden_layers, num_attention_heads=config.num_attention_heads,
                             intermediate_size=config.intermediate_size,
                             hidden_dropout_prob=config.hidden_dropout_prob,
                             attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                             do_return_all_layers=True, name='encoder', ckpt=ckpt)
        self.pooled_layer = \
            tf.keras.layers.Dense(
                config.hidden_size, activation=tf.tanh,
                kernel_initializer=ckpt_initializer(ckpt, 'bert/pooler/dense/kernel', 'glorot_uniform'),
                bias_initializer=ckpt_initializer(ckpt, 'bert/pooler/dense/bias', 'zeros'),
            )

    def call(self, inputs, training=None, mask=None):
        embedding_output = self.embedding(inputs)
        embedding_output = self.embedding_postprocessor(embedding_output)
        sequence_output = self.all_encoder_layers(embedding_output)[-1]
        if self.sequence_output:
            return sequence_output
        # The "pooler" converts the encoded sequence tensor of shape
        # [batch_size, seq_length, hidden_size] to a tensor of shape
        # [batch_size, hidden_size]. This is necessary for segment-level
        # (or segment-pair-level) classification tasks where we need a fixed
        # dimensional representation of the segment.

        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
        pooled_output = self.pooled_layer(first_token_tensor)
        return pooled_output

    @staticmethod
    def create_attention_mask_from_input_mask(from_shape, to_mask, to_seq_length):
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]

        to_mask = tf.cast(
            tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

        # We don't assume that `from_tensor` is a mask (although it could be). We
        # don't actually care if we attend *from* padding tokens (only *to* padding)
        # tokens so we create a tensor of all ones.
        #
        # `broadcast_ones` = [batch_size, from_seq_length, 1]
        broadcast_ones = tf.ones(
            shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

        # Here we broadcast along two dimensions to create the mask.
        mask = broadcast_ones * to_mask
        return mask
