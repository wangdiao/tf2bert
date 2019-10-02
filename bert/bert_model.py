import copy

import tensorflow as tf

from bert import ckpt_initializer
from bert.bert_config import BertConfig
from bert.layers.embedding_postprocessor_layer import EmbeddingPostprocessorLayer
from bert.layers.transformer_layer import TransformerLayer


class BertModel(tf.keras.Model):
    def __init__(self,
                 config: BertConfig,
                 ckpt=None,
                 sequence_output=False,
                 mask_zero=True, **kwargs):
        dtype = kwargs.pop('dtype', tf.float32)
        super().__init__(dtype=dtype, **kwargs)
        self.sequence_output = sequence_output
        self.mask_zero = mask_zero
        config = copy.deepcopy(config)
        self.ckpt = ckpt
        if self.trainable:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
        self.hidden_size = config.hidden_size
        self.type_vocab_size = config.type_vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.vocab_size = config.vocab_size
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.num_hidden_layers = config.num_hidden_layers

    def build(self, input_shape):
        self.seq_length = input_shape[-1]
        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size, self.hidden_size, mask_zero=self.mask_zero,
            embeddings_initializer=ckpt_initializer(self.ckpt, 'bert/embeddings/word_embeddings'))
        embedding_input_shape = [self.seq_length, self.hidden_size]
        self.embedding_postprocessor = \
            EmbeddingPostprocessorLayer(
                embedding_input_shape,
                use_token_type=False,
                token_type_vocab_size=self.type_vocab_size, use_position_embeddings=True,
                max_position_embeddings=self.max_position_embeddings,
                token_type_embeddings_initializer=ckpt_initializer(self.ckpt, 'bert/embeddings/token_type_embeddings',
                                                                   'glorot_uniform'),
                position_embeddings_initializer=ckpt_initializer(self.ckpt, 'bert/embeddings/position_embeddings',
                                                                 'glorot_uniform'),
                dropout_prob=self.hidden_dropout_prob,
                layer_norm_beta_initializer=ckpt_initializer(self.ckpt, 'bert/embeddings/LayerNorm/beta', 'zeros'),
                layer_norm_gamma_initializer=ckpt_initializer(self.ckpt, 'bert/embeddings/LayerNorm/gamma', 'ones')
            )
        self.all_encoder_layers = \
            TransformerLayer(embedding_input_shape, hidden_size=self.hidden_size,
                             num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads,
                             intermediate_size=self.intermediate_size,
                             hidden_dropout_prob=self.hidden_dropout_prob,
                             attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                             do_return_all_layers=True, name='encoder', ckpt=self.ckpt)
        self.pooled_layer = \
            tf.keras.layers.Dense(
                self.hidden_size, activation=tf.tanh,
                kernel_initializer=ckpt_initializer(self.ckpt, 'bert/pooler/dense/kernel', 'glorot_uniform'),
                bias_initializer=ckpt_initializer(self.ckpt, 'bert/pooler/dense/bias', 'zeros'),
            )
        self.built = True

    def call(self, inputs):
        input_tensor = inputs
        embedding_output = self.embedding(input_tensor)
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

    def compute_output_shape(self, input_shape):
        if self.sequence_output:
            return input_shape + [self.hidden_size]
        else:
            return input_shape[:-1] + [self.hidden_size]
