import tensorflow as tf

from bert.bert_model import BertModel


class ClassifierModel(tf.keras.Model):
    def __init__(self,
                 bert_config,
                 num_labels,
                 bert_ckpt=None, **kwargs):
        dtype = kwargs.pop('dtype', tf.float32)
        super().__init__(dtype=dtype, **kwargs)
        self.num_labels = int(num_labels)
        self.bert_model = BertModel(
            config=bert_config, ckpt=bert_ckpt, trainable=False)
        self.output_layer = tf.keras.layers.Dense(
            self.num_labels, activation=tf.nn.softmax)

    def call(self, inputs, training=None):
        bert_output = self.bert_model(inputs, training=training)
        return self.output_layer(bert_output)
