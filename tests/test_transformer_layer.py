import unittest

import numpy as np
import tensorflow as tf
from tensorflow import keras

from bert.layers.transformer_layer import TransformerLayer


class TestTransformerLayer(unittest.TestCase):
    def test_layer(self):
        train_examples = np.random.randn(1024, 128, 32)
        train_labels = np.random.randint(2, size=(1024, 1))
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels)).batch(32)

        inputs = keras.Input(shape=(128, 32,))
        # [32, 128, 32]
        x = TransformerLayer([128, 32], hidden_size=32, num_attention_heads=1, intermediate_size=64)(inputs)
        # x = TransformerSingleLayer([32, 128, 16], attention_head_size=16, hidden_size=16, num_attention_heads=1,
        #                            intermediate_size=64)(inputs)

        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(2, activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.summary()

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=keras.optimizers.RMSprop(),
                      metrics=['accuracy'])
        history = model.fit(train_dataset, epochs=5)
        print(history)

    def test_mask_layer(self):
        data_nums = 1024
        seq_len = 128

        train_examples = np.random.randint(1000, size=(data_nums, seq_len)) * (
                np.arange(seq_len) < np.random.randint(seq_len, size=(data_nums, 1)))
        train_labels = np.random.randint(2, size=(data_nums, 1))
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels)).batch(32)

        inputs = keras.Input(shape=128)
        x = keras.layers.Embedding(1000, 32, mask_zero=True)(inputs)
        # [32, 128, 32]
        x = TransformerLayer([128, 32], hidden_size=32, num_attention_heads=1, intermediate_size=64, name="test_transformer_layer")(x)
        # x = TransformerSingleLayer([32, 128, 16], attention_head_size=16, hidden_size=16, num_attention_heads=1,
        #                            intermediate_size=64)(inputs)

        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(2, activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.summary()

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=keras.optimizers.RMSprop(),
                      metrics=['accuracy'])
        history = model.fit(train_dataset, epochs=2)

        model.save_weights('/tmp/test_layer_transformer.h5')

        layer = model.get_layer("test_transformer_layer")
        qlayer = layer.layers[0].attention_layer.query_layer
        print(type(qlayer), qlayer)
        print(qlayer.kernel)

    def test_load_model(self):

        inputs = keras.Input(shape=128)
        x = keras.layers.Embedding(1000, 32, mask_zero=True)(inputs)
        # [32, 128, 32]
        x = TransformerLayer([128, 32], hidden_size=32, num_attention_heads=1, intermediate_size=64, name="test_transformer_layer")(x)
        # x = TransformerSingleLayer([32, 128, 16], attention_head_size=16, hidden_size=16, num_attention_heads=1,
        #                            intermediate_size=64)(inputs)

        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(2, activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.summary()

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=keras.optimizers.RMSprop(),
                      metrics=['accuracy'])
        model.load_weights('/tmp/test_layer_transformer.h5')
        model.summary()
        layer = model.get_layer("test_transformer_layer")
        qlayer = layer.layers[0].attention_layer.query_layer
        print(type(qlayer), qlayer)
        print(qlayer.kernel)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    unittest.main()
