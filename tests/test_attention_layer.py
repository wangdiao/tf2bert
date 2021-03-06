import unittest

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tf2bert.layers.attention_layer import AttentionLayer


class TestAttentionLayer(unittest.TestCase):
    def test_layer(self):
        train_examples = np.random.randn(1024, 128 * 16)
        train_labels = np.random.randint(2, size=(1024, 1))
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels)).batch(32).shuffle(100)

        inputs = keras.Input(shape=(128 * 16))
        # [32, 128, 16], [32, 128, 16],
        x = AttentionLayer([128, 16], [128, 16], size_per_head=32, do_return_2d_tensor=False)(inputs)
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(2, activation=tf.nn.softmax)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.summary()

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=keras.optimizers.RMSprop(),
                      metrics=['accuracy'])
        history = model.fit(train_dataset, epochs=1)
        print(history)

    def test_mask_layer(self):
        data_nums = 1024
        seq_len = 128

        train_examples = np.random.randint(1000, size=(data_nums, seq_len)) * (
                np.arange(seq_len) < np.random.randint(seq_len, size=(data_nums, 1)))
        train_labels = np.random.randint(2, size=(data_nums, 1))
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels)).batch(32)

        inputs = keras.Input(shape=128)
        x = keras.layers.Embedding(1000, 16, mask_zero=True)(inputs)
        x = AttentionLayer([128, 16], [128, 16], size_per_head=32, do_return_2d_tensor=False)([x, x])
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(2, activation=tf.nn.softmax)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.summary()

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=keras.optimizers.RMSprop(),
                      metrics=['accuracy'])
        history = model.fit(train_dataset, epochs=1)
        print(history)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    unittest.main()
