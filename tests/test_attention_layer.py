import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python import keras

from bert.layers.attention_layer import AttentionLayer


class TestAttentionLayer(unittest.TestCase):
    def test_layer(self):
        train_examples = np.random.randn(1024, 128 * 16)
        train_labels = np.random.randint(2, size=(1024, 1))
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels)).batch(32).shuffle(100)

        inputs = keras.Input(shape=(128 * 16,))
        x = AttentionLayer([32, 128, 16], [32, 128, 16], size_per_head=32, do_return_2d_tensor=False)(inputs)
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
    unittest.main()
