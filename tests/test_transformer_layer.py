import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python import keras

from bert.layers.transformer_layer import TransformerLayer


class TestTransformerLayer(unittest.TestCase):
    def test_layer(self):
        train_examples = np.random.randn(1024, 128, 32)
        train_labels = np.random.randint(2, size=(1024, 1))
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels)).batch(32)

        inputs = keras.Input(shape=(128, 32,))
        x = TransformerLayer([32, 128, 32], hidden_size=32, num_attention_heads=1, intermediate_size=64)(inputs)
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


if __name__ == '__main__':
    unittest.main()
