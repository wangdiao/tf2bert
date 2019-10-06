import tensorflow as tf

from . import tokenization


def ckpt_initializer(ckpt, name, default='uniform'):
    if ckpt:
        def initializer(shape, dtype):
            return tf.train.load_variable(ckpt, name)

        return initializer
    else:
        return default
