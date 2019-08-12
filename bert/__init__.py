import tensorflow as tf


def ckpt_initializer(ckpt, name, default='uniform'):
    if ckpt:
        return lambda: tf.train.load_variable(ckpt, name)
    else:
        return default
