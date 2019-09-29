import unittest

import tensorflow as tf


class TestBert(unittest.TestCase):
    def test_variable(self):
        for item in tf.train.list_variables("/home/wangdiao/project/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"):
            print(*item)


if __name__ == '__main__':
    unittest.main()
