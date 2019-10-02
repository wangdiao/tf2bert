import os
import re
import shutil
import unittest

import tensorflow as tf

from bert import tokenization
from bert.bert_config import BertConfig
from bert.models.classifier_model import ClassifierModel
from bert.tokenization import convert_to_unicode

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])


class TestClassifier(unittest.TestCase):

    def test_prepare(self):
        path = '/home/wangdiao/data/THUCNews'
        write_path = '/home/wangdiao/data/news'
        if os.path.exists(write_path):
            shutil.rmtree(write_path)
        os.mkdir(write_path)
        i = 0
        with open('/home/wangdiao/data/news/train.txt', 'w', encoding='utf-8') as trainf, \
                open('/home/wangdiao/data/news/vali.txt', 'w', encoding='utf-8') as valif, \
                open('/home/wangdiao/data/news/test.txt', 'w', encoding='utf-8') as testf:
            for root, _, files in os.walk(path):
                if files:
                    for file in files:
                        with open(os.path.join(root, file), encoding='utf-8') as f:
                            s = f.read()
                            s = re.sub(r'\s+', ' ', s, flags=re.MULTILINE)
                        label = root[len(path) + 1:]
                        wf = trainf
                        if i % 20 == 1:
                            wf = valif
                        elif i % 20 == 2:
                            wf = testf
                        print(label, s, sep='\t', file=wf)
                        i += 1

    def test_train(self):
        max_seq_length = 512
        batch_size = 2
        label_list = ['财经', '彩票', '房产', '股票', '家居', '教育',
                      '科技', '社会', '时尚', '时政', '体育', '星座', '游戏', '娱乐']
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i
        dataset = tf.data.TextLineDataset(
            '/home/wangdiao/data/news/train.txt')

        vocab_path = '/home/wangdiao/project/bert/chinese_L-12_H-768_A-12/vocab.txt'
        tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=True)

        def line_feature(line):
            line = convert_to_unicode(line.numpy())
            [lab, sentence] = line.split('\t', 2)
            label_id = label_map[lab]
            tokens = tokenizer.tokenize(sentence)
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[0:(max_seq_length - 2)]
            tokens.insert(0, '[CLS]')
            tokens.append('[SEP]')
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            return input_ids, label_id

        dataset = dataset.map(lambda x: tf.py_function(line_feature, [x], (tf.int32, tf.int32))) \
            .padded_batch(batch_size, padded_shapes=([max_seq_length], []), padding_values=(0, 0),
                          drop_remainder=True).shuffle(100)

        model = ClassifierModel(
            bert_config=BertConfig.from_json_file(
                "/home/wangdiao/project/bert/chinese_L-12_H-768_A-12/bert_config.json"),
            bert_ckpt="/home/wangdiao/project/bert/chinese_L-12_H-768_A-12/bert_model.ckpt",
            num_labels=len(label_list))
        model.compile(optimizer='Nadam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.build(input_shape=(max_seq_length,))
        # print('weights:', len(model.variables))
        # print('trainable weights:', len(model.trainable_variables))
        # print('bert trainable weights:', len(model.output_layer.variables))
        model.summary()
        model.fit(dataset)


if __name__ == '__main__':
    unittest.main()
