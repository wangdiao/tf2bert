import tensorflow as tf

from bert import tokenization
from bert.bert_config import BertConfig
from bert.bert_model import BertModel
from bert.tokenization import convert_to_unicode


class ClassifierModel(tf.keras.Model):
    def compute_output_signature(self, input_signature):
        return tf.TensorSpec((None, self.num_labels), tf.float32, name="compute_output_spec")

    def __init__(self,
                 bert_config,
                 num_labels,
                 seq_length,
                 bert_ckpt=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_labels = int(num_labels)
        self.bert_model = BertModel(
            config=bert_config,
            seq_length=seq_length, ckpt=bert_ckpt, trainable=False)
        self.output_layer = tf.keras.layers.Dense(self.num_labels, activation=tf.nn.softmax)

    def call(self, inputs, training=None, mask=None):
        bert_output = self.bert_model(inputs)
        return self.output_layer(bert_output)


if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

    max_seq_length = 512
    batch_size = 2
    label_list = ['财经', '彩票', '房产', '股票', '家居', '教育', '科技', '社会', '时尚', '时政', '体育', '星座', '游戏', '娱乐']
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    dataset = tf.data.TextLineDataset('/home/wangdiao/data/news/strain.txt')

    vocab_path = '/home/wangdiao/project/bert/chinese_L-12_H-768_A-12/vocab.txt'
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=True)


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
        bert_config=BertConfig.from_json_file("/home/wangdiao/project/bert/chinese_L-12_H-768_A-12/bert_config.json"),
        bert_ckpt="/home/wangdiao/project/bert/chinese_L-12_H-768_A-12/bert_model.ckpt",
        num_labels=len(label_list), seq_length=max_seq_length)
    print('weights:', len(model.variables))
    print('trainable weights:', len(model.trainable_variables))
    print('bert trainable weights:', len(model.output_layer.variables))
    model.compile(optimizer='Nadam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.build(input_shape=(None, max_seq_length))
    model.summary()
    model.fit(dataset)
