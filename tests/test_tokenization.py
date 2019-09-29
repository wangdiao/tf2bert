import unittest

from bert import tokenization


class TestTokenization(unittest.TestCase):

    def test_sentence(self):
        vocab_path = '/home/wangdiao/project/bert/chinese_L-12_H-768_A-12/vocab.txt'
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=True)
        tokens = tokenizer.tokenize("为研究数学基础而产生的集合论和数理逻辑等也开始hello HAHA发展")
        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        print(tokens)
        print(input_ids)


if __name__ == '__main__':
    unittest.main()
