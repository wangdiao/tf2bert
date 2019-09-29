import os
import re
import shutil
import unittest


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
            for root, dirs, files in os.walk(path):
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

    def test_layer(self):
        pass


if __name__ == '__main__':
    unittest.main()
