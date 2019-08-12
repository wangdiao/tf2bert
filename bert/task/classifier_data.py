if __name__ == '__main__':
    labels = set()
    with open('/home/wangdiao/data/cnews.val.txt', encoding='UTF-8') as f:
        for line in f:
            if not line.strip():
                continue
            [label, text] = line.split('\t', 2)
            labels.add(label)
    print(labels)
