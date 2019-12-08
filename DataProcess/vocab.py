# 获取词典

from Public.path import path_vocab

unk_flag = '[UNK]'
pad_flag = '[PAD]'
cls_flag = '[CLS]'
sep_flag = '[SEP]'


# 获取 word to index 词典
def get_w2i(vocab_path = path_vocab):
    w2i = {}
    with open(vocab_path, 'r') as f:
        while True:
            text = f.readline()
            if not text:
                break
            text = text.strip()
            if text and len(text) > 0:
                w2i[text] = len(w2i) + 1
    return w2i


# 获取 tag to index 词典
def get_tag2index():
    return {"O": 0,
            "B-PER": 1, "I-PER": 2,
            "B-LOC": 3, "I-LOC": 4,
            "B-ORG": 5, "I-ORG": 6
            }


if __name__ == '__main__':
    get_w2i()






















