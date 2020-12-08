from typing import Tuple

import numpy as np
import torch
import tqdm
from torch.utils import data
import pickle


class VocabEntry(object):
    def __init__(self):
        self.word2id = dict()
        self.unk_id = 3
        self.word2id['<pad>'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = 3

        self.id2word = {self.word2id[w]: w for w in self.word2id}
        # self.id2word = {v: k for k, v in self.word2id.iteritems()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word) -> object:
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]


def readFile(path: str) -> list:
    with open(path) as file:
        fileContent = file.readlines()
    fileContent = [x.strip() for x in fileContent]
    # fileContent = ' '.join(fileContent)
    return fileContent


def construct_vocab(file):
    vocab = VocabEntry()
    for sent in file:
        for word in sent.split():
            vocab.add(word)
    return vocab


def make_alignment(srcs, tgts, aligns):
    alignment = []
    for src, tgt, align in tqdm.tqdm(zip(srcs, tgts, aligns)):
        ans = np.zeros([len(src.split()), len(tgt.split())])
        for a in align.split():
            x, y = a.split("-")
            x = int(x)
            y = int(y)
            ans[x][y] = 1
        alignment.append(ans)
    return alignment


def process_tags(tag_list):
    outputs = []
    for tags in tag_list:
        outputs.append([1 if tag == "OK" else 0 for tag in tags.split()[1::2]])
    return outputs


class trainDataset(data.Dataset):

    def __init__(self):
        super(trainDataset, self).__init__()
        self.train_mt = readFile("data/train/train.mt")
        self.train_src = readFile("data/train/train.src")
        self.train_src_alignments = readFile("data/train/train.src-mt.alignments")
        self.train_tags = process_tags(readFile("data/train/train.tags"))
        self.source_vocab = construct_vocab(self.train_src)
        self.target_vocab = construct_vocab(self.train_mt)
        self.train_src_alignments = make_alignment(self.train_src, self.train_mt, self.train_src_alignments)
        with open("src_vocab", "wb") as file:
            pickle.dump(self.source_vocab, file)
        with open("tgt_vocab", "wb") as file:
            pickle.dump(self.target_vocab, file)

    def __len__(self) -> int:
        return 10000

    def __getitem__(self, item):
        src = [self.source_vocab[word] for word in self.train_src[item].split()]
        tgt = [self.target_vocab[word] for word in self.train_mt[item].split()]
        align = self.train_src_alignments[item]
        return {
            "source": torch.tensor(src),
            "target": torch.tensor(tgt),
            "alignment": torch.from_numpy(align),
            "predictions": torch.tensor(self.train_tags[item])
        }

    def vocab_size(self) -> Tuple[int, int]:
        return len(self.source_vocab), len(self.target_vocab)


class testDataset(data.Dataset):

    def __init__(self):
        super(testDataset, self).__init__()
        self.train_mt = readFile("data/test/test.mt")
        self.train_src = readFile("data/test/test.src")
        self.train_src_alignments = readFile("data/test/test.src-mt.alignments")
        self.train_tags = process_tags(readFile("data/test/test.tags"))
        with open("src_vocab", "rb") as file:
            self.source_vocab = pickle.load(file)
        with open("tgt_vocab", "rb") as file:
            self.target_vocab = pickle.load(file)

        self.train_src_alignments = make_alignment(self.train_src, self.train_mt, self.train_src_alignments)

    def __len__(self) -> int:
        return 10000

    def __getitem__(self, item):
        src = [self.source_vocab[word] for word in self.train_src[item].split()]
        tgt = [self.target_vocab[word] for word in self.train_mt[item].split()]
        align = self.train_src_alignments[item]
        return {
            "source": torch.tensor(src),
            "target": torch.tensor(tgt),
            "alignment": torch.from_numpy(align),
            "predictions": torch.tensor(self.train_tags[item])
        }

    def vocab_size(self) -> Tuple[int, int]:
        return len(self.source_vocab), len(self.target_vocab)


if __name__ == "__main__":
    trainDataset()
