#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__author__ = "theeluwin"
__email__ = "theeluwin@gmail.com"

import codecs
import pycrfsuite
import sklearn

from itertools import chain
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer


def corpus2sent(path):
    # "당/ko 신/ko 의/ko 하/en 트/en 에/ko 니/ja 코/ja 니/ja 코/ja 니/ja"와 같은 형식이면 됩니다.
    corpus = codecs.open(path, encoding='utf-8').read()
    raws = corpus.split('\n')
    sentences = []
    for raw in raws:
        tokens = raw.split(' ')
        sentence = []
        for token in tokens:
            try:
                word, tag = token.split('/')
                if word and tag:
                    sentence.append([word, tag])
            except:
                pass
        sentences.append(sentence)
    return sentences


def index2feature(sent, i, offset):
    word, tag = sent[i + offset]
    if offset < 0:
        sign = ''
    else:
        sign = '+'
    return '{}{}:word={}'.format(sign, offset, word)


def word2features(sent, i):
    L = len(sent)
    word, tag = sent[i]
    features = ['bias']
    features.append(index2feature(sent, i, 0))
    if i > 1:
        features.append(index2feature(sent, i, -2))
    if i > 0:
        features.append(index2feature(sent, i, -1))
    else:
        features.append('bos')
    if i < L - 2:
        features.append(index2feature(sent, i, 2))
    if i < L - 1:
        features.append(index2feature(sent, i, 1))
    else:
        features.append('eos')
    return features


def sent2words(sent):
    return [word for word, tag in sent]


def sent2tags(sent):
    return [tag for word, tag in sent]


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def flush(path, X, Y):
    result = codecs.open(path, 'w', encoding='utf-8')
    for x, y in zip(X, Y):
        result.write(' '.join(['{}/{}'.format(feature[1].split('=')[1], tag) for feature, tag in zip(x, y)]))
        result.write('\n')
    result.close()


def report(test_y, pred_y):
    lb = LabelBinarizer()
    test_y_combined = lb.fit_transform(list(chain.from_iterable(test_y)))
    pred_y_combined = lb.transform(list(chain.from_iterable(pred_y)))
    tagset = sorted(set(lb.classes_))
    class_indices = {cls: idx for idx, cls in enumerate(tagset)}
    print(classification_report(test_y_combined, pred_y_combined, labels=[class_indices[cls] for cls in tagset], target_names=tagset))


def transition(tagger):
    def print_transitions(features):
        for (tag_from, tag_to), weight in features:
            print("%s -> %s | %0.6f" % (tag_from, tag_to, weight))
    info = tagger.info()
    print("Likely transitions:")
    print_transitions(Counter(info.transitions).most_common(15))
    print("")
    print("Unlikely transitions:")
    print_transitions(Counter(info.transitions).most_common()[-15:])


def sample(tagger, sent):
    print("Sentence: ", ' '.join(sent2words(sent)))
    print("Correct:  ", ' '.join(sent2tags(sent)))
    print("Predicted:", ' '.join(tagger.tag(sent2features(sent))))


def main():
    train_sents = corpus2sent('train.txt')
    test_sents = corpus2sent('test.txt')
    train_x = [sent2features(sent) for sent in train_sents]
    train_y = [sent2tags(sent) for sent in train_sents]
    test_x = [sent2features(sent) for sent in test_sents]
    test_y = [sent2tags(sent) for sent in test_sents]
    trainer = pycrfsuite.Trainer()
    for x, y in zip(train_x, train_y):
        trainer.append(x, y)
    trainer.train('locale.crfsuite')
    tagger = pycrfsuite.Tagger()
    tagger.open('locale.crfsuite')
    sample(tagger, test_sents[0])
    print("\n---\n")
    pred_y = [tagger.tag(x) for x in test_x]
    flush('pred.txt', test_x, pred_y)
    report(test_y, pred_y)
    print("---\n")
    transition(tagger)
    print("\n---\n")


if __name__ == '__main__':
    main()
