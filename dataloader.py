import gzip
import json
import os
import string
import sys
import io
import re
import random
import csv
import numpy as np
import torch

csv.field_size_limit(sys.maxsize)

from local_models.NLI_config import NLI_LABEL_STR2NUM

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", ' \'s', string)
    string = re.sub(r"\'ve", ' \'ve', string)
    string = re.sub(r"n\'t", ' n\'t', string)
    string = re.sub(r"\'re", ' \'re', string)
    string = re.sub(r"\'d", ' \'d', string)
    string = re.sub(r"\'ll", ' \'ll', string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def read_corpus(path, csvf=False , clean=True, MR=True, encoding='utf8', shuffle=False, lower=True):
    data = []
    labels = []
    if not csvf:
        with open(path, encoding=encoding) as fin:
            for line in fin:
                if MR:
                    label, sep, text = line.partition(' ')
                    label = int(label)
                else:
                    label, sep, text = line.partition(',')
                    label = int(label) - 1
                if clean:
                    text = clean_str(text.strip()) if clean else text.strip()
                if lower:
                    text = text.lower()
                labels.append(label)
                data.append(text.split())
    else:
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for line in reader:
                text = line[0]
                label = int(line[1])
                if clean:
                    text = clean_str(text.strip()) if clean else text.strip()
                if lower:
                    text = text.lower()
                labels.append(label)
                data.append(text.split())

    if shuffle:
        perm = list(range(len(data)))
        random.shuffle(perm)
        data = [data[i] for i in perm]
        labels = [labels[i] for i in perm]

    return data, labels

# def read_MR(path, seed=1234):
#     file_path = os.path.join(path, "rt-polarity.all")
#     data, labels = read_corpus(file_path, encoding='latin-1')
#     random.seed(seed)
#     perm = list(range(len(data)))
#     random.shuffle(perm)
#     data = [ data[i] for i in perm ]
#     labels = [ labels[i] for i in perm ]
#     return data, labels
#
# def read_SUBJ(path, seed=1234):
#     file_path = os.path.join(path, "subj.all")
#     data, labels = read_corpus(file_path, encoding='latin-1')
#     random.seed(seed)
#     perm = list(range(len(data)))
#     random.shuffle(perm)
#     data = [ data[i] for i in perm ]
#     labels = [ labels[i] for i in perm ]
#     return data, labels
#
# def read_CR(path, seed=1234):
#     file_path = os.path.join(path, "custrev.all")
#     data, labels = read_corpus(file_path)
#     random.seed(seed)
#     perm = list(range(len(data)))
#     random.shuffle(perm)
#     data = [ data[i] for i in perm ]
#     labels = [ labels[i] for i in perm ]
#     return data, labels
#
# def read_MPQA(path, seed=1234):
#     file_path = os.path.join(path, "mpqa.all")
#     data, labels = read_corpus(file_path)
#     random.seed(seed)
#     perm = list(range(len(data)))
#     random.shuffle(perm)
#     data = [ data[i] for i in perm ]
#     labels = [ labels[i] for i in perm ]
#     return data, labels
#
# def read_TREC(path, seed=1234):
#     train_path = os.path.join(path, "TREC.train.all")
#     test_path = os.path.join(path, "TREC.test.all")
#     train_x, train_y = read_corpus(train_path, TREC=True, encoding='latin-1')
#     test_x, test_y = read_corpus(test_path, TREC=True, encoding='latin-1')
#     random.seed(seed)
#     perm = list(range(len(train_x)))
#     random.shuffle(perm)
#     train_x = [ train_x[i] for i in perm ]
#     train_y = [ train_y[i] for i in perm ]
#     return train_x, train_y, test_x, test_y
#
# def read_SST(path, seed=1234):
#     train_path = os.path.join(path, "stsa.binary.phrases.train")
#     valid_path = os.path.join(path, "stsa.binary.dev")
#     test_path = os.path.join(path, "stsa.binary.test")
#     train_x, train_y = read_corpus(train_path, False)
#     valid_x, valid_y = read_corpus(valid_path, False)
#     test_x, test_y = read_corpus(test_path, False)
#     random.seed(seed)
#     perm = list(range(len(train_x)))
#     random.shuffle(perm)
#     train_x = [ train_x[i] for i in perm ]
#     train_y = [ train_y[i] for i in perm ]
#     return train_x, train_y, valid_x, valid_y, test_x, test_y

def cv_split(data, labels, nfold, test_id):
    assert (nfold > 1) and (test_id >= 0) and (test_id < nfold)
    lst_x = [ x for i, x in enumerate(data) if i%nfold != test_id ]
    lst_y = [ y for i, y in enumerate(labels) if i%nfold != test_id ]
    test_x = [ x for i, x in enumerate(data) if i%nfold == test_id ]
    test_y = [ y for i, y in enumerate(labels) if i%nfold == test_id ]
    perm = list(range(len(lst_x)))
    random.shuffle(perm)
    M = int(len(lst_x)*0.9)
    train_x = [ lst_x[i] for i in perm[:M] ]
    train_y = [ lst_y[i] for i in perm[:M] ]
    valid_x = [ lst_x[i] for i in perm[M:] ]
    valid_y = [ lst_y[i] for i in perm[M:] ]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def cv_split2(data, labels, nfold, valid_id):
    assert (nfold > 1) and (valid_id >= 0) and (valid_id < nfold)
    train_x = [ x for i, x in enumerate(data) if i%nfold != valid_id ]
    train_y = [ y for i, y in enumerate(labels) if i%nfold != valid_id ]
    valid_x = [ x for i, x in enumerate(data) if i%nfold == valid_id ]
    valid_y = [ y for i, y in enumerate(labels) if i%nfold == valid_id ]
    return train_x, train_y, valid_x, valid_y

def pad(sequences, pad_token='<pad>', pad_left=True):
    ''' input sequences is a list of text sequence [[str]]
        pad each text sequence to the length of the longest
    '''
    max_len = max(5,max(len(seq) for seq in sequences))
    if pad_left:
        return [ [pad_token]*(max_len-len(seq)) + seq for seq in sequences ]
    return [ seq + [pad_token]*(max_len-len(seq)) for seq in sequences ]


def create_one_batch(x, y, map2id, oov='<oov>'):
    oov_id = map2id[oov]
    x = pad(x)
    length = len(x[0])
    batch_size = len(x)
    x = [ map2id.get(w, oov_id) for seq in x for w in seq ]
    x = torch.LongTensor(x)
    assert x.size(0) == length*batch_size
    return x.view(batch_size, length).t().contiguous().cuda(), torch.LongTensor(y).cuda()


def create_one_batch_x(x, map2id, oov='<oov>'):
    oov_id = map2id[oov]
    x = pad(x)
    length = len(x[0])
    batch_size = len(x)
    x = [ map2id.get(w, oov_id) for seq in x for w in seq ]
    x = torch.LongTensor(x)
    assert x.size(0) == length*batch_size
    return x.view(batch_size, length).t().contiguous().cuda()


# shuffle training examples and create mini-batches
def create_batches(x, y, batch_size, map2id, perm=None, sort=False):

    lst = perm or range(len(x))

    # sort sequences based on their length; necessary for SST
    if sort:
        lst = sorted(lst, key=lambda i: len(x[i]))

    x = [ x[i] for i in lst ]
    y = [ y[i] for i in lst ]

    sum_len = 0.
    for ii in x:
        sum_len += len(ii)
    batches_x = [ ]
    batches_y = [ ]
    size = batch_size
    nbatch = (len(x)-1) // size + 1
    for i in range(nbatch):
        bx, by = create_one_batch(x[i*size:(i+1)*size], y[i*size:(i+1)*size], map2id)
        batches_x.append(bx)
        batches_y.append(by)

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_x = [ batches_x[i] for i in perm ]
        batches_y = [ batches_y[i] for i in perm ]

    sys.stdout.write("{} batches, avg sent len: {:.1f}\n".format(
        nbatch, sum_len/len(x)
    ))

    return batches_x, batches_y


# shuffle training examples and create mini-batches
def create_batches_x(x, batch_size, map2id, perm=None, sort=False):

    lst = perm or range(len(x))

    # sort sequences based on their length; necessary for SST
    if sort:
        lst = sorted(lst, key=lambda i: len(x[i]))

    x = [ x[i] for i in lst ]

    sum_len = 0.0
    batches_x = [ ]
    size = batch_size
    nbatch = (len(x)-1) // size + 1
    for i in range(nbatch):
        bx = create_one_batch_x(x[i*size:(i+1)*size], map2id)
        sum_len += len(bx)
        batches_x.append(bx)

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_x = [ batches_x[i] for i in perm ]

    # sys.stdout.write("{} batches, avg len: {:.1f}\n".format(
    #     nbatch, sum_len/nbatch
    # ))

    return batches_x


def load_embedding_npz(path):
    data = np.load(path)
    return [ w.decode('utf8') for w in data['words'] ], data['vals']

def load_embedding_txt(path):
    file_open = gzip.open if path.endswith(".gz") else open
    words = [ ]
    vals = [ ]
    with file_open(path, encoding='utf-8') as fin:
        fin.readline()
        for line in fin:
            line = line.rstrip()
            if line:
                parts = line.split(' ')
                words.append(parts[0])
                vals += [ float(x) for x in parts[1:] ]
    return words, np.asarray(vals).reshape(len(words),-1)

def load_embedding(path):
    if path.endswith(".npz"):
        return load_embedding_npz(path)
    else:
        return load_embedding_txt(path)


def read_data_nli(filepath, data_size=None, lowercase=False, ignore_punctuation=False, stopwords=[]):
    """
    Read the premises, hypotheses and labels from some NLI dataset's
    file and return them in a dictionary. The file should be in the same
    form as SNLI's .txt files.

    Args:
        filepath: The path to a file containing some premises, hypotheses
            and labels that must be read. The file should be formatted in
            the same way as the SNLI (and MultiNLI) dataset.

    Returns:
        A dictionary containing three lists, one for the premises, one for
        the hypotheses, and one for the labels in the input data.
    """


    labeldict = NLI_LABEL_STR2NUM

    with open(filepath, 'r', encoding='utf8') as input_data:
        premises, hypotheses, labels = [], [], []

        # Translation tables to remove punctuation from strings.
        punct_table = str.maketrans({key: ' '
                                     for key in string.punctuation})

        for idx, line in enumerate(input_data):
            if data_size and idx >= data_size:
                break

            line = line.strip().split('\t')

            # Ignore sentences that have no gold label.
            if line[0] == '-':
                continue

            premise = line[1]
            hypothesis = line[2]

            if lowercase:
                premise = premise.lower()
                hypothesis = hypothesis.lower()

            if ignore_punctuation:
                premise = premise.translate(punct_table)
                hypothesis = hypothesis.translate(punct_table)

            # Each premise and hypothesis is split into a list of words.
            premises.append([w for w in premise.rstrip().split()
                             if w not in stopwords])
            hypotheses.append([w for w in hypothesis.rstrip().split()
                               if w not in stopwords])
            labels.append(labeldict[line[0]])

        return {"premises": premises,
                "hypotheses": hypotheses,
                "labels": labels}


def read_nli_target(filepath):
    target_labels = []
    with open(filepath, 'r') as f:
        for target_label_str in f:
            target_labels.append(NLI_LABEL_STR2NUM[target_label_str.strip()])

    return target_labels


def read_classification_target(filepath):
    target_labels = []
    with open(filepath, 'r') as f:
        for target_label_str in f:
            target_labels.append(int(target_label_str.strip()))

    return target_labels




def read_orig_imdb(imdb_folder_dir, filetype):
    """
    filetype: 'train' or 'test'
    """

    # 1 means positive，0 means negative
    all_labels = []
    for _ in range(12500):
        all_labels.append(1)
    for _ in range(12500):
        all_labels.append(0)

    all_texts = []
    file_list = []
    pos_path = os.path.join(imdb_folder_dir, filetype, 'pos/')
    for file in os.listdir(pos_path):
        file_list.append(pos_path + file)
    neg_path = os.path.join(imdb_folder_dir, filetype, 'neg/')
    for file in os.listdir(neg_path):
        file_list.append(neg_path + file)
    for file_name in file_list:
        with open(file_name, 'r', encoding='utf-8') as f:
            all_texts.append(" ".join(f.readlines()))

    return all_texts, all_labels

def read_orig_snli(snli_folder_dir, filetype):
    """
    filetype: 'train', 'test', 'dev'
    """
    file_name = f'snli_1.0_{filetype}.jsonl'
    file_path = os.path.join(snli_folder_dir, file_name)

    data_list = []

    for i, line in enumerate(open(file_path)):
        data = json.loads(line)
        label = data['gold_label']
        if label == '-':
            continue

        # s1 = ' '.join(extract_tokens_from_binary_parse(
        #     data['sentence1_binary_parse']))
        # s2 = ' '.join(extract_tokens_from_binary_parse(
        #     data['sentence2_binary_parse']))

        s1 = data['sentence1']
        s2 = data['sentence2']


        # yield (label, s1, s2)
        data_list.append((label, s1, s2))

    return data_list

def read_orig_mnli(mnli_folder_dir, filetype):
    """
    filetype: 'train', 'dev_matched', 'dev_mismatched', 'test_matched', 'test_mismatched'
    """
    if filetype[:4] == 'test':
        file_name = f'multinli_0.9_{filetype}_unlabeled.jsonl'
    else:
        file_name = f'multinli_1.0_{filetype}.jsonl'

    file_path = os.path.join(mnli_folder_dir, file_name)
    data_list = []

    for i, line in enumerate(open(file_path)):
        data = json.loads(line)
        label = data['gold_label']
        if label == '-':
            continue

        s1 = data['sentence1']
        s2 = data['sentence2']
        pairID = data['pairID']

        # yield (label, s1, s2)
        data_list.append((label, s1, s2, pairID))

    return data_list

def read_orig_agnews(agnews_folder_dir, filetype):
    """
    filetype: 'train', 'test'
    """
    text_list = []
    label_idx_list = []
    
    file_dir = os.path.join(agnews_folder_dir, filetype + '.csv')
    with open(file_dir, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='"')
        next(csv_reader)
        for line in csv_reader:
            content = line[1] + ". " + line[2]
            text_list.append(content)
            label_idx_list.append(int(line[0]) - 1)

    return text_list, label_idx_list

def read_orig_mr(folder_dir):
    #download from http://www.cs.cornell.edu/people/pabo/movie-review-data/
    text_list = []
    label_list = []
    for s in ['pos', 'neg']:
        file_path = os.path.join(folder_dir, "rt-polarity." + s)
        label_idx = 1 if s == 'pos' else 0
        with open(file_path, encoding='latin-1') as f:
            for line in f:
                text_list.append(line)
                label_list.append(label_idx)

    return text_list, label_list

def read_orig_yahoo(foler_dir, filtype):
    """
    filetype: 'train' or 'test'
    """
    file_path = os.path.join(foler_dir, filtype + '.csv')

    label_list = []
    text_list = []

    with open(file_path, 'r') as f:
        csv_f = csv.reader(f)

        for line in csv_f:
            _label = int(line[0]) - 1
            _text = line[1] + " " + line[2] + " " +  line[3]

            label_list.append(_label)
            text_list.append(_text)

    return text_list, label_list






if __name__ == '__main__':
    # d = read_orig_yahoo('/home/workspace/big_data/datasets/yahoo_answers', 'test')
    # print(d)
    print()