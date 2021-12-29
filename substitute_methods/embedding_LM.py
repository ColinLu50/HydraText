import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils import glove_utils

import os
from tensorflow.keras.preprocessing.text import Tokenizer

from utils import my_file
import dataloader




def _read_text_imdb(raw_data_path):
    path = raw_data_path
    """ Returns a list of text documents and a list of their labels
    (pos = +1, neg = 0) """
    pos_list = []
    neg_list = []

    pos_path = path + '/pos'
    neg_path = path + '/neg'
    pos_files = [pos_path + '/' + x for x in os.listdir(pos_path) if x.endswith('.txt')]
    neg_files = [neg_path + '/' + x for x in os.listdir(neg_path) if x.endswith('.txt')]

    pos_list = [open(x, 'r').read().lower() for x in pos_files]
    neg_list = [open(x, 'r').read().lower() for x in neg_files]
    data_list = pos_list + neg_list
    labels_list = [1] * len(pos_list) + [0] * len(neg_list)
    return data_list, labels_list

def build_dataset(dataset, dataset_path):
    if dataset == 'imdb':
        max_vocab_size = 50000
        train_path = dataset_path + '/train'
        train_texts, train_y = _read_text_imdb(train_path)
    if dataset == 'agnews':
        max_vocab_size = None
        train_texts, train_y = dataloader.read_orig_agnews(dataset_path, 'train')
    elif dataset == 'yahoo':
        max_vocab_size = 50000
        train_texts, train_y = dataloader.read_orig_yahoo(dataset_path, 'train')
    elif dataset == 'mr':
        max_vocab_size = 50000
        train_texts, train_y = dataloader.read_orig_mr(dataset_path)
    elif dataset == 'snli':
        from dataloader import read_orig_snli
        max_vocab_size = None
        train_data_list = read_orig_snli(dataset_path, 'train')

        train_texts = []
        for label, s1, s2 in train_data_list:
            train_texts.append(s1)
            train_texts.append(s2)
    elif dataset[:4] == 'mnli':
        from dataloader import read_orig_mnli
        max_vocab_size = None
        train_data_list = read_orig_mnli(dataset_path, 'train')

        train_texts = []
        for label, s1, s2, _ in train_data_list:
            train_texts.append(s1)
            train_texts.append(s2)

    print('tokenizing...')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_texts)

    dict_ = {}
    inv_dict = {}
    full_dict = {}
    inv_full_dict = {}

    if not max_vocab_size:
        max_vocab_size = len(tokenizer.word_index.items())
    else:
        dict_[max_vocab_size] = 'UNK'
        inv_dict['UNK'] = max_vocab_size

    for word, idx in tokenizer.word_index.items():
        if idx < max_vocab_size:
            inv_dict[idx] = word
            dict_[word] = idx
        full_dict[word] = idx
        inv_full_dict[idx] = word



    print('Dataset built !')

    return dict_, inv_dict, full_dict, inv_full_dict, max_vocab_size







def build_dist_mat(output_dir, dataset, dataset_path, COUNTER_FITTED_PATH):

    dict_, inv_dict, full_dict, inv_full_dict, max_vocab_size = build_dataset(dataset, dataset_path)
    dataset_tuple = (dict_, inv_dict, full_dict, inv_full_dict, max_vocab_size)
    datset_save_path = os.path.join(output_dir, f'{dataset}_dataset.pkl')
    my_file.save_pkl(dataset_tuple, datset_save_path)

    # Load the counterfitted-vectors
    glove2 = glove_utils.loadGloveModel(COUNTER_FITTED_PATH)
    # create embeddings matrix for our vocabulary
    embedding_matrix, missed = glove_utils.create_embeddings_matrix(glove2, dict_, full_dict)


    # compute distance
    print('Compute Distance Matrix ...')
    c_ = -2 * np.dot(embedding_matrix.T, embedding_matrix)
    a = np.sum(np.square(embedding_matrix), axis=0).reshape((1, -1))
    b = a.T
    dist = a + b + c_

    save_path1 = os.path.join(output_dir, 'dist_matrix_%s.npy' % (dataset))
    np.save((save_path1), dist)

    print('All done')

def load_generated_files(save_dir, dataset):
    datset_save_path = os.path.join(save_dir, f'{dataset}_dataset.pkl')
    dict_, inv_dict, full_dict, inv_full_dict, max_vocab_size = my_file.load_pkl(datset_save_path)

    npy_path1 = os.path.join(save_dir, 'dist_matrix_%s.npy' % (dataset))
    dist_mat = np.load(npy_path1)
    # Prevent returning 0 as most similar word because it is not part of the dictionary
    dist_mat[0, :] = 100000
    dist_mat[:, 0] = 100000

    return dict_, inv_dict, full_dict, inv_full_dict, max_vocab_size, dist_mat





if __name__ == '__main__':
    save_dir = 'path/to/preprocess_folder/'

    dict_, inv_dict, full_dict, inv_full_dict, max_vocab_size, dist_mat = load_generated_files(save_dir, 'imdb')

    # Try an example
    for w in ['movie', 'happy', 'player']:

        if w not in dict_:
            print(w, 'is not valid')
            continue

        nearest, nearest_dist = glove_utils.pick_most_similar_words(dict_[w], dist_mat, 20, 0.5)

        print('Closest to `%s` are:' % (w))
        for w_id, w_dist in zip(nearest, nearest_dist):
            print(' -- ', inv_dict[w_id], ' ', w_dist)

        print('----')