import os
import pickle
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import criteria
from dataloader import read_data_nli


def com_sim_compute_textfooler(output_dir, embedding_path):
    embeddings = []
    with open(embedding_path, 'r') as ifile:
        for line in ifile:
            embedding = [float(num) for num in line.strip().split()[1:]]
            embeddings.append(embedding)
    embeddings = np.array(embeddings)
    # embeddings = embeddings[:30000]
    print(embeddings.T.shape)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.asarray(embeddings / norm, "float32")
    product = np.dot(embeddings, embeddings.T)
    np.save(os.path.join(output_dir, 'syno50/cos_sim_counter_fitting.npy'), product)
    return product


def _syno50_preprocess_one_text(text_ls, word2idx, idx2word, cos_sim):
    len_text = len(text_ls)
    # get the pos and verb tense info
    words_perturb = []
    pos_ls = criteria.get_pos(text_ls)
    pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
    for pos in pos_pref:
        for i in range(len(pos_ls)):
            if pos_ls[i] == pos and len(text_ls[i]) > 2:
                words_perturb.append((i, text_ls[i]))

    words_perturb.sort(key= lambda x:x[0])

    # find synonyms and make a dict of synonyms of each word.
    words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
    synonym_words, synonym_values = [], []
    for idx in words_perturb_idx:
        res = list(zip(*(cos_sim[idx])))
        temp = []
        for ii in res[1]:
            temp.append(idx2word[ii])
        synonym_words.append(temp)
        temp = []
        for ii in res[0]:
            temp.append(ii)
        synonym_values.append(temp)
    synonyms_all = []
    synonyms_dict = defaultdict(list)
    for idx, word in words_perturb:
        if word in word2idx:
            synonyms = synonym_words.pop(0)
            if synonyms:
                synonyms_all.append((idx, synonyms))
                synonyms_dict[word] = synonyms

    pos_synoyms_dict = defaultdict(list)

    final_idx_word_list = []

    for i in range(len(synonyms_all)):
        random_text = text_ls[:]

        idx = synonyms_all[i][0]
        synonyms = synonyms_all[i][1]

        for j, syn in enumerate(synonyms):

            # skip the word itself
            if j == 0:
                continue

            random_text[idx] = syn

            if len(random_text) > 10:
                new_pos = criteria.get_pos(random_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
            else:
                new_pos = criteria.get_pos(random_text)[idx]

            pos_same = criteria.pos_filter(pos_ls[idx], [new_pos])[0]

            if pos_same:
                pos_synoyms_dict[(idx, text_ls[idx])].append(syn)

        if len(pos_synoyms_dict[(idx, text_ls[idx])]) > 0:
            final_idx_word_list.append((idx, text_ls[idx]))

    return final_idx_word_list, pos_synoyms_dict


def syno50_preprocess(output_dir, dataset, dataset_path, counter_fitting_embeddings_path, counter_fitting_cos_sim_path):
    if dataset in ['snli', 'mnli', 'mnli_matched', 'mnli_mismatched']:
        data = read_data_nli(dataset_path, data_size=999999999)
    else:
        import dataloader
        texts, labels = dataloader.read_corpus(dataset_path, csvf=False)
        data = list(zip(texts, labels))
    print("Data import finished!")

    # prepare synonym extractor
    # build dictionary via the embedding file
    print("Building vocab...")
    idx2word = {}
    word2idx = {}
    sim_lis = []
    with open(counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    # for cosine similarity matrix
    print("Building cos sim matrix...")
    if counter_fitting_cos_sim_path:
        # load pre-computed cosine similarity matrix if provided
        print('Load pre-computed cosine similarity matrix from {}'.format(counter_fitting_cos_sim_path))
        with open(counter_fitting_cos_sim_path, "rb") as fp:
            sim_lis = pickle.load(fp)
    else:
        # calculate the cosine similarity matrix
        print('Start computing the cosine similarity matrix!')
        embeddings = []
        with open(counter_fitting_embeddings_path, 'r') as ifile:
            for line in ifile:
                embedding = [float(num) for num in line.strip().split()[1:]]
                embeddings.append(embedding)
        embeddings = np.array(embeddings)
        print(embeddings.T.shape)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.asarray(embeddings / norm, "float64")
        cos_sim = np.dot(embeddings, embeddings.T)
    print("Cos sim import finished!")

    res_list = []

    if dataset in ['snli', 'mnli', 'mnli_matched', 'mnli_mismatched']:
        for idx, premise in tqdm(enumerate(data['premises'])):

            hypothesis, true_label = data['hypotheses'][idx], data['labels'][idx]
            idx_word_list, pos_syno_dict = _syno50_preprocess_one_text(hypothesis, word2idx, idx2word, sim_lis)

            res_list.append((idx_word_list, pos_syno_dict))
    elif dataset in ['imdb', 'mr', 'agnews', 'yahoo']:
        for idx, (text, true_label) in tqdm(enumerate(data)):
            idx_word_list, pos_syno_dict = _syno50_preprocess_one_text(text, word2idx, idx2word, sim_lis)

            res_list.append((idx_word_list, pos_syno_dict))

    save_path = os.path.join(output_dir, dataset_path.split('/')[-1])

    with open(save_path, 'wb') as f:
        pickle.dump(res_list, f)

    print('save to', save_path)

# if __name__ == '__main__':
    # com_sim_compute("/home/workspace/big_data/hardlabel/counter-fitted-vectors.txt")