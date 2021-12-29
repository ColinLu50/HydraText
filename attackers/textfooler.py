import os
import numpy as np
import criteria
import random
from local_models.sim_models import calc_sim

import torch


class PredictorCache(object):
    def __init__(self, predictor, is_classification=True):
        self.predictor = predictor

        self.cache_dict = {}
        self.real_qry_num = 0

        self.premise = None
        self.is_classification = is_classification

    def reset_state(self, premise):
        if self.is_classification:
            assert premise is None
        else:
            self.premise = premise

        self.cache_dict = {}
        self.real_qry_num = 0

    def get_probs_cache(self, new_word_lists, batch_size=64):

        # split into cache and uncache
        uncache_w_lists = []
        uncache_indices = []

        pr_final = [None for _ in range(len(new_word_lists))]

        for _idx, _w_list in enumerate(new_word_lists):
            _text = " ".join(_w_list)
            if _text in self.cache_dict:
                pr_final[_idx] = self.cache_dict[_text]
            else:
                uncache_w_lists.append(_w_list)
                uncache_indices.append(_idx)

        if len(uncache_indices) > 0:
            uncache_pr = self._get_probs_uncache(uncache_w_lists, batch_size)
            for i, _pr in enumerate(uncache_pr):
                _idx = uncache_indices[i]
                pr_final[_idx] = _pr

        pr_final = torch.tensor(pr_final)

        return pr_final

    def _get_probs_uncache(self, uncache_word_lists, batch_size):
        if not self.is_classification:
            uncache_probs = self.predictor({'premises': [self.premise] * len(uncache_word_lists), 'hypotheses': uncache_word_lists})
        else:
            uncache_probs = self.predictor(uncache_word_lists, batch_size=batch_size)
        uncache_probs = uncache_probs.data.cpu().numpy()

        self.real_qry_num += len(uncache_word_lists)

        for _idx, _w_list in enumerate(uncache_word_lists):
            _text = " ".join(_w_list)
            self.cache_dict[_text] = uncache_probs[_idx]

        return uncache_probs


def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values


def attack(oov_string, text_ls, true_label, target_label, predictor_cache, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
           batch_size=32, is_targeted_goal=False):

    if not is_targeted_goal:
        assert target_label is None

    orig_probs = predictor_cache.get_probs_cache([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()

    len_text = len(text_ls)
    if len_text < sim_score_window:
        sim_score_threshold = 0.1  # shut down the similarity thresholding function
    half_sim_score_window = (sim_score_window - 1) // 2
    num_queries = 1

    # get the pos and verb tense info
    pos_ls = criteria.get_pos(text_ls)

    # get importance score
    leave_1_texts = [text_ls[:ii] + [oov_string] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
    leave_1_probs = predictor_cache.get_probs_cache(leave_1_texts, batch_size=batch_size)
    num_queries += len(leave_1_texts)
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                                  leave_1_probs_argmax))).data.cpu().numpy()

    # get words to perturb ranked by importance scorefor word in words_perturb
    words_perturb = []
    for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
        # try:
        #     if score > import_score_threshold and text_ls[idx] not in stop_words_set:
        #         words_perturb.append((idx, text_ls[idx]))
        # except:
        #     print(idx, len(text_ls), import_scores.shape, text_ls, len(leave_1_texts))
        if score > import_score_threshold and text_ls[idx] not in stop_words_set:
            words_perturb.append((idx, text_ls[idx]))

    # find synonyms
    words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
    synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
    synonyms_all = []
    for idx, word in words_perturb:
        if word in word2idx:
            synonyms = synonym_words.pop(0)
            if synonyms:
                synonyms_all.append((idx, synonyms))

    # start replacing and attacking
    text_prime = text_ls[:]
    text_cache = text_prime[:]
    num_changed = 0
    for idx, synonyms in synonyms_all:
        new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
        new_probs = predictor_cache.get_probs_cache(new_texts, batch_size=batch_size)

        # Error: raw code have problem, sometimes [text_range_min] is negative, which is obviously unexpected

        # compute semantic similarity
        # if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        #     text_range_min = idx - half_sim_score_window
        #     text_range_max = idx + half_sim_score_window + 1
        # elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        #     text_range_min = 0
        #     text_range_max = sim_score_window
        # elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
        #     text_range_min = len_text - sim_score_window
        #     text_range_max = len_text
        # else:
        #     text_range_min = 0
        #     text_range_max = len_text
        #
        #
        #
        # semantic_sims = \
        # sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
        #                            list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

        semantic_sims = calc_sim(text_cache, new_texts, idx, sim_score_window, sim_predictor)

        num_queries += len(new_texts)
        if len(new_probs.shape) < 2:
            new_probs = new_probs.unsqueeze(0)

        if not is_targeted_goal:
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
        else:
            new_probs_mask = (target_label == torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
        # prevent bad synonyms
        new_probs_mask *= (semantic_sims >= sim_score_threshold)
        # prevent incompatible pos
        synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                           if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
        pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
        new_probs_mask *= pos_mask

        if np.sum(new_probs_mask) > 0:
            text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
            num_changed += 1
            break
        else:
            new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                    (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float() #.cuda()
            new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
            if new_label_prob_min < orig_prob:
                text_prime[idx] = synonyms[new_label_prob_argmin]
                num_changed += 1
        text_cache = text_prime[:]

    sim = calc_sim(text_ls, [text_prime], -1, sim_score_window, sim_predictor)[0]

    return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor_cache.get_probs_cache([text_prime])), num_queries, sim





# def C_attack(text_ls, true_label, predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
#            import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
#            batch_size=32):
#     # first check the prediction of the original text
#     orig_probs = predictor([text_ls]).squeeze()
#     orig_label = torch.argmax(orig_probs)
#     orig_prob = orig_probs.max()
#     if true_label != orig_label:
#         return '', 0, orig_label, orig_label, 0
#     else:
#         len_text = len(text_ls)
#         if len_text < sim_score_window:
#             sim_score_threshold = 0.1  # shut down the similarity thresholding function
#         half_sim_score_window = (sim_score_window - 1) // 2
#         num_queries = 1
#
#         # get the pos and verb tense info
#         pos_ls = criteria.get_pos(text_ls)
#
#         # get importance score
#         leave_1_texts = [text_ls[:ii] + ['<oov>'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
#         leave_1_probs = predictor(leave_1_texts, batch_size=batch_size)
#         num_queries += len(leave_1_texts)
#         leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
#         import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
#                     leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
#                                                                       leave_1_probs_argmax))).data.cpu().numpy()
#
#         # get words to perturb ranked by importance scorefor word in words_perturb
#         words_perturb = []
#         for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
#             try:
#                 if score > import_score_threshold and text_ls[idx] not in stop_words_set:
#                     words_perturb.append((idx, text_ls[idx]))
#             except:
#                 print(idx, len(text_ls), import_scores.shape, text_ls, len(leave_1_texts))
#
#         # find synonyms
#         words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
#         synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
#         synonyms_all = []
#         for idx, word in words_perturb:
#             if word in word2idx:
#                 synonyms = synonym_words.pop(0)
#                 if synonyms:
#                     synonyms_all.append((idx, synonyms))
#
#         # start replacing and attacking
#         text_prime = text_ls[:]
#         text_cache = text_prime[:]
#         num_changed = 0
#         for idx, synonyms in synonyms_all:
#             new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
#             new_probs = predictor(new_texts, batch_size=batch_size)
#
#             # compute semantic similarity
#             if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
#                 text_range_min = idx - half_sim_score_window
#                 text_range_max = idx + half_sim_score_window + 1
#             elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
#                 text_range_min = 0
#                 text_range_max = sim_score_window
#             elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
#                 text_range_min = len_text - sim_score_window
#                 text_range_max = len_text
#             else:
#                 text_range_min = 0
#                 text_range_max = len_text
#             semantic_sims = \
#             sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
#                                        list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]
#
#             num_queries += len(new_texts)
#             if len(new_probs.shape) < 2:
#                 new_probs = new_probs.unsqueeze(0)
#             new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
#             # prevent bad synonyms
#             new_probs_mask *= (semantic_sims >= sim_score_threshold)
#             # prevent incompatible pos
#             synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
#                                if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
#             pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
#             new_probs_mask *= pos_mask
#
#             if np.sum(new_probs_mask) > 0:
#                 text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
#                 num_changed += 1
#                 break
#             else:
#                 new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
#                         (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
#                 new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
#                 if new_label_prob_min < orig_prob:
#                     text_prime[idx] = synonyms[new_label_prob_argmin]
#                     num_changed += 1
#             text_cache = text_prime[:]
#         return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])), num_queries
#
#
# def NLI_attack(premise, hypothese, true_label, predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
#            import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50, batch_size=32):
#     # first check the prediction of the original text
#     orig_probs = predictor({'premises': [premise], 'hypotheses': [hypothese]}).squeeze()
#     orig_label = torch.argmax(orig_probs)
#     orig_prob = orig_probs.max()
#     if true_label != orig_label:
#         return '', 0, orig_label, orig_label, 0
#     else:
#         len_text = len(hypothese)
#         if len_text < sim_score_window:
#             sim_score_threshold = 0.1  # shut down the similarity thresholding function
#         half_sim_score_window = (sim_score_window - 1) // 2
#         num_queries = 1
#
#         # get the pos and verb tense info
#         pos_ls = criteria.get_pos(hypothese)
#
#         # get importance score
#         leave_1_texts = [hypothese[:ii]+['<oov>']+hypothese[min(ii+1, len_text):] for ii in range(len_text)]
#         leave_1_probs = predictor({'premises':[premise]*len_text, 'hypotheses': leave_1_texts})
#         num_queries += len(leave_1_texts)
#         leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
#         import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
#                     leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
#                                                                       leave_1_probs_argmax))).data.cpu().numpy()
#
#         # get words to perturb ranked by importance scorefor word in words_perturb
#         words_perturb = []
#         for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
#             if score > import_score_threshold and hypothese[idx] not in stop_words_set:
#                 words_perturb.append((idx, hypothese[idx]))
#
#         # find synonyms
#         words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
#         synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
#         synonyms_all = []
#         for idx, word in words_perturb:
#             if word in word2idx:
#                 synonyms = synonym_words.pop(0)
#                 if synonyms:
#                     synonyms_all.append((idx, synonyms))
#
#         # start replacing and attacking
#         text_prime = hypothese[:]
#         text_cache = text_prime[:]
#         num_changed = 0
#         for idx, synonyms in synonyms_all:
#             new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
#             new_probs = predictor({'premises': [premise] * len(synonyms), 'hypotheses': new_texts})
#
#             # compute semantic similarity
#             if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
#                 text_range_min = idx - half_sim_score_window
#                 text_range_max = idx + half_sim_score_window + 1
#             elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
#                 text_range_min = 0
#                 text_range_max = sim_score_window
#             elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
#                 text_range_min = len_text - sim_score_window
#                 text_range_max = len_text
#             else:
#                 text_range_min = 0
#                 text_range_max = len_text
#             semantic_sims = \
#             sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
#                                        list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]
#
#             num_queries += len(new_texts)
#             if len(new_probs.shape) < 2:
#                 new_probs = new_probs.unsqueeze(0)
#             new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
#             # prevent bad synonyms
#             new_probs_mask *= (semantic_sims >= sim_score_threshold)
#             # prevent incompatible pos
#             synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
#                                if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
#             pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
#             new_probs_mask *= pos_mask
#
#             if np.sum(new_probs_mask) > 0:
#                 text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
#                 num_changed += 1
#                 break
#             else:
#                 new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
#                     (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
#                 new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
#                 if new_label_prob_min < orig_prob:
#                     text_prime[idx] = synonyms[new_label_prob_argmin]
#                     num_changed += 1
#             text_cache = text_prime[:]
#
#         return ' '.join(text_prime), num_changed, orig_label, \
#                torch.argmax(predictor({'premises':[premise], 'hypotheses': [text_prime]})), num_queries

