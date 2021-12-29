import numpy as np

# np.random.seed(1234)
import time
from multiprocessing import Pool

import criteria
import random

# random.seed(2333)
from collections import defaultdict
import itertools
import tensorflow.compat.v1 as tf

# To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.compat.v1.disable_eager_execution()
import torch

from local_models.sim_models import calc_sim
from attackers.score_based_attacker import SolutionScoreBase, ScoreBasedAttacker

MY_INF = 9999


# class SolutionLS:
#     def __init__(self, sent, diff_list, attack_score, attack_success):
#
#         self.sent = sent
#         self.diff_list = diff_list
#         self.diff_indices = set()
#         for w_i, w in self.diff_list:
#             self.diff_indices.add(w_i)
#
#         self.attack_score = attack_score
#         self.attack_success = attack_success
#
#         self._text = None
#
#     @property
#     def text(self):
#         if not self._text:
#             self._text = " ".join(self.sent)
#         return self._text
#
#     def __str__(self):
#         return f'Attack Score: {self.attack_score}, Change Num: {len(self.diff_indices)}' \
#                f'\nText: {self.text}\n' \
#                f'Diff List: {self.diff_list}\n'
#
# class LocalSearchAttacker(object):
#
#     def __init__(self,
#                  predictor,
#                  sim_predictor,
#                  sim_score_window,
#                  is_classification,
#                  ):
#         self.predictor = predictor
#         self.sim_predictor = sim_predictor
#         self.sim_score_window = sim_score_window
#         self.is_classification = is_classification
#
#         print('************* Local Search ******************')
#         print(f'No Init | Sample Func: Softmax')
#
#         # =========== init when attacking ==================
#
#         # == assign in each attack
#         # for NLI task
#         if not is_classification:
#             self.premise = None
#
#         self.sent_orig = None
#
#         self.true_label = None
#
#         self.idx_word_list = None
#
#         self.syno_dict = None
#
#
#         # == reset in each attack
#         # record query number
#         self.qry_num = None
#
#         # debug
#         self.debug = False
#
#
#     def init_attack(self, idx_word_list, syno_dict, sent_orig, premise, true_label):
#         # assign value
#         self.idx_word_list = idx_word_list
#
#         self.syno_dict = syno_dict
#
#         self.sent_orig = sent_orig
#
#         if self.is_classification:
#             assert premise is None
#         else:
#             self.premise = premise
#
#         self.true_label = true_label
#
#
#         # reset
#         self.qry_num = 0
#         self.real_qry_num = 0
#         self.reached = set()
#         self.is_success = False
#         self.eval_cache = {}
#
#
#     def init_sol(self):
#
#             # rnd_indices = list(range(len(self.idx_word_list)))
#             # init_best_sent = None
#             # init_best_score = 0
#             #
#             # for cur_change_num in range(len(self.idx_word_list)):
#             #     random_sent = self.sent_orig.copy()
#             #
#             #     random.shuffle(rnd_indices)
#             #
#             #     for i in rnd_indices[:cur_change_num + 1]:
#             #         idx, o_word = self.idx_word_list[i]
#             #         syn = self.syno_dict[(idx, o_word)]
#             #         random_sent[idx] = random.choice(syn)
#             #
#             #     attack_score = self._attack_score_untargetted([random_sent])[0]
#             #
#             #     if attack_score > init_best_score:
#             #         init_best_sent = random_sent
#             #         init_best_score = attack_score
#             #
#             #         if attack_score > self.attack_success_threshold:
#             #             self.is_success = True
#             #             break
#             #
#             # if not self.is_success:
#             #     for i in range(2500):
#             #         random_sent = self.sent_orig.copy()
#             #         for idx, o_word in self.idx_word_list:
#             #             syn = self.syno_dict[(idx, o_word)]
#             #             random_sent[idx] = random.choice(syn)
#             #
#             #         attack_score = self._attack_score_untargetted([random_sent])[0]
#             #
#             #         if attack_score > init_best_score:
#             #             init_best_sent = random_sent
#             #             init_best_score = attack_score
#             #
#             #             if attack_score > self.attack_success_threshold:
#             #                 self.is_success = True
#             #                 break
#             #
#             # sol_init = self.evaluate_sents([init_best_sent])[0]
#
#         sol_init = self.evaluate_sents([self.sent_orig.copy()])[0]
#         self.qry_num = self.real_qry_num
#
#         return sol_init
#
#
#     def _geedy(self, sol_: SolutionLS):
#
#         best_sol = sol_
#         # best_sent = sol.sent
#         if best_sol.attack_success:
#             print(best_sol)
#             tmp = self.predictor({'premises': [self.premise], 'hypotheses': [best_sol.sent]})
#             print(tmp)
#
#
#         assert not best_sol.attack_success
#
#
#         while True:
#             new_sent_list = []
#
#             # generate new solution
#             for idx, orig_word in self.idx_word_list:
#
#                 # add and swap
#                 syno_list = self.syno_dict[(idx, orig_word)]
#                 for syno in syno_list:
#                     if syno == best_sol.sent[idx]:
#                         continue
#                     _new_sent = self.perturb(best_sol.sent, idx, syno)
#                     new_sent_list.append(_new_sent)
#
#                 # delete
#                 if idx in best_sol.diff_indices:
#                     _new_sent = self.perturb(best_sol.sent, idx, self.sent_orig[idx])
#                     new_sent_list.append(_new_sent)
#
#             # evaluate
#             new_sol_list = self.evaluate_sents(new_sent_list)
#
#             # get the best of new generated
#             cur_best_score = -1
#             cur_best_sol = None
#
#             for new_sol in new_sol_list:
#                 if new_sol.attack_score > cur_best_score:
#                     cur_best_score = new_sol.attack_score
#                     cur_best_sol = new_sol
#
#             # check if attack is success
#             if cur_best_sol.attack_success:
#                 best_sol = cur_best_sol
#                 break
#
#             # update global best
#             if cur_best_score > best_sol.attack_score:
#                 best_sol = cur_best_sol
#             else:
#                 break
#
#         return best_sol
#
#     def _check_supplement(self, sol: SolutionLS):
#         new_sent = self.sent_orig.copy()
#
#         for idx, orig_w in self.idx_word_list:
#             cur_key = (idx, orig_w)
#
#             if idx in sol.diff_indices:
#                 if len(self.syno_dict[cur_key]) > 2:
#                     return sol
#                 else:
#                     for syno in self.syno_dict[cur_key]:
#                         if sol.sent[idx] != syno:
#                             new_sent[idx] = syno
#             else:
#                 if len(self.syno_dict[cur_key]) > 1:
#                     return sol
#                 else:
#                     new_sent[idx] = self.syno_dict[cur_key][0]
#
#         new_sol = self.evaluate_sents(new_sent)[0]
#
#         if new_sol.attack_score > sol.attack_success:
#             return new_sol
#         else:
#             return sol
#
#
#     def local_search(self, sol_init: SolutionLS):
#
#         best_sol = self._geedy(sol_init)
#
#         if not best_sol.attack_success:
#             best_sol = self._check_supplement(best_sol)
#
#         best_sim = calc_sim(self.sent_orig, [best_sol.sent], -1, self.sim_score_window, self.sim_predictor)[0]
#
#         print('--- Best ---')
#         print('Similarity', best_sim)
#         print(best_sol)
#         print('------------')
#
#         return best_sol.sent, len(best_sol.diff_list), best_sim, 1
#
#     def attack(self, idx_word_list, syno_dict, sent_orig, premise, true_label):
#
#         self.init_attack(idx_word_list, syno_dict, sent_orig, premise, true_label)
#
#         sol_init = self.init_sol()
#         if self.is_success:
#             print('Init Success!')
#         # else:
#         #     print('Init Fail!')
#         #     return None, 0, 1, self.qry_num, 0, 0, 0, self.real_qry_num
#
#         sent, change_num, sim, converge = self.local_search(sol_init)
#         print(f'Query Number: {self.qry_num} Real Query Number: {self.real_qry_num}')
#
#         return sent, change_num, sim,  self.qry_num, converge, 0, 0, self.real_qry_num
#
#     def evaluate_sents(self, list_of_sent, diff_lists=None):
#
#         self.qry_num += len(list_of_sent) # may be cached
#
#         if diff_lists is not None:
#             assert len(list_of_sent) == len(diff_lists)
#
#
#         uncache_indices = []
#         uncache_texts = []
#         cache_indices = []
#         cache_texts = []
#
#
#         list_of_sent_uncache = []
#         diff_lists_uncache = []
#         # list_of_sent_cache = []
#
#         for idx_, sent_ in enumerate(list_of_sent):
#             text_ = " ".join(sent_)
#             if text_ not in self.eval_cache:
#                 uncache_indices.append(idx_)
#                 uncache_texts.append(text_)
#                 list_of_sent_uncache.append(sent_)
#                 if diff_lists:
#                     diff_lists_uncache.append(diff_lists[idx_])
#             else:
#                 cache_indices.append(idx_)
#                 cache_texts.append(text_)
#
#         sol_list_uncache = []
#
#         # generate uncached solution
#         if len(uncache_indices) > 0:
#             if diff_lists is None:
#                 diff_lists_uncache = self._diff_list(list_of_sent_uncache)
#
#             score_list_uncache, attack_success_list_uncache = self._attack_score_untargetted(list_of_sent_uncache)
#
#             for i in range(len(list_of_sent_uncache)):
#                 sent = list_of_sent_uncache[i]
#                 diff_list = diff_lists_uncache[i]
#                 attack_score = score_list_uncache[i]
#                 attack_success = attack_success_list_uncache[i]
#
#                 if attack_success:
#                     self.is_success = True
#
#                 new_sol = SolutionLS(sent, diff_list, attack_score, attack_success)
#
#                 sol_list_uncache.append(new_sol)
#
#
#         sol_list = [None for _ in range(len(list_of_sent))]
#
#         # cached solution
#         for j in range(len(cache_indices)):
#             cache_idx = cache_indices[j]
#             cache_text = cache_texts[j]
#             cache_sol = self.eval_cache[cache_text]
#
#             assert cache_sol.attack_success == False
#
#             sol_list[cache_idx] = cache_sol
#
#         # uncached solution
#         for j in range(len(uncache_indices)):
#             uncache_idx = uncache_indices[j]
#             uncache_sol = sol_list_uncache[j]
#             uncache_text = uncache_texts[j]
#             sol_list[uncache_idx] = uncache_sol
#
#             self.eval_cache[uncache_text] = uncache_sol
#
#         return sol_list
#
#     def _diff_list(self, list_of_sent):
#
#         ret = []
#
#         for _sent in list_of_sent:
#             _diff_list = []
#             for i, _ in self.idx_word_list:
#                 if _sent[i] != self.sent_orig[i]:
#                     _diff_list.append((i, _sent[i]))
#
#             ret.append(_diff_list)
#
#         return ret
#
#     def _attack_score_untargetted(self, list_of_sent):
#
#         # TODO: untargeted score
#         self.real_qry_num += len(list_of_sent)
#
#         if self.is_classification:
#             new_probs = self.predictor(list_of_sent, batch_size=128)
#         else:
#             new_probs = self.predictor({'premises': [self.premise] * len(list_of_sent), 'hypotheses': list_of_sent})
#
#         attack_success_list = (self.true_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
#         new_probs = new_probs.data.cpu().numpy()
#         untargeted_scores = 1 - new_probs[:, self.true_label]
#
#         return untargeted_scores, attack_success_list
#
#     def perturb(self, sent, change_idx, change_w):
#         new_sent = sent.copy()
#         new_sent[change_idx] = change_w
#
#         return new_sent


class LocalSearchAttacker(ScoreBasedAttacker):

    # def __init__(self,
    #              predictor,
    #              sim_predictor,
    #              sim_score_window,
    #              is_classification,
    #              goal_function
    #              ):
    #
    #     super().__init__(predictor=predictor,
    #              sim_predictor=sim_predictor,
    #              sim_score_window=sim_score_window,
    #              is_classification=is_classification,
    #                      goal_function=goal_function
    #                      )

        # self.predictor = predictor
        # self.sim_predictor = sim_predictor
        # self.sim_score_window = sim_score_window
        # self.is_classification = is_classification
        #
        # print('************* Local Search ******************')
        # print(f'No Init | Sample Func: Softmax')
        #
        # # =========== init when attacking ==================
        #
        # # == assign in each attack
        # # for NLI task
        # if not is_classification:
        #     self.premise = None
        #
        # self.sent_orig = None
        #
        # self.true_label = None
        #
        # self.idx_word_list = None
        #
        # self.syno_dict = None
        #
        #
        # # == reset in each attack
        # # record query number
        # self.qry_num = None
        #
        # # debug
        # self.debug = False


    # def init_attack(self, idx_word_list, syno_dict, sent_orig, premise, true_label):
    #     # assign value
    #     self.idx_word_list = idx_word_list
    #
    #     self.syno_dict = syno_dict
    #
    #     self.sent_orig = sent_orig
    #
    #     if self.is_classification:
    #         assert premise is None
    #     else:
    #         self.premise = premise
    #
    #     self.true_label = true_label
    #
    #
    #     # reset
    #     self.qry_num = 0
    #     self.real_qry_num = 0
    #     self.reached = set()
    #     self.is_success = False
    #     self.eval_cache = {}


    def init_sol(self):

        sol_init = super().evaluate_sents([self.sent_orig.copy()])[0]
        self.qry_num = self.real_qry_num

        return sol_init


    def _geedy(self, sol_):

        best_sol = sol_
        # best_sent = sol.sent
        if best_sol.attack_success:
            print(best_sol)
            tmp = self.predictor({'premises': [self.premise], 'hypotheses': [best_sol.sent]})
            print(tmp)


        assert not best_sol.attack_success


        while True:
            new_sent_list = []

            # generate new solution
            for idx, orig_word in self.idx_word_list:

                # add and swap
                syno_list = self.syno_dict[(idx, orig_word)]
                for syno in syno_list:
                    if syno == best_sol.sent[idx]:
                        continue
                    _new_sent = self.perturb(best_sol.sent, idx, syno)
                    new_sent_list.append(_new_sent)

                # delete
                if idx in best_sol.diff_indices:
                    _new_sent = self.perturb(best_sol.sent, idx, self.sent_orig[idx])
                    new_sent_list.append(_new_sent)

            # evaluate
            new_sol_list = super().evaluate_sents(new_sent_list)

            # get the best of new generated
            cur_best_score = -1
            cur_best_sol = None

            for new_sol in new_sol_list:
                if new_sol.attack_score > cur_best_score:
                    cur_best_score = new_sol.attack_score
                    cur_best_sol = new_sol

            # check if attack is success
            if cur_best_sol.attack_success:
                best_sol = cur_best_sol
                break

            # update global best
            if cur_best_score > best_sol.attack_score:
                best_sol = cur_best_sol
            else:
                break

        return best_sol

    def _check_supplement(self, sol):
        new_sent = self.sent_orig.copy()

        for idx, orig_w in self.idx_word_list:
            cur_key = (idx, orig_w)

            if idx in sol.diff_indices:
                if len(self.syno_dict[cur_key]) > 2:
                    return sol
                else:
                    for syno in self.syno_dict[cur_key]:
                        if sol.sent[idx] != syno:
                            new_sent[idx] = syno
            else:
                if len(self.syno_dict[cur_key]) > 1:
                    return sol
                else:
                    new_sent[idx] = self.syno_dict[cur_key][0]

        new_sol = super().evaluate_sents(new_sent)[0]

        if new_sol.attack_score > sol.attack_success:
            return new_sol
        else:
            return sol

    def local_search(self, sol_init):

        best_sol = self._geedy(sol_init)

        if not best_sol.attack_success:
            best_sol = self._check_supplement(best_sol)

        best_sim = calc_sim(self.sent_orig, [best_sol.sent], -1, self.sim_score_window, self.sim_predictor)[0]

        print('--- Best ---')
        print('Similarity', best_sim)
        print(best_sol)
        print('------------')

        return best_sol.sent, len(best_sol.diff_list), best_sim, 1

    def attack(self, idx_word_list, syno_dict, sent_orig, premise, true_label, target_label):

        self.init_attack(idx_word_list, syno_dict, sent_orig, premise, true_label, target_label)

        sol_init = self.init_sol()
        if self.is_success:
            print('Init Success!')
        # else:
        #     print('Init Fail!')
        #     return None, 0, 1, self.qry_num, 0, 0, 0, self.real_qry_num

        sent, change_num, sim, converge = self.local_search(sol_init)
        print(f'Query Number: {self.qry_num} Real Query Number: {self.real_qry_num}')

        return sent, change_num, sim,  self.qry_num, converge, 0, 0, self.real_qry_num

    def perturb(self, sent, change_idx, change_w):
        new_sent = sent.copy()
        new_sent[change_idx] = change_w

        return new_sent
