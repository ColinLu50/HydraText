import random
import itertools
import torch
import numpy as np
from scipy.special import softmax

from local_models.sim_models import calc_sim


MY_INF = 9999


class Solution:
    def __init__(self, sent, diff_list, attack_score, change_number, sim):

        self.sent = sent
        self.diff_list = diff_list

        self.scores = [attack_score, change_number]
        self.sim = sim

        self.mut_tag = np.ones(3)
        self.search_space = [None, None, None]
        self.search_st = [None, None, None]

        self.attack_success = False
        self.reached_spaces = [[False], [False], [False]]

        self._text = None


    def dominate(self, other_solution):

        better_exist = False
        for f_idx in range(len(self.scores)):
            if self.scores[f_idx] < other_solution.scores[f_idx]:
                return False, False

            if not better_exist and self.scores[f_idx] > other_solution.scores[f_idx]:
                better_exist = True

        return True, better_exist


    def same_as(self, other_solution):
        for f_idx in range(len(self.scores)):
            if self.scores[f_idx] != other_solution.scores[f_idx]:
                return False

        return True

    @property
    def text(self):
        if not self._text:
            self._text = " ".join(self.sent)
        return self._text

    def __str__(self):
        return f'Attack Score: {self.scores[0]}, Change Score: {self.scores[1]}, Sim Score: {self.sim}' \
               f'\nText: {self.text}\n' \
               f'Diff List: {self.diff_list}\n'

class MOSearchAttacker(object):

    def __init__(self,
                 predictor,
                 sim_predictor,
                 sim_score_window,
                 qry_budget,
                 is_classification,
                 goal_function='untarget',
                 setting='decision'
                 ):
        self.predictor = predictor
        self.sim_predictor = sim_predictor
        self.sim_score_window = sim_score_window
        self.qry_budget = qry_budget
        self.is_classification = is_classification
        self.goal_function = goal_function
        self.setting = setting

        print('*******************************')
        print(f'Init Func: Orig | Sample Func: Softmax | Setting: {self.setting} | Goal Function: {self.goal_function}')

        # =========== init when attacking ==================

        # == assign in each attack
        # for NLI task
        if not is_classification:
            self.premise = None

        self.sent_orig = None

        self.true_label = None
        self.target_label = None

        self.idx_word_list = None

        self.pos_syno_dict = None


        # == reset in each attack
        # record query number
        self.qry_num = None

        # record reached solutions
        self.reached = None

        # debug
        self.debug = False


    def init_attack(self, idx_word_list, pos_syno_dict, sent_orig, premise, true_label, target_label):
        # assign value
        self.idx_word_list = idx_word_list

        self.pos_syno_dict = pos_syno_dict

        self.sent_orig = sent_orig

        if self.is_classification:
            assert premise is None
        else:
            self.premise = premise

        self.true_label = true_label
        if self.goal_function == 'target':
            self.target_label = target_label
        else:
            assert target_label is None


        # reset
        self.qry_num = 0
        self.real_qry_num = 0
        self.reached = set()
        self.is_success = False
        self.eval_cache = {}


    def init_P_decision(self):

        rnd_indices = list(range(len(self.idx_word_list)))

        for cur_change_num in range(len(self.idx_word_list)):
            random_sent = self.sent_orig.copy()

            random.shuffle(rnd_indices)

            for i in rnd_indices[:cur_change_num + 1]:
                idx, o_word = self.idx_word_list[i]
                syn = self.pos_syno_dict[(idx, o_word)]
                random_sent[idx] = random.choice(syn)

            attack_success = self._attack_success([random_sent])[0]
            if attack_success:
                self.is_success = True
                break

        if not self.is_success:
            for i in range(2500):
                random_sent = self.sent_orig.copy()
                for idx, o_word in self.idx_word_list:
                    syn = self.pos_syno_dict[(idx, o_word)]
                    random_sent[idx] = random.choice(syn)

                attack_success = self._attack_success([random_sent])[0]
                if attack_success:
                    self.is_success = True
                    break

        P_init = self.evaluate_sents([random_sent])
        self.qry_num = self.real_qry_num


        return P_init

    def init_P_score(self):

        P_init = self.evaluate_sents([self.sent_orig.copy()])
        self.qry_num = self.real_qry_num

        return P_init

        # init_best_sent = None
        # init_best_score = -MY_INF
        #
        # rnd_indices = list(range(len(self.idx_word_list)))
        #
        # for cur_change_num in range(len(self.idx_word_list)):
        #     random_sent = self.sent_orig.copy()
        #
        #     random.shuffle(rnd_indices)
        #
        #     for i in rnd_indices[:cur_change_num + 1]:
        #         idx, o_word = self.idx_word_list[i]
        #         syn = self.pos_syno_dict[(idx, o_word)]
        #         random_sent[idx] = random.choice(syn)
        #
        #     attack_score_l, attack_success_l = self._attack_score_untargetted([random_sent])
        #     attack_score = attack_score_l[0]
        #     attack_success = attack_success_l[0]
        #
        #     if init_best_score < attack_score:
        #         init_best_score = attack_score
        #         init_best_sent = random_sent
        #
        #     if attack_success:
        #         self.is_success = True
        #         break
        #
        # if not self.is_success:
        #     for i in range(2500):
        #         random_sent = self.sent_orig.copy()
        #         for idx, o_word in self.idx_word_list:
        #             syn = self.pos_syno_dict[(idx, o_word)]
        #             random_sent[idx] = random.choice(syn)
        #
        #         attack_score_l, attack_success_l = self._attack_score_untargetted([random_sent])
        #         attack_score = attack_score_l[0]
        #         attack_success = attack_success_l[0]
        #
        #         if init_best_score < attack_score:
        #             init_best_score = attack_score
        #             init_best_sent = random_sent
        #
        #         if attack_success:
        #             self.is_success = True
        #             break
        #
        # P_init = self.evaluate_sents([init_best_sent])
        # self.qry_num = self.real_qry_num
        #
        #
        # return P_init


    def _MO_search(self, P_init):

        # init P
        P = P_init

        best_sol = P[-1]
        # best_sol_idx = len(P) - 1

        rnd_sim = best_sol.sim
        rnd_change = len(best_sol.diff_list)

        converge_flag = 0

        for _sol in P:
            if _sol.attack_success:
                self.reached.add(_sol.text)


        # while True:
        for _t in range(20000):

            # check if there is search space
            tmp = 0
            for sol in P:
                tmp += sum(sol.mut_tag)
            if tmp == 0:
                print('NO search space!')
                converge_flag = -1
                break


            # random sample a solution from P
            if len(P) > 1:
                F2_abs = [abs(_sol.scores[1]) for _sol in P]
                sample_prob = softmax(F2_abs)
                cur_sol = np.random.choice(P, p=sample_prob)
            else:
                cur_sol = P[0]

            self.build_search_space(cur_sol)

            # mutate
            new_sent_list = []
            new_diff_lists = []

            # add mutation
            if cur_sol.mut_tag[0] > 0 and not cur_sol.attack_success:
                add_sent, add_diff_list = self._mutate_add(cur_sol)
                if add_sent:
                    new_sent_list.append(add_sent)
                    new_diff_lists.append(add_diff_list)

            # swap mut
            if cur_sol.mut_tag[1] > 0:
                swap_sent, swap_diff_list = self._mutate_swap(cur_sol)
                if swap_sent:
                    new_sent_list.append(swap_sent)
                    new_diff_lists.append(swap_diff_list)

            # remove mut
            if cur_sol.mut_tag[2] > 0:
                rm_sent, rm_diff_list = self._mutate_rm(cur_sol)
                if rm_sent:
                    new_sent_list.append(rm_sent)
                    new_diff_lists.append(rm_diff_list)


            if len(new_sent_list) > 0:

                P_new = self.evaluate_sents(new_sent_list, new_diff_lists)

                if self.debug:
                    print('============ New P ==================')
                    print('Cur sol:', str(cur_sol))
                    for _sol in P_new:
                        print(_sol)

                # udpate P
                for new_sol in P_new:
                    if new_sol:
                        P = self.update_P(P, new_sol)

                if self.debug:
                    print('=============== Updated P ===============')
                    for _sol in P:
                        print(_sol)

                # update best solution
                best_sol = P[0]
                best_sore = best_sol.scores[0]
                for sol_idx, sol in enumerate(P):
                    if sol.scores[0] > best_sore:
                        best_sore = sol.scores[0]
                        best_sol = sol

                # check if new best solution converges
                if self.is_success and all(best_sol.reached_spaces[1]) and all(best_sol.reached_spaces[2]):
                    print('Best Solution Converge!')
                    converge_flag = 1
                    break

            else:
                # check the converge solution is the best
                if self.is_success and all(best_sol.reached_spaces[1]) and all(best_sol.reached_spaces[2]):
                    print('Best Solution Converge!')
                    converge_flag = 1
                    break
                else:
                    continue

            # # check if query number is used off
            # if self.qry_num >= qry_budget:
            #     print('Out of Budget!')
            #
            #     break

        print('Pareto Size', len(P), 'Converge', converge_flag, 'Iteration:', _t)
        # for _sol in P:
        #     print(_sol)
        print('--- Best ---')
        print(best_sol)
        print('------------')

        return best_sol.sent, -best_sol.scores[1], best_sol.sim, self.qry_num, converge_flag, rnd_sim, rnd_change, self.real_qry_num

    def update_P(self, P, new_sol):

        add_new = True
        for exist_sol in P:
            # check strictly dominant
            is_dom, is_strict_dom = exist_sol.dominate(new_sol)
            if is_strict_dom:
                add_new = False
                break

            if new_sol.attack_success:
                is_same = exist_sol.same_as(new_sol)
                if is_same and exist_sol.sim > new_sol.sim:
                    add_new = False
                    break

        # add new solution, remove old solution
        if add_new:
            P_new = []
            for exist_sol in P:
                is_dom, is_strict_dom = new_sol.dominate(exist_sol)
                if not is_dom:
                    P_new.append(exist_sol)

            P_new.append(new_sol)
            return P_new
        else:
            return P

    def attack(self, idx_word_list, sub_dict, sent_orig, premise, true_label, target_label):

        self.init_attack(idx_word_list, sub_dict, sent_orig, premise, true_label, target_label)

        if self.setting == 'decision':
            P_init = self.init_P_decision()
            if self.is_success:
                print('Init Success!')
            else:
                print('Init Fail!')
        else:
            P_init = self.init_P_score()

        ret = self._MO_search(P_init)
        print(f'Query Number: {self.qry_num} Real Query Number: {self.real_qry_num}')

        return ret

    def evaluate_sents(self, list_of_sent, diff_lists=None):

        self.qry_num += len(list_of_sent) # may be cached

        if diff_lists is not None:
            assert len(list_of_sent) == len(diff_lists)


        uncache_indices = []
        uncache_texts = []
        cache_indices = []
        cache_texts = []


        list_of_sent_uncache = []
        diff_lists_uncache = []
        # list_of_sent_cache = []

        for idx_, sent_ in enumerate(list_of_sent):
            text_ = " ".join(sent_)
            if text_ not in self.eval_cache:
                uncache_indices.append(idx_)
                uncache_texts.append(text_)
                list_of_sent_uncache.append(sent_)
                if diff_lists:
                    diff_lists_uncache.append(diff_lists[idx_])
            else:
                cache_indices.append(idx_)
                cache_texts.append(text_)

        # generate uncached solution
        if len(uncache_indices) > 0:
            # get uncached list of {diff list}
            if diff_lists is None:
                diff_lists_uncache = self._diff_list(list_of_sent_uncache)

            sol_list_uncache = self._generate_uncache_solutions(list_of_sent_uncache, diff_lists_uncache)

        sol_list = [None for _ in range(len(list_of_sent))]

        # cached solution
        for j in range(len(cache_indices)):
            cache_idx = cache_indices[j]
            cache_text = cache_texts[j]
            cache_sol = self.eval_cache[cache_text]

            assert cache_sol.attack_success == False

            sol_list[cache_idx] = cache_sol

        # uncached solution
        for j in range(len(uncache_indices)):
            uncache_idx = uncache_indices[j]
            uncache_sol = sol_list_uncache[j]
            uncache_text = uncache_texts[j]
            sol_list[uncache_idx] = uncache_sol

            self.eval_cache[uncache_text] = uncache_sol

        return sol_list


    def _generate_uncache_solutions(self, list_of_sent_uncache, diff_lists_uncache):
        if self.setting == 'decision':
            return self._generate_uncache_decision(list_of_sent_uncache, diff_lists_uncache)
        else:
            return self._generate_uncache_score(list_of_sent_uncache, diff_lists_uncache)

    def _generate_uncache_decision(self, list_of_sent_uncache, diff_lists_uncache):
        sol_list_uncache = []

        sim_list_uncache = self._sematic_similarity(list_of_sent_uncache)
        success_list_uncache = self._attack_success(list_of_sent_uncache)

        for i in range(len(list_of_sent_uncache)):
            sent = list_of_sent_uncache[i]
            diff_list = diff_lists_uncache[i]
            sim = sim_list_uncache[i]
            success = success_list_uncache[i]

            if success:
                # print(f'Success on {self.qry_num}')
                self.is_success = True
                attack_score = MY_INF
            else:
                attack_score = len(diff_list)

            new_sol = Solution(sent, diff_list, attack_score, -len(diff_list), sim)

            if success:
                new_sol.mut_tag[0] = 0  # no add
                new_sol.attack_success = True
                self.reached.add(new_sol.text)

            sol_list_uncache.append(new_sol)

        return sol_list_uncache

    def _generate_uncache_score(self, list_of_sent_uncache, diff_lists_uncache):
        sol_list_uncache = []

        sim_list_uncache = self._sematic_similarity(list_of_sent_uncache)
        score_list_uncache, attack_success_list_uncache = self._attack_score(list_of_sent_uncache)

        for i in range(len(list_of_sent_uncache)):
            sent = list_of_sent_uncache[i]
            diff_list = diff_lists_uncache[i]
            sim = sim_list_uncache[i]
            attack_score = score_list_uncache[i]
            attack_success = attack_success_list_uncache[i]

            if attack_success:
                attack_score = MY_INF # step attack score
                self.is_success = True

            new_sol = Solution(sent, diff_list, attack_score, -len(diff_list), sim)

            if attack_success:
                new_sol.mut_tag[0] = 0  # no add
                new_sol.attack_success = True
                self.reached.add(new_sol.text)

            sol_list_uncache.append(new_sol)

        return sol_list_uncache

    # def _generate_uncache_target(self, list_of_sent_uncache, diff_lists_uncache):
    #     sol_list_uncache = []
    #
    #     sim_list_uncache = self._sematic_similarity(list_of_sent_uncache)
    #     score_list_uncache, attack_success_list_uncache = self._attack_score_targetted(list_of_sent_uncache)
    #
    #     for i in range(len(list_of_sent_uncache)):
    #         sent = list_of_sent_uncache[i]
    #         diff_list = diff_lists_uncache[i]
    #         sim = sim_list_uncache[i]
    #         attack_score = score_list_uncache[i]
    #         attack_success = attack_success_list_uncache[i]
    #
    #         if attack_success:
    #             attack_score = MY_INF # step attack score
    #             self.is_success = True
    #
    #         new_sol = Solution(sent, diff_list, attack_score, -len(diff_list), sim)
    #
    #         if attack_success:
    #             new_sol.mut_tag[0] = 0  # no add
    #             new_sol.attack_success = True
    #             self.reached.add(new_sol.text)
    #
    #         sol_list_uncache.append(new_sol)
    #
    #     return sol_list_uncache


    def _diff_list(self, list_of_sent):

        ret = []

        for _sent in list_of_sent:
            _diff_list = []
            for i, _ in self.idx_word_list:
                if _sent[i] != self.sent_orig[i]:
                    _diff_list.append((i, _sent[i]))

            ret.append(_diff_list)

        return ret

    def _sematic_similarity(self, list_of_sent):
        return calc_sim(self.sent_orig, list_of_sent, -1, self.sim_score_window, self.sim_predictor)

    def _attack_success(self, list_of_sent):

        self.real_qry_num += len(list_of_sent)

        if self.is_classification:
            new_probs = self.predictor(list_of_sent, batch_size=128)
        else:
            new_probs = self.predictor({'premises': [self.premise] * len(list_of_sent), 'hypotheses': list_of_sent})

        if self.goal_function == 'untarget':
            pr = (self.true_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
        else:
            pr = (self.target_label == torch.argmax(new_probs, dim=-1)).data.cpu().numpy()

        return pr

    def _attack_score(self, list_of_sent):

        # TODO: untargeted score
        self.real_qry_num += len(list_of_sent)

        if self.is_classification:
            new_probs = self.predictor(list_of_sent, batch_size=128)
        else:
            new_probs = self.predictor({'premises': [self.premise] * len(list_of_sent), 'hypotheses': list_of_sent})

        if self.goal_function == 'untarget':
            attack_success_list = (self.true_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            new_probs = new_probs.data.cpu().numpy()
            scores = 1 - new_probs[:, self.true_label]
        else:
            attack_success_list = (self.target_label == torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            new_probs = new_probs.data.cpu().numpy()
            scores = new_probs[:, self.target_label]


        return scores, attack_success_list

    # ====================================================
    def build_search_space(self, sol: Solution):
        add_op = 0
        swap_op = 1
        rm_op = 2

        if any(sol.search_space) == False: # not build yet
            sol.search_space = [[], [], []]
            sol.search_st = [0, 0, 0]
            # sol.reached_spaces = [[], [], []]


            diff_idx_set = set([i for i, w in sol.diff_list])

            for idx_in_sent, orig_word in self.idx_word_list:
                syno_key = (idx_in_sent, orig_word)

                if idx_in_sent in diff_idx_set:
                    swap_space = list(itertools.product([idx_in_sent], self.pos_syno_dict[syno_key]))
                    sol.search_space[swap_op].extend(swap_space)

                    sol.search_space[rm_op].append(idx_in_sent)
                else:
                    if not sol.attack_success:
                        add_space = list(itertools.product([idx_in_sent], self.pos_syno_dict[syno_key]))
                        sol.search_space[add_op].extend(add_space)

            for op_idx, _space in enumerate(sol.search_space):
                if len(_space) > 0:
                    random.shuffle(_space)
                    sol.reached_spaces[op_idx] = np.zeros(len(_space), dtype=bool)
                else:
                    sol.mut_tag[op_idx] = 0

    def _mutate_add(self, sol: Solution):
        op_idx = 0


        assert len(sol.search_space[op_idx]) > 0 and (not sol.attack_success)


        cur_space = sol.search_space[op_idx]
        cur_rnd_indices = list(range(len(cur_space)))
        random.shuffle(cur_rnd_indices)

        for add_idx in cur_rnd_indices:

            # construct new solution
            new_sent = sol.sent.copy()
            new_diff_list = sol.diff_list.copy()

            _add = cur_space[add_idx]
            change_idx = _add[0]
            new_syno = _add[1]


            new_sent[change_idx] = new_syno
            new_diff_list.append((change_idx, new_syno))

            # check if reached
            new_text = " ".join(new_sent)
            if new_text not in self.reached:
                break
        else:
            new_sent = None
            new_diff_list = None
            sol.mut_tag[op_idx] = 0

        return new_sent, new_diff_list

    def _mutate_swap(self, sol: Solution):
        op_idx = 1

        # assert sol.search_st[op_idx] is not None

        cur_space = sol.search_space[op_idx]
        cur_rnd_indices = list(range(len(cur_space)))
        random.shuffle(cur_rnd_indices)

        for swap_idx in cur_rnd_indices:
            # record best solution searched space
            if sol.attack_success:
                sol.reached_spaces[op_idx][swap_idx] = True

            # construct new solution
            new_sent = sol.sent.copy()
            new_diff_list = sol.diff_list.copy()

            _swap = cur_space[swap_idx]
            change_idx = _swap[0]
            new_syno = _swap[1]


            new_diff_list.remove((change_idx, sol.sent[change_idx]))
            new_diff_list.append((change_idx, new_syno))
            new_sent[change_idx] = new_syno

            # check if reached
            new_text = " ".join(new_sent)
            if new_text not in self.reached:
                break
        else:
            new_sent = None
            new_diff_list = None
            sol.mut_tag[op_idx] = 0


        return new_sent, new_diff_list

    def _mutate_rm(self, sol: Solution):
        op_idx = 2

        assert sol.search_st[op_idx] is not None

        cur_space = sol.search_space[op_idx]
        cur_rnd_indices = list(range(len(cur_space)))
        random.shuffle(cur_rnd_indices)

        for rm_idx in cur_rnd_indices:
            # record best solution searched space
            if sol.attack_success:
                sol.reached_spaces[op_idx][rm_idx] = True

            # construct new solution
            new_sent = sol.sent.copy()
            new_diff_list = sol.diff_list.copy()

            rm_idx = cur_space[rm_idx]

            new_diff_list.remove((rm_idx, sol.sent[rm_idx]))
            new_sent[rm_idx] = self.sent_orig[rm_idx]

            # check if reached
            new_text = " ".join(new_sent)
            if new_text not in self.reached:
                break
        else:
            new_sent = None
            new_diff_list = None
            sol.mut_tag[op_idx] = 0

        return new_sent, new_diff_list

