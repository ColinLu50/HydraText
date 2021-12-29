

from attackers.score_based_attacker import ScoreBasedAttacker

from local_models.sim_models import calc_sim
from scipy.special import softmax



class PWWSAttacker(ScoreBasedAttacker):
    def __init__(self,
                 predictor,
                 sim_predictor,
                 sim_score_window,
                 is_classification,
                 goal_function,
                 oov_str
                 ):

        super().__init__(
            predictor=predictor,
            sim_predictor=sim_predictor,
            sim_score_window=sim_score_window,
            is_classification=is_classification,
            goal_function=goal_function
        )
        self.unknown_str = oov_str


    def _do_replace(self, sent, change_idx, change_w):
        new_sent = sent.copy()
        new_sent[change_idx] = change_w

        return new_sent


    def attack(self, idx_word_list, syno_dict, sent_orig, premise, true_label, target_label):

        self.init_attack(idx_word_list, syno_dict, sent_orig, premise, true_label, target_label)

        final_sol = self._pwws_search()
        print(f'Query Number: {self.qry_num} Real Query Number: {self.real_qry_num}')
        if final_sol:
            sent = final_sol.sent
            change_num = len(final_sol.diff_list)
            sim = calc_sim(self.sent_orig, [sent], -1, self.sim_score_window, self.sim_predictor)[0]
            converge = 1

            # print
            print(final_sol)
            print(f'Change: {change_num}, Sim: {sim}')

            return final_sol.sent, change_num, sim, self.qry_num, self.real_qry_num, converge
        else:
            return None, 0, 0, self.qry_num, self.real_qry_num, 0


    def _evaluate_word_saliency(self, orig_sol):
        word_saliency_list = []

        without_word_sent_list = []
        for word_idx, orig_word in self.idx_word_list:
            # replace original word with 'unknown'
            without_word_sent = self.sent_orig.copy()
            without_word_sent[word_idx] = self.unknown_str

            without_word_sent_list.append(without_word_sent)

        without_word_sol_list = self.evaluate_sents(without_word_sent_list)

        for without_word_sol in without_word_sol_list:
            # word_saliency = orig_prob[self.true_label] - without_word_sol.predict_prob[self.true_label]
            word_saliency = self.heuristic_fn(orig_sol, without_word_sol)
            word_saliency_list.append(word_saliency)

        word_saliency_list = softmax(word_saliency_list)

        return word_saliency_list


    def heuristic_fn(self, orig_sol, new_sol):

        return orig_sol.predict_prob[self.true_label] - new_sol.predict_prob[self.true_label]



    def _pwws_search(self):

        if len(self.idx_word_list) == 0:
            return None

        orig_sol = self.evaluate_sents([self.sent_orig])[0]

        word_saliency_list = self._evaluate_word_saliency(orig_sol)

        substitute_tuple_list = []

        for i in range(len(word_saliency_list)):

            word_idx, orig_word = self.idx_word_list[i]
            word_saliency = word_saliency_list[i]

            cur_sub_words = self.syno_dict[(word_idx, orig_word)]

            cur_new_sent_list = []
            for _sub_word in cur_sub_words:
                cur_new_sent_list.append(self._do_replace(self.sent_orig, word_idx, _sub_word))

            cur_new_sol_list = self.evaluate_sents(cur_new_sent_list)

            cur_best_sol_idx = -1
            cur_best_delta_p_star = -9999
            for sol_idx, new_sol in enumerate(cur_new_sol_list):
                cur_del_p_star = self.heuristic_fn(orig_sol, new_sol)
                if cur_del_p_star > cur_best_delta_p_star:
                    cur_best_sol_idx = sol_idx
                    cur_best_delta_p_star = cur_del_p_star

            assert cur_best_sol_idx >= 0

            cur_best_sub_word = cur_sub_words[cur_best_sol_idx]
            H_score = cur_best_delta_p_star * word_saliency

            substitute_tuple_list.append((word_idx, cur_best_sub_word, H_score))

        substitute_tuple_list.sort(key=lambda x: x[-1], reverse=True)

        if len(substitute_tuple_list) == 0:
            print(self.evaluate_sents([self.sent_orig]))
            print(self.syno_dict)
            raise Exception('Wrong !!!!')

        perturb_sent = self.sent_orig.copy()
        for change_idx, sub_word, H_score in substitute_tuple_list:
            perturb_sent = self._do_replace(perturb_sent, change_idx, sub_word)

            cur_sol = self.evaluate_sents([perturb_sent])[0]

            if cur_sol.attack_success:
                return cur_sol

        return None




