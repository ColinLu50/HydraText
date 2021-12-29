from attackers.score_based_attacker import ScoreBasedAttacker, SolutionScoreBase
import numpy as np

from local_models.sim_models import calc_sim
from utils import glove_utils


class GAScoreAttackerOrig(ScoreBasedAttacker):
    def __init__(self,
                 predictor,
                 sim_predictor,
                 sim_score_window,
                 is_classification,
                 goal_function,
                 word2emb,
                 emb2word,
                 dist_matrix,
                 lm,
                 pop_size,
                 max_iters,
                 top_n1,
                 top_n2,
                 use_suffix
                 ):

        super().__init__(
            predictor=predictor,
            sim_predictor=sim_predictor,
            sim_score_window=sim_score_window,
            is_classification=is_classification,
            goal_function=goal_function
        )

        self.lm = lm
        self.word2emb = word2emb
        self.emb2word = emb2word
        self.dist_matrix = dist_matrix
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.top_n1 = top_n1  # similar words
        self.top_n2 = top_n2
        self.use_suffix = use_suffix
        self.temp = 0.3




    def _do_replace(self, sent, change_idx, change_w):
        new_sent = sent.copy()
        new_sent[change_idx] = change_w

        return new_sent


    def attack(self, sent_orig, premise, true_label, target_label):

        all_idx_word_list = [(i, w) for i, w in enumerate(sent_orig)]

        self.init_attack(all_idx_word_list, None, sent_orig, premise, true_label, target_label)

        final_sol = self._GA_search()
        print(f'Query Number: {self.qry_num} Real Query Number: {self.real_qry_num}')
        if final_sol:
            sent = final_sol.sent
            change_num = len(final_sol.diff_list)
            sim = calc_sim(self.sent_orig, [sent], -1, self.sim_score_window, self.sim_predictor)[0]
            converge = 1

            # print
            print(final_sol)
            print(f'Change Rate: {change_num / len(self.sent_orig):.2%}, Sim: {sim}')

            return final_sol.sent, change_num, sim, self.qry_num, self.real_qry_num, converge
        else:
            return None, 0, 0, self.qry_num, self.real_qry_num, 0


    def _GA_search(self):

        # form substitute words
        adv_sent = self.sent_orig.copy()

        x_len = len(adv_sent)
        # Neigbhours for every word.
        neigbhours_list = [self._pick_most_similar_words(self.sent_orig[i], 50, 0.5) for i in range(x_len)]
        neighbours_len = [len(x) for x in neigbhours_list]
        for i in range(x_len):
            if (adv_sent[i] not in self.word2emb) or (self.word2emb[adv_sent[i]] < 27):
                # To prevent replacement of words like 'the', 'a', 'of', etc.
                neighbours_len[i] = 0

        if np.sum(neighbours_len) == 0:
            return None

        w_select_probs = neighbours_len / np.sum(neighbours_len)
        neigbhours_list = [self._pick_most_similar_words(self.sent_orig[i], self.top_n1, 0.5) for i in range(x_len)]

        # init
        sol_init = self.evaluate_sents([self.sent_orig])[0]
        sol_adv = sol_init

        pop = self._generate_population(sol_init, neigbhours_list, None, w_select_probs, self.pop_size)

        for i in range(self.max_iters):
            # print(i)
            pop_scores = np.array([_sol.attack_score for _sol in pop])
            print('\t\t', i, ' -- ', np.max(pop_scores))

            pop_ranks = np.argsort(pop_scores)[::-1]
            top_attack = pop_ranks[0]

            logits = np.exp(pop_scores / self.temp)
            select_probs = logits / np.sum(logits)

            if pop[top_attack].attack_success:
                return pop[top_attack]

            elite = [pop[top_attack]]  # elite
            # print(select_probs.shape)
            parent1_idx = np.random.choice(self.pop_size, size=self.pop_size - 1, p=select_probs)
            parent2_idx = np.random.choice(self.pop_size, size=self.pop_size - 1, p=select_probs)

            child_sent_list = [self._crossover(pop[parent1_idx[i]], pop[parent2_idx[i]])
                      for i in range(self.pop_size - 1)]

            child_sol_list = self.evaluate_sents(child_sent_list)

            childs = [self._perturb(child_sol, neigbhours_list, None, w_select_probs)
                for child_sol in child_sol_list]

            pop = elite + childs

        return None

    def _generate_population(self, sol_orig, neigbhours_list, _neighbours_dist, w_select_probs, pop_size):
        return [self._perturb(sol_orig, neigbhours_list, _neighbours_dist, w_select_probs) for _ in range(pop_size)]

    def _perturb(self, sol_cur: SolutionScoreBase, neigbhours, no_use_neighbours_dist,  w_select_probs):

        # Pick a word that is not modified yet and have substitute words
        x_len = w_select_probs.shape[0]
        rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        while len(sol_cur.diff_list) < np.sum(np.sign(w_select_probs)) and sol_cur.sent[rand_idx] != self.sent_orig[rand_idx]:
            # The conition above has a quick hack to prevent getting stuck in infinite loop while processing too short examples
            # and all words `excluding articles` have been already replaced and still no-successful attack found.
            # a more elegent way to handle this could be done in attack to abort early based on the status of all population members
            # or to improve select_best_replacement by making it schocastic.
            rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]

        # src_word = x_cur[rand_idx]
        # replace_list,_ =  glove_utils.pick_most_similar_words(src_word, self.dist_mat, self.top_n, 0.5)
        replace_list = neigbhours[rand_idx]
        return self._select_best_replacement(rand_idx, sol_cur, replace_list)

    def _select_best_replacement(self, change_idx, sol_cur, replace_list):
        """ Select the most effective replacement to word at pos (pos)
        in (x_cur) between the words in replace_list """
        # TODO: remove condition x != 0, what is embedding

        sent_cur = sol_cur.sent
        new_sent_list = [self._do_replace(sent_cur, change_idx, sub_w)
                      if self.sent_orig[change_idx] != sub_w else sent_cur
                      for sub_w in replace_list]

        new_sol_list = self.evaluate_sents(new_sent_list)
        orig_sol = self.evaluate_sents([self.sent_orig])[0]

        new_scores = [_sol.attack_score - orig_sol.attack_score for _sol in new_sol_list]
        new_scores = np.array(new_scores)
        new_scores[self.top_n1:] = -10000000

        # LM evaluate
        prefix = ""
        suffix = None
        if change_idx > 0:
            prefix = sent_cur[change_idx - 1]

        orig_word = self.sent_orig[change_idx]
        if self.use_suffix and change_idx + 1 < len(sent_cur):
            suffix = sent_cur[change_idx + 1]

        replace_words_and_orig = replace_list[:self.top_n1] + [orig_word]
        replace_words_lm_scores = self.lm.get_words_probs(prefix, replace_words_and_orig, suffix)

        # select words
        new_words_lm_scores = np.array(replace_words_lm_scores[:-1])
        rank_replaces_by_lm = np.argsort(-new_words_lm_scores)
        filtered_words_idx = rank_replaces_by_lm[self.top_n2:]

        new_scores[filtered_words_idx] = -10000000

        if (np.max(new_scores) > 0):
            return new_sol_list[np.argsort(new_scores)[-1]]
        return sol_cur


    def _crossover(self, sol1, sol2):
        sent_new = sol1.sent.copy()
        for word_idx, orig_word in self.idx_word_list:
            if np.random.uniform() < 0.5:
                sent_new[word_idx] = sol2.sent[word_idx]
        return sent_new

    def _pick_most_similar_words(self, orig_word, ret_count, threshold):

        if orig_word not in self.word2emb:
            return []

        orig_embedding = self.word2emb[orig_word]
        sub_word_emb_list, _dist_no_use = \
            glove_utils.pick_most_similar_words(orig_embedding, self.dist_matrix, ret_count, threshold)
        sub_word_list = [self.emb2word[_emb] for _emb in sub_word_emb_list]

        return sub_word_list


