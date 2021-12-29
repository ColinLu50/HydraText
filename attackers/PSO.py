import numpy as np

from attackers.score_based_attacker import SolutionScoreBase, ScoreBasedAttacker

from local_models.sim_models import calc_sim



class PSOAttacker(ScoreBasedAttacker):
    def __init__(self,
                 predictor,
                 sim_predictor,
                 sim_score_window,
                 is_classification,
                 goal_function,
                 pop_size,
                 max_iters
                 ):

        super().__init__(
            predictor=predictor,
            sim_predictor=sim_predictor,
            sim_score_window=sim_score_window,
            is_classification=is_classification,
            goal_function=goal_function
        )

        # PSO parameters
        self.pop_size = pop_size
        self.max_iters = max_iters


    def _do_replace(self, sent, change_idx, change_w):
        new_sent = sent.copy()
        new_sent[change_idx] = change_w

        return new_sent


    def attack(self, idx_word_list, syno_dict, sent_orig, premise, true_label, target_label):

        self.init_attack(idx_word_list, syno_dict, sent_orig, premise, true_label, target_label)

        sol_init = self._init_sol()

        final_sol = self._pso_search(sol_init)
        print(f'Query Number: {self.qry_num} Real Query Number: {self.real_qry_num}')
        if final_sol:

            print(final_sol)
            print('Change rate:', self._count_change_ratio(final_sol))

            sent = final_sol.sent
            change_num = len(final_sol.diff_list)
            sim = calc_sim(self.sent_orig, [sent], -1, self.sim_score_window, self.sim_predictor)[0]
            converge = 1

            return final_sol.sent, change_num, sim, self.qry_num, self.real_qry_num, converge
        else:
            return None, 0, 0, self.qry_num, self.real_qry_num, 0


    def _init_sol(self):
        sol_init = self.evaluate_sents([self.sent_orig.copy()])[0]
        return sol_init

    def _pso_search(self, sol_init):
        perturb_number = len(self.idx_word_list)

        # form selecte prob
        select_probs = []
        for _k in self.syno_dict:
            _syno_list = self.syno_dict[_k]
            select_probs.append(min(len(_syno_list), 10))

        if np.sum(select_probs) == 0:
            return None

        select_probs = select_probs / np.sum(select_probs)

        pop = self._generate_population(sol_init, select_probs, self.pop_size)
        if pop is None:
            return None
        if self.is_success:
            assert len(pop) == 1
            return pop[0] # TODO:

        part_elites = pop[:]

        all_elite = pop[0]
        for _sol in pop:
            if _sol.attack_score > all_elite.attack_score:
                all_elite = _sol

        Omega_1 = 0.8
        Omega_2 = 0.2
        C1_origin = 0.8
        C2_origin = 0.2

        V = [np.random.uniform(-3, 3) for _ in range(self.pop_size)]
        V_P = [[V[t] for _ in range(perturb_number)] for t in range(self.pop_size)]

        # TODO: max iter
        for i in range(self.max_iters):
            Omega = (Omega_1 - Omega_2) * (self.max_iters - i) / self.max_iters + Omega_2
            C1 = C1_origin - i / self.max_iters * (C1_origin - C2_origin)
            C2 = C2_origin + i / self.max_iters * (C1_origin - C2_origin)

            new_sent_list = [_sol.sent.copy() for _sol in pop]
            for sol_id in range(self.pop_size):

                for dim in range(perturb_number):
                    V_P[sol_id][dim] = Omega * V_P[sol_id][dim] + (1 - Omega) * (
                            self._equal(pop[sol_id].sent[dim], part_elites[sol_id].sent[dim]) +
                            self._equal(pop[sol_id].sent[dim], all_elite.sent[dim])
                    )

                turn_prob = [self._sigmod(V_P[sol_id][d]) for d in range(perturb_number)]
                P1 = C1
                P2 = C2



                if np.random.uniform() < P1:
                    new_sent_list[sol_id] = self._turn(part_elites[sol_id], pop[sol_id], turn_prob, perturb_number)
                if np.random.uniform() < P2:
                    new_sent_list[sol_id] = self._turn(all_elite, pop[sol_id], turn_prob, perturb_number)

            pop = self.evaluate_sents(new_sent_list)

            # check attack success
            for _sol in pop:
                if _sol.attack_success:
                    return _sol

            new_pop = []
            # mutation
            for _sol in pop:
                change_ratio = self._count_change_ratio(_sol)
                p_change = 1 - 2 * change_ratio

                if np.random.uniform() < p_change:
                    tem = self._perturb(_sol, select_probs)
                    if tem is None:
                        return None
                    if tem.attack_success:
                        return tem
                    else:
                        new_pop.append(tem)
                else:
                    new_pop.append(_sol)
            pop = new_pop

            # udpate elite
            for _sol_idx in range(self.pop_size):
                new_sol = new_pop[_sol_idx]

                # update global best
                if new_sol.attack_score > all_elite.attack_score:
                    all_elite = new_sol

                # update part best
                if new_sol.attack_score > part_elites[_sol_idx].attack_score:
                    part_elites[_sol_idx] = new_sol


        return all_elite

    def _count_change_ratio(self, sol):
        change_ratio = len(sol.diff_list) / self.sent_len
        return change_ratio

    def _equal(self, a, b):
        if a == b:
            return -3
        else:
            return 3

    def _sigmod(self, n):
        return 1 / (1 + np.exp(-n))

    def _turn(self, sol1, sol2, prob, perturb_number):
        sent_new = sol2.sent.copy()
        for i in range(perturb_number):
            word_idx, orig_word = self.idx_word_list[i]
            if np.random.uniform() < prob[i]:
                sent_new[word_idx] = sol1.sent[word_idx]
        return sent_new

    def _generate_population(self, sol_init, select_probs, pop_size):
        pop = []
        for i in range(pop_size):
            cur_sol = self._perturb(sol_init, select_probs)
            if cur_sol is None:
                return None
            if cur_sol.attack_success:
                return [cur_sol]
            else:
                pop.append(cur_sol)
        return pop

    def _perturb(self, sol_cur: SolutionScoreBase, select_probs):
        # Pick a word that is not modified and is not UNK
        x_len = select_probs.shape[0]

        perturb_idx, orig_word = self.idx_word_list[np.random.choice(x_len, 1, p=select_probs)[0]]
        # sample a position which is not pertrubed if any
        while len(sol_cur.diff_list) < len(self.idx_word_list) and sol_cur.sent[perturb_idx] != orig_word:

            perturb_idx, orig_word = self.idx_word_list[np.random.choice(x_len, 1, p=select_probs)[0]]

        replace_list = self.syno_dict[(perturb_idx, orig_word)]
        return self._select_best_replacement(perturb_idx, sol_cur, replace_list)

    def _select_best_replacement(self, perturb_idx, sol_cur: SolutionScoreBase, replace_list):
        """ Select the most effective replacement to word at pos (pos)
        in (x_cur) between the words in replace_list """
        new_sent_list = [self._do_replace(sol_cur.sent, perturb_idx, w) if self.sent_orig[perturb_idx] != w else sol_cur.sent
                         for w in replace_list]

        new_sol_list = self.evaluate_sents(new_sent_list)

        cur_best_sol = sol_cur

        for _sol in new_sol_list:

            if _sol.attack_success:
                return _sol

            if _sol.attack_score > cur_best_sol.attack_score:
                cur_best_sol = _sol

        return cur_best_sol
