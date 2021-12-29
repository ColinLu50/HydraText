import torch
from abc import ABC

class SolutionScoreBase:
    def __init__(self, sent, diff_list, attack_score, attack_success, predict_prob):

        self.sent = sent
        self.diff_list = diff_list
        self.diff_indices = set()
        for w_i, w in self.diff_list:
            self.diff_indices.add(w_i)

        self.attack_score = attack_score
        self.attack_success = attack_success
        self.predict_prob = predict_prob

        self._text = None

    @property
    def text(self):
        if not self._text:
            self._text = " ".join(self.sent)
        return self._text

    def __str__(self):
        return f'Attack Success: {self.attack_success}, Attack Score: {self.attack_score}, Change Num: {len(self.diff_indices)}' \
               f'\nText: {self.text}' \
               f'\nDiff List: {self.diff_list}\n'


class ScoreBasedAttacker(ABC):

    def __init__(self,
                 predictor,
                 sim_predictor,
                 sim_score_window,
                 is_classification,
                 goal_function
                 ):
        self.predictor = predictor
        self.sim_predictor = sim_predictor
        self.sim_score_window = sim_score_window
        self.is_classification = is_classification
        self.goal_function = goal_function
        self.target_label = None

        print('************* Local Search ******************')
        print(f'No Init | Goal Function: ', goal_function)

        # =========== init when attacking ==================

        # == assign in each attack
        # for NLI task
        if not is_classification:
            self.premise = None

        self.sent_orig = None

        self.true_label = None

        self.idx_word_list = None

        self.syno_dict = None


        # == reset in each attack
        # record query number
        self.qry_num = None
        self.real_qry_num = None

        # debug
        self.debug = False

    def init_attack(self, idx_word_list, syno_dict, sent_orig, premise, true_label, target_label):
        # assign value
        self.idx_word_list = idx_word_list

        self.syno_dict = syno_dict

        self.sent_orig = sent_orig
        self.sent_len = len(sent_orig)

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
        # self.reached = set()
        self.is_success = False
        self.eval_cache = {}

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

        sol_list_uncache = []

        # generate uncached solution
        if len(uncache_indices) > 0:
            if diff_lists is None:
                diff_lists_uncache = self._diff_list(list_of_sent_uncache)

            score_list_uncache, attack_success_list_uncache, probs_uncache = self._attack_score(list_of_sent_uncache)

            for i in range(len(list_of_sent_uncache)):
                sent = list_of_sent_uncache[i]
                diff_list = diff_lists_uncache[i]
                attack_score = score_list_uncache[i]
                attack_success = attack_success_list_uncache[i]
                predict_prob = probs_uncache[i]

                if attack_success:
                    self.is_success = True

                new_sol = SolutionScoreBase(sent, diff_list, attack_score, attack_success, predict_prob)

                sol_list_uncache.append(new_sol)


        sol_list = [None for _ in range(len(list_of_sent))]

        # cached solution
        for j in range(len(cache_indices)):
            cache_idx = cache_indices[j]
            cache_text = cache_texts[j]
            cache_sol = self.eval_cache[cache_text]

            sol_list[cache_idx] = cache_sol

        # uncached solution
        for j in range(len(uncache_indices)):
            uncache_idx = uncache_indices[j]
            uncache_sol = sol_list_uncache[j]
            uncache_text = uncache_texts[j]
            sol_list[uncache_idx] = uncache_sol

            self.eval_cache[uncache_text] = uncache_sol

        return sol_list

    def _diff_list(self, list_of_sent):

        ret = []

        for _sent in list_of_sent:
            _diff_list = []
            for i, _ in self.idx_word_list:
                if _sent[i] != self.sent_orig[i]:
                    _diff_list.append((i, _sent[i]))

            ret.append(_diff_list)

        return ret

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
        elif self.goal_function == 'target':
            attack_success_list = (self.target_label == torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            new_probs = new_probs.data.cpu().numpy()
            scores = new_probs[:, self.target_label]
        else:
            raise Exception('Wrong goal function type:', self.goal_function)

        return scores, attack_success_list, new_probs