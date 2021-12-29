import torch
import numpy as np
import pickle

from attackers import GA_decision
from utils import my_file


class AttackWrapperDecision(object):

    def __init__(self, args, predictor, sim_predictor, classification_task, is_targeted_goal):
        self.args = args
        self.attacker = args.attacker
        self.predictor = predictor
        self.sim_predictor = sim_predictor
        self.qry_budget = args.qry_budget
        self.classification_task = classification_task
        self.is_targeted_goal = is_targeted_goal

        if args.attacker == 'GA':

            # prepare synonym extractor
            # build dictionary via the embedding file
            print("Building vocab...")
            idx2word = {}
            word2idx = {}
            sim_lis = []
            with open(args.counter_fitting_embeddings_path, 'r') as ifile:
                for line in ifile:
                    word = line.split()[0]
                    if word not in idx2word:
                        idx2word[len(idx2word)] = word
                        word2idx[word] = len(idx2word) - 1

            self.idx2word = idx2word
            self.word2idx = word2idx

            print("Building cos sim matrix...")
            if args.counter_fitting_cos_sim_path:
                # load pre-computed cosine similarity matrix if provided
                print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
                with open(args.counter_fitting_cos_sim_path, "rb") as fp:
                    sim_lis = pickle.load(fp)
            else:
                print('Start computing the cosine similarity matrix!')
                embeddings = []
                with open(args.counter_fitting_embeddings_path, 'r') as ifile:
                    for line in ifile:
                        embedding = [float(num) for num in line.strip().split()[1:]]
                        embeddings.append(embedding)
                embeddings = np.array(embeddings)
                print(embeddings.T.shape)
                norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = np.asarray(embeddings / norm, "float64")
                cos_sim = np.dot(embeddings, embeddings.T)
            print("Cos sim import finished!")
            self.sim_lis = sim_lis

            if classification_task:
                from attackers.GA_decision import CPredictorCache
                self.predictor_cache = CPredictorCache(predictor, is_targeted_goal)
            else:
                from attackers.GA_decision import NliPredictorCache
                self.predictor_cache = NliPredictorCache(predictor, is_targeted_goal)

        # load MO attacker
        elif args.attacker == 'MO':
            # load pre-pos
            self.preprocess_data_list = my_file.load_pkl(args.preprocess_path)
            # attacker
            from attackers.MO import MOSearchAttacker
            self.search_agent_MO = MOSearchAttacker(predictor, sim_predictor, args.sim_score_window, args.qry_budget,
                                                    is_classification=classification_task,
                                                    goal_function=args.goal_function,
                                                    setting='decision'
                                                    )

    def _predict(self, premise, orig_text):
        if self.classification_task:
            assert premise is None
            orig_probs = self.predictor([orig_text]).squeeze()
        else:
            orig_probs = self.predictor(
                {'premises': [premise], 'hypotheses': [orig_text]}).squeeze()  # predictor(premise,hypothese).squeeze()

        return orig_probs

    def feed_data(self, idx, premise, orig_text, true_label, target_label):

        # first check the prediction of the original text
        orig_probs = self._predict(premise, orig_text)
        orig_label = torch.argmax(orig_probs)
        if true_label != orig_label:
            return '', 0, 0, orig_label, orig_label, 0, 0, 0, 0, 0

        if self.attacker == 'MO':
            # target label no use in decision-based attack
            return self._run_MO(idx, premise, orig_text, true_label, orig_label, target_label)
        elif self.attacker == 'GA':
            return self._run_GA(premise, orig_text, true_label, target_label)
        # elif self.attacker == 'LS':




    def _run_MO(self, idx, premise, orig_text, true_label, orig_label, target_label):
        idx_word_pert_list = self.preprocess_data_list[idx][0]
        pos_syno_dict = self.preprocess_data_list[idx][1]
        text_ls = orig_text[:]

        best_sent, best_changed, best_sim, qrys, converge, \
        random_sim, random_change, real_qrys = self.search_agent_MO.attack(idx_word_pert_list,
                                                                           pos_syno_dict, text_ls, premise,
                                                                           true_label, target_label)
        # check is attack success
        if best_sent is None:
            return '', 0, random_change, orig_label, orig_label, \
                   qrys, 0, random_sim, converge, real_qrys
        else:
            return " ".join(best_sent), best_changed, random_change, \
                   orig_label, torch.argmax(self._predict(premise, best_sent)), \
                   qrys, best_sim, random_sim, converge, real_qrys


    def _run_GA(self, premise, orig_text, true_label, target_label):
        text_ls = orig_text[:]

        self.predictor_cache.reset_state(premise, target_label)

        res = GA_decision.attack(None, None, 0, None, text_ls, true_label,
                                 self.predictor_cache, None, self.word2idx, self.idx2word, self.sim_lis, self.sim_predictor,
                                 None, None, self.args.sim_score_window, None, self.args.batch_size, classification_task=self.classification_task)

        new_text, num_changed, random_changed, orig_label, \
        new_label, num_queries, sim, random_sim = res

        real_qry_num = self.predictor_cache.real_qry_num

        # format return
        return new_text, num_changed, random_changed, orig_label, \
        new_label, num_queries, sim, random_sim, 0, real_qry_num


