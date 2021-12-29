import torch
import numpy as np
import pickle

from attackers import textfooler
from utils import my_file

UNKNOWN_DICT = {
    'bert': '[UNK]',
    'wordLSTM': '<oov>',
    'wordCNN': '<oov>',
    'esim': '<oov>',
    'infersent': '<oov>'
}


class AttackWrapperScore(object):

    def __init__(self, args, predictor, sim_predictor, classification_task, is_targeted_goal):
        self.args = args
        self.attacker = args.attacker
        self.predictor = predictor
        self.sim_predictor = sim_predictor
        self.qry_budget = args.qry_budget
        self.classification_task = classification_task
        self.is_targeted_goal = is_targeted_goal

        if args.attacker == 'GA':
            if args.sub_method != 'embedding_LM':
                raise Exception('GA now only support original substitute methods')

            from attackers.GA_score import GAScoreAttackerOrig
            from substitute_methods import embedding_LM
            from local_models.google_lm import LM

            # hyper paras
            n1 = 8
            n2 = 4
            pop_size = 60
            max_iters = 20
            use_suffix = False

            # for substitute
            goog_lm = LM()
            dict_, inv_dict, full_dict, inv_full_dict, max_vocab_size, dist_mat \
                = embedding_LM.load_generated_files('preprocess_data/' + args.sub_method, args.target_dataset)

            self.attacker_GA_orig = GAScoreAttackerOrig(predictor, sim_predictor, args.sim_score_window,
                                                        classification_task, args.goal_function,
                                                        word2emb=dict_,
                                                        emb2word=inv_dict,
                                                        dist_matrix=dist_mat,
                                                        pop_size=pop_size,
                                                        max_iters=max_iters,
                                                        lm=goog_lm,
                                                        top_n1=n1,
                                                        top_n2=n2,
                                                        use_suffix=use_suffix
                                                        )

        # MO attacker
        elif args.attacker == 'MO':
            # load pre-pos
            self.preprocess_data_list = my_file.load_pkl(args.preprocess_path)

            # attacker
            from attackers.MO import MOSearchAttacker
            self.search_agent_MO = MOSearchAttacker(predictor, sim_predictor, args.sim_score_window, args.qry_budget,
                                                    is_classification=classification_task, goal_function=args.goal_function)
        elif args.attacker == 'LS': # TODO: remove this
            # load pre-pos
            self.preprocess_data_list = my_file.load_pkl(args.preprocess_path)
            # attacker
            from attackers.LS import LocalSearchAttacker
            self.search_agent_LS = LocalSearchAttacker(predictor, sim_predictor, args.sim_score_window,
                                                       is_classification=classification_task, goal_function=args.goal_function)

        elif args.attacker == 'PSO':
            self.preprocess_data_list = my_file.load_pkl(args.preprocess_path)
            max_iter = 20
            pop_size = 60

            from attackers.PSO import PSOAttacker
            self.attacker_PSO = PSOAttacker(predictor, sim_predictor, args.sim_score_window,
                                            is_classification=classification_task, goal_function=args.goal_function,
                                            max_iters=max_iter, pop_size=pop_size)

        elif args.attacker == 'PWWS':
            self.preprocess_data_list = my_file.load_pkl(args.preprocess_path)
            oov_str = UNKNOWN_DICT[args.target_model]
            from attackers.PWWS import PWWSAttacker
            self.attacker_PWWS = PWWSAttacker(predictor, sim_predictor, args.sim_score_window,
                                            is_classification=classification_task, goal_function=args.goal_function,
                                            oov_str=oov_str)

        elif args.attacker == 'TF': # textfooler
            if args.sub_method != 'syno50':
                raise Exception('TextFooler only support original substitute methods now')

            # prepare synonym extractor
            # build dictionary via the embedding file
            idx2word = {}
            word2idx = {}

            print("Building vocab...")
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
                cos_sim = np.load(args.counter_fitting_cos_sim_path)
            else:
                # calculate the cosine similarity matrix
                print('Start computing the cosine similarity matrix!')
                embeddings = []
                with open(args.counter_fitting_embeddings_path, 'r') as ifile:
                    for line in ifile:
                        embedding = [float(num) for num in line.strip().split()[1:]]
                        embeddings.append(embedding)
                embeddings = np.array(embeddings)
                product = np.dot(embeddings, embeddings.T)
                norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
                cos_sim = product / np.dot(norm, norm.T)
            print("Cos sim import finished!")

            self.cos_sim = cos_sim

            from attackers.textfooler import PredictorCache
            self.predictor_cache = PredictorCache(self.predictor, is_classification=self.classification_task)

            self.oov_str = UNKNOWN_DICT[args.target_model]
            from criteria import get_stopwords
            self.stop_words = get_stopwords()

            self.sim_score_threshold = 0.7
            self.syno_num = 50



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

        if self.attacker[:2] == 'MO':
            return self._run_MO(idx, premise, orig_text, true_label, target_label, orig_label)
        elif self.attacker == 'GA':
            return self._run_GA(idx, premise, orig_text, true_label, target_label, orig_label)
        elif self.attacker == 'LS':
            return self._run_LS(idx, premise, orig_text, true_label, target_label, orig_label)
        elif self.attacker == 'PSO':
            return self._run_PSO(idx, premise, orig_text, true_label, target_label, orig_label)
        elif self.attacker == 'PWWS':
            return self._run_PWWS(idx, premise, orig_text, true_label, target_label, orig_label)
        elif self.attacker == 'TF':
            return self._run_textfooler(premise, orig_text, true_label, target_label)



    def _run_MO(self, idx, premise, orig_text, true_label, target_label, orig_label):
        idx_word_pert_list = self.preprocess_data_list[idx][0]
        sub_words_dict = self.preprocess_data_list[idx][1]
        text_ls = orig_text[:]

        best_sent, best_changed, best_sim, qrys, converge, \
        random_sim, random_change, real_qrys = self.search_agent_MO.attack(idx_word_pert_list,
                                                                           sub_words_dict, text_ls, premise,
                                                                           true_label, target_label)
        # check is attack success
        if best_sent is None:
            return '', 0, random_change, orig_label, orig_label, \
                   qrys, 0, random_sim, converge, real_qrys
        else:
            return " ".join(best_sent), best_changed, random_change, \
                   orig_label, torch.argmax(self._predict(premise, best_sent)), \
                   qrys, best_sim, random_sim, converge, real_qrys


    def _run_GA(self, idx, premise, orig_text, true_label, target_label, orig_label):
        # idx_word_pert_list = self.preprocess_data_list[idx][0]
        # sub_words_dict = self.preprocess_data_list[idx][1]
        orig_text = orig_text[:]

        best_sent, best_changed, best_sim, qrys, real_qrys, converge = \
            self.attacker_GA_orig.attack(orig_text, premise, true_label, target_label)

        random_change = 0
        random_sim = 1

        # check is attack success
        if best_sent is None:
            return '', 0, random_change, orig_label, orig_label, \
                   qrys, 0, random_sim, converge, real_qrys
        else:
            return " ".join(best_sent), best_changed, random_change, \
                   orig_label, torch.argmax(self._predict(premise, best_sent)), \
                   qrys, best_sim, random_sim, converge, real_qrys

    def _run_LS(self, idx, premise, orig_text, true_label, target_label, orig_label):
        # TODO: remove this algorithm
        idx_word_pert_list = self.preprocess_data_list[idx][0]
        sub_words_dict = self.preprocess_data_list[idx][1]
        text_ls = orig_text[:]


        best_sent, best_changed, best_sim, qrys, converge, \
        random_sim, random_change, real_qrys = self.search_agent_LS.attack(idx_word_pert_list,
                                                                           sub_words_dict, text_ls, premise,
                                                                           true_label, target_label)
        # check is attack success
        if best_sent is None:
            return '', 0, random_change, orig_label, orig_label, \
                   qrys, 0, random_sim, converge, real_qrys
        else:
            return " ".join(best_sent), best_changed, random_change, \
                   orig_label, torch.argmax(self._predict(premise, best_sent)), \
                   qrys, best_sim, random_sim, converge, real_qrys


    def _run_PSO(self, idx, premise, orig_text, true_label, target_label, orig_label):
        # TODO:
        idx_word_pert_list = self.preprocess_data_list[idx][0]
        sub_words_dict = self.preprocess_data_list[idx][1]
        orig_text = orig_text[:]


        best_sent, best_changed, best_sim, qrys, \
        real_qrys, converge = self.attacker_PSO.attack(idx_word_pert_list,
                                                                           sub_words_dict, orig_text, premise,
                                                                           true_label, target_label)

        random_change = 0
        random_sim = 1
        # check is attack success
        if best_sent is None:
            return '', 0, random_change, orig_label, orig_label, \
                   qrys, 0, random_sim, converge, real_qrys
        else:
            return " ".join(best_sent), best_changed, random_change, \
                   orig_label, torch.argmax(self._predict(premise, best_sent)), \
                   qrys, best_sim, random_sim, converge, real_qrys


    def _run_PWWS(self, idx, premise, orig_text, true_label, target_label, orig_label):
        # TODO:
        idx_word_pert_list = self.preprocess_data_list[idx][0]
        sub_words_dict = self.preprocess_data_list[idx][1]
        orig_text = orig_text[:]


        best_sent, best_changed, best_sim, qrys, real_qrys, converge =\
            self.attacker_PWWS.attack(idx_word_pert_list, sub_words_dict,
                                                        orig_text, premise, true_label, target_label)

        random_change = 0
        random_sim = 1

        # check is attack success
        if best_sent is None:
            return '', 0, random_change, orig_label, orig_label, \
                   qrys, 0, random_sim, converge, real_qrys
        else:
            return " ".join(best_sent), best_changed, random_change, \
                   orig_label, torch.argmax(self._predict(premise, best_sent)), \
                   qrys, best_sim, random_sim, converge, real_qrys

    def _run_textfooler(self, premise, orig_text, true_label, target_label):

        self.predictor_cache.reset_state(premise)
        text_ls = orig_text[:]


        new_text, num_changed, orig_label, \
        new_label, num_queries, sim = textfooler.attack(self.oov_str, text_ls, true_label, target_label, self.predictor_cache, self.stop_words,
                                        self.word2idx, self.idx2word, self.cos_sim, sim_predictor=self.sim_predictor,
                                        sim_score_threshold=self.sim_score_threshold,
                                        sim_score_window=self.args.sim_score_window,
                                        synonym_num=self.syno_num,
                                        batch_size=self.args.batch_size,
                                        is_targeted_goal=self.is_targeted_goal)

        real_qry_num = self.predictor_cache.real_qry_num

        random_changed = 0
        random_sim = 1

        return new_text, num_changed, random_changed, orig_label, \
        new_label, num_queries, sim, random_sim, 0, real_qry_num

