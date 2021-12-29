import torch
import numpy as np
import criteria
import random
from collections import defaultdict
from scipy.special import softmax

from local_models.sim_models import calc_sim

def rnd_mute_pos(idx, random_text, orig_pos_list, syn_list):
    pos_same = False
    shuffled_syn = syn_list[1:].copy()

    raw_w = random_text[idx]

    random.shuffle(shuffled_syn)

    for _new_word in shuffled_syn:
        # random select a word, replace it
        random_text[idx] = _new_word

        # filtered out the word has different pos
        if len(random_text) > 10:
            new_pos = criteria.get_pos(random_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
        else:
            new_pos = criteria.get_pos(random_text)[idx]

        pos_same = criteria.pos_filter(orig_pos_list[idx], [new_pos])[0]

        if pos_same:
            break

    if pos_same == False:
        # print('=============== Wrong POS! =================')
        random_text[idx] = raw_w


    return random_text


class NliPredictorCache(object):
    def __init__(self, predictor, is_target=False):
        self.predictor = predictor
        self.is_target = is_target

        self.cache_dict = {}
        self.real_qry_num = 0

        self.premise = None
        self.target_label = None


    def reset_state(self, premise, target=None):
        self.premise = premise

        if self.is_target:
            self.target_label = target
        else:
            assert target is None

        self.cache_dict = {}
        self.real_qry_num = 0



    def get_attack_result(self, new_word_lists, orig_label, batch_size):

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
            uncache_pr = self._get_attack_result_uncache(uncache_w_lists, orig_label)
            for i, _pr in enumerate(uncache_pr):
                _idx = uncache_indices[i]
                pr_final[_idx] = _pr

        pr_final = np.array(pr_final)

        return pr_final

    def _get_attack_result_uncache(self, uncache_word_lists, orig_label):
        uncache_outs = self.predictor({'premises': [self.premise] * len(uncache_word_lists), 'hypotheses': uncache_word_lists})
        if self.is_target:
            uncache_pr = (self.target_label == torch.argmax(uncache_outs, dim=-1)).data.cpu().numpy()
        else:
            uncache_pr = (orig_label != torch.argmax(uncache_outs, dim=-1)).data.cpu().numpy()

        self.real_qry_num += len(uncache_word_lists)

        for _idx, _w_list in enumerate(uncache_word_lists):
            _text = " ".join(_w_list)
            self.cache_dict[_text] = uncache_pr[_idx]

        return uncache_pr

    def raw_out(self, word_lists):
        return self.predictor({'premises': [self.premise] * len(word_lists), 'hypotheses': word_lists})

class CPredictorCache(object):
    def __init__(self, predictor, is_target=False):
        self.predictor = predictor
        self.is_target = is_target

        self.cache_dict = {}
        self.real_qry_num = 0

        self.target_label = None

    def reset_state(self, _no_use_premise, target=None):
        assert _no_use_premise is None

        if self.is_target:
            self.target_label = target
        else:
            assert target is None

        self.cache_dict = {}
        self.real_qry_num = 0


    def get_attack_result(self, new_word_lists, orig_label, batch_size):

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
            uncache_pr = self._get_attack_result_uncache(uncache_w_lists, orig_label, batch_size)
            for i, _pr in enumerate(uncache_pr):
                _idx = uncache_indices[i]
                pr_final[_idx] = _pr

        pr_final = np.array(pr_final)

        return pr_final

    def _get_attack_result_uncache(self, uncache_word_lists, orig_label, batch_size):
        uncache_outs = self.predictor(uncache_word_lists, batch_size=batch_size)

        if self.is_target:
            uncache_pr = (self.target_label == torch.argmax(uncache_outs, dim=-1)).data.cpu().numpy()
        else:
            uncache_pr = (orig_label != torch.argmax(uncache_outs, dim=-1)).data.cpu().numpy()

        self.real_qry_num += len(uncache_word_lists)

        for _idx, _w_list in enumerate(uncache_word_lists):
            _text = " ".join(_w_list)
            self.cache_dict[_text] = uncache_pr[_idx]

        return uncache_pr

    def raw_out(self, word_lists):
        return self.predictor(word_lists)

# It changes the inputt text at the specified index.
# rand_idx (int): Index to be mutated.
# text_ls (list): Original text.
# pos_ls (list): POS tage list.
# new_attack (list): The changed text during genetic optimization.
# best_attack (list): The best attack until now.
# remaining_indices (list): The indices in text input different from original input.
# synonyms_dict (dict): Synonym dict for each word.
# orig_label (int): Original prediction of the target model.
# sim_score_window (int): The number of words to consider around idx.
# predictor: Target model.
# sim_predictor: USE to compute semantic similarity.
# batch_size (int): batch size.
def mutate(rand_idx, text_ls, pos_ls, new_attack, best_attack, remaining_indices,
           synonyms_dict, old_syns, orig_label, sim_score_window,
           predictor_cache, sim_predictor, batch_size):
    # Calculates the semantic similarity before mutation.
    random_text = new_attack[:]
    syns = synonyms_dict[text_ls[rand_idx]]
    prev_semantic_sims = calc_sim(text_ls, [best_attack], rand_idx, sim_score_window, sim_predictor)

    # Gives Priority to Original Word
    orig_word = 0
    if random_text[rand_idx] != text_ls[rand_idx]:

        temp_text = random_text[:]
        temp_text[rand_idx] = text_ls[rand_idx]
        pr = predictor_cache.get_attack_result([temp_text], orig_label, batch_size)
        semantic_sims = calc_sim(text_ls, [temp_text], rand_idx, sim_score_window, sim_predictor)
        if np.sum(pr) > 0:
            orig_word = 1
            return temp_text, 1  # (updated_text, queries_taken)

    # If replacing with original word does not yield adversarial text, then try to replace with other synonyms.
    if orig_word == 0:
        final_mask = []
        new_texts = []
        final_texts = []

        # Replace with synonyms.
        for syn in syns:

            # Ignore the synonym already present at position rand_idx.
            if syn == best_attack[rand_idx]:
                final_mask.append(0)
            else:
                final_mask.append(1)
            temp_text = random_text[:]
            temp_text[rand_idx] = syn
            new_texts.append(temp_text[:])

        # Filter out mutated texts that: (1) are not having same POS tag of the synonym, (2) lowers Semantic Similarity and (3) Do not satisfy adversarial criteria.
        synonyms_pos_ls = [criteria.get_pos(new_text[max(rand_idx - 4, 0):rand_idx + 5])[min(4, rand_idx)]
                           if len(new_text) > 10 else criteria.get_pos(new_text)[rand_idx] for new_text in new_texts]
        pos_mask = np.array(criteria.pos_filter(pos_ls[rand_idx], synonyms_pos_ls))
        semantic_sims = calc_sim(text_ls, new_texts, rand_idx, sim_score_window, sim_predictor)
        pr = predictor_cache.get_attack_result(new_texts, orig_label, batch_size)
        final_mask = np.asarray(final_mask)
        sem_filter = semantic_sims >= prev_semantic_sims[0]
        prediction_filter = pr > 0
        final_mask = final_mask * sem_filter
        final_mask = final_mask * prediction_filter
        final_mask = final_mask * pos_mask
        sem_vals = final_mask * semantic_sims

        for i in range(len(sem_vals)):
            if sem_vals[i] > 0:
                final_texts.append((new_texts[i], sem_vals[i]))

        # Return mutated text with best semantic similarity.
        final_texts.sort(key=lambda x: x[1])
        final_texts.reverse()

        if len(final_texts) > 0:
            # old_syns[rand_idx].append(final_texts[0][0][rand_idx])
            return final_texts[0][0], len(new_texts)
        else:
            return [], len(new_texts)


# It generates children texts from the parent texts using crossover.
# population_size (int): Size of population used.
# population (list): The population currently in the optimization process.
# parent1_idx (int): The index of parent text input 1.
# parent2_idx (int): The index of parent text input 2.
# text_ls (list): Original text.
# best_attack (list): The best attack until now in the optimization.
# max_changes (int): The number of words substituted in the best_attack.
# changed_indices (list): The indices in text input different from original input.
# sim_score_window (int): The number of words to consider around idx.
# predictor: Target model.
# sim_predictor: USE to compute semantic similarity.
# orig_label (int): Original prediction of the target model.
# batch_size (int): batch size.
def crossover(population_size, population, parent1_idx, parent2_idx,
              text_ls, best_attack, max_changes, changed_indices,
              sim_score_window, sim_predictor,
              predictor_cache, orig_label, batch_size):
    childs = []
    changes = []

    # Do crossover till population_size-1.
    for i in range(population_size - 1):

        # Generates new child.
        p1 = population[parent1_idx[i]]
        p2 = population[parent2_idx[i]]
        assert len(p1) == len(p2)
        new_child = []
        for j in range(len(p1)):
            if np.random.uniform() < 0.5:
                new_child.append(p1[j])
            else:
                new_child.append(p2[j])
        change = 0
        cnt = 0
        mismatches = 0
        # Filter out crossover child which (1) Do not improve semantic similarity, (2) Have number of words substituted
        # more than the current best_attack.
        for k in range(len(changed_indices)):
            j = changed_indices[k]
            if new_child[j] == text_ls[j]:
                change += 1
                cnt += 1
            elif new_child[j] == best_attack[j]:
                change += 1
                cnt += 1
            elif new_child[j] != best_attack[j]:
                change += 1
                prev_semantic_sims = calc_sim(text_ls, [best_attack], j, sim_score_window, sim_predictor)
                semantic_sims = calc_sim(text_ls, [new_child], j, sim_score_window, sim_predictor)
                if semantic_sims[0] >= prev_semantic_sims[0]:
                    mismatches += 1
                    cnt += 1
        if cnt == change and mismatches <= max_changes:
            childs.append(new_child)
        changes.append(change)
    if len(childs) == 0:
        return [], 0

    # Filter out children which do not satisfy the adversarial criteria.
    pr = predictor_cache.get_attack_result(childs, orig_label, batch_size)
    final_childs = [childs[i] for i in range(len(pr)) if pr[i] > 0]
    return final_childs, len(final_childs)


def attack(fuzz_val, top_k_words, qrs, sample_index, text_ls, true_label,
           predictor_cache, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
           batch_size=32, classification_task=False):

    # first check the prediction of the original text
    orig_probs = predictor_cache.raw_out([text_ls]).squeeze()  # predictor(premise,hypothese).squeeze()
    orig_label = torch.argmax(orig_probs)
    # orig_prob = orig_probs.max()

    # if true_label != orig_label:
    #     return '', 0, 0, orig_label, orig_label, 0, 0, 0
    # else:
    len_text = len(text_ls)
    if len_text < sim_score_window:
        sim_score_threshold = 0.1  # shut down the similarity thresholding function
    half_sim_score_window = (sim_score_window - 1) // 2
    num_queries = 1
    rank = {}
    # get the pos and verb tense info
    words_perturb = []
    pos_ls = criteria.get_pos(text_ls)
    pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
    for pos in pos_pref:
        for i in range(len(pos_ls)):
            if pos_ls[i] == pos and len(text_ls[i]) > 2:
                words_perturb.append((i, text_ls[i]))

    random.shuffle(words_perturb)

    # find synonyms and make a dict of synonyms of each word.
    # words_perturb = words_perturb[:top_k_words]
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

    # STEP 1: Random initialisation.
    qrs = 0
    num_changed = 0
    flag = 0
    th = 0

    # Try substituting a random index with its random synonym.
    while qrs < len(text_ls):
        random_text = text_ls[:]
        for i in range(len(synonyms_all)):
            idx = synonyms_all[i][0]
            syn = synonyms_all[i][1]
            random_text = rnd_mute_pos(idx, random_text, pos_ls, syn)
            # random_text[idx] = random.choice(syn)
            if i >= th:
                break

        # pr = get_attack_result([random_text], predictor, orig_label, batch_size)
        pr = predictor_cache.get_attack_result([random_text], orig_label, batch_size)
        qrs += 1
        th += 1
        if th > len_text:
            break
        if np.sum(pr) > 0:
            flag = 1
            break
    old_qrs = qrs

    # If adversarial text is not yet generated try to substitute more words than 30%.
    while qrs < old_qrs + 2500 and flag == 0:
        random_text = text_ls[:]
        for j in range(len(synonyms_all)):
            idx = synonyms_all[j][0]
            syn = synonyms_all[j][1]
            random_text = rnd_mute_pos(idx, random_text, pos_ls, syn)
            if j >= len_text:
                break
        pr = predictor_cache.get_attack_result([random_text], orig_label, batch_size)
        qrs += 1
        if np.sum(pr) > 0:
            flag = 1
            break

    if flag == 1:
        # print("Found "+str(sample_index))
        changed = 0
        for i in range(len(text_ls)):
            if text_ls[i] != random_text[i]:
                changed += 1
        print(changed)

        # STEP 2: Search Space Reduction i.e.  Move Sample Close to Boundary
        while True:
            choices = []

            # For each word substituted in the original text, change it with its original word and compute
            # the change in semantic similarity.
            for i in range(len(text_ls)):
                if random_text[i] != text_ls[i]:
                    new_text = random_text[:]
                    new_text[i] = text_ls[i]
                    semantic_sims = calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)
                    qrs += 1
                    pr = predictor_cache.get_attack_result([new_text], orig_label, batch_size)
                    if np.sum(pr) > 0:
                        choices.append((i, semantic_sims[0]))

            # Sort the relacements by semantic similarity and replace back the words with their original
            # counterparts till text remains adversarial.
            if len(choices) > 0:
                choices.sort(key=lambda x: x[1])
                choices.reverse()
                for i in range(len(choices)):
                    new_text = random_text[:]
                    new_text[choices[i][0]] = text_ls[choices[i][0]]
                    pr = predictor_cache.get_attack_result([new_text], orig_label, batch_size)
                    qrs += 1
                    if pr[0] == 0:
                        break
                    random_text[choices[i][0]] = text_ls[choices[i][0]]

            if len(choices) == 0:
                break

        changed_indices = []
        num_changed = 0
        for i in range(len(text_ls)):
            if text_ls[i] != random_text[i]:
                changed_indices.append(i)
                num_changed += 1
        print(str(num_changed) + " " + str(qrs))
        random_sim = calc_sim(text_ls, [random_text], -1, sim_score_window, sim_predictor)[0]
        # return '', 0, orig_label, orig_label, 0
        if num_changed == 1:
            return ' '.join(random_text), 1, 1, \
                   orig_label, torch.argmax(predictor_cache.raw_out([random_text])), qrs, random_sim, random_sim
        population_size = 30
        population = []
        old_syns = {}
        if classification_task:
            max_replacements = defaultdict(int)
        # STEP 3: Genetic Optimization
        # Genertaes initial population by mutating the substituted indices.
        for i in range(len(changed_indices)):
            txt, mut_qrs = mutate(changed_indices[i], text_ls, pos_ls, random_text, random_text, changed_indices,
                                  synonyms_dict, old_syns, orig_label, sim_score_window,
                                  predictor_cache, sim_predictor, batch_size)
            qrs += mut_qrs
            if len(txt) != 0:
                population.append(txt)
        if classification_task:
            max_iters = 100
        else:
            max_iters = 1000
        pop_count = 0
        attack_same = 0
        old_best_attack = random_text[:]
        if len(population) == 0:
            return ' '.join(random_text), len(changed_indices), len(changed_indices), \
                   orig_label, torch.argmax(predictor_cache.raw_out([random_text])), qrs, random_sim, random_sim

        ## Genetic Optimization
        for _ in range(max_iters):
            max_changes = len_text

            # Find the best_attack text in the current population.
            for txt in population:
                changes = 0
                for i in range(len(changed_indices)):
                    j = changed_indices[i]
                    if txt[j] != text_ls[j]:
                        changes += 1
                if changes <= max_changes:
                    max_changes = changes
                    best_attack = txt

            # Check that it is adversarial.
            pr = predictor_cache.get_attack_result([best_attack], orig_label, batch_size)
            assert pr[0] > 0
            flag = 0

            # If the new best attack is the same as the old best attack for last 15 consecutive iterations tham
            # stop optimization.
            for i in range(len(changed_indices)):
                k = changed_indices[i]
                if best_attack[k] != old_best_attack[k]:
                    flag = 1
                    break
            if flag == 1:
                attack_same = 0
            else:
                attack_same += 1

            if attack_same >= 15:
                sim = calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)[0]
                return ' '.join(best_attack), max_changes, len(changed_indices), \
                       orig_label, torch.argmax(predictor_cache.raw_out([best_attack])), qrs, sim, random_sim

            old_best_attack = best_attack[:]

            # print(str(max_changes)+" After Genetic")

            # If only 1 input word substituted return it.
            if max_changes == 1:
                sim = calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)[0]
                return ' '.join(best_attack), max_changes, len(changed_indices), \
                       orig_label, torch.argmax(predictor_cache.raw_out([best_attack])), qrs, sim, random_sim

            # Sample two parent input propotional to semantic similarity.
            sem_scores = calc_sim(text_ls, population, -1, sim_score_window, sim_predictor)
            sem_scores = np.asarray(sem_scores)
            scrs = softmax(sem_scores)
            parent1_idx = np.random.choice(len(population), size=population_size - 1, p=scrs)
            parent2_idx = np.random.choice(len(population), size=population_size - 1, p=scrs)

            ## Crossover
            final_childs, cross_qrs = crossover(population_size, population, parent1_idx, parent2_idx,
                                                text_ls, best_attack, max_changes, changed_indices,
                                                sim_score_window, sim_predictor,
                                                predictor_cache, orig_label, batch_size)
            qrs += cross_qrs
            population = []
            indices_done = []

            # Randomly select indices for mutation from the changed indices. The changed indices contains indices
            # which has not been replaced by original word.
            indices = np.random.choice(len(changed_indices), size=min(len(changed_indices), len(final_childs)))
            for i in range(len(indices)):
                child = final_childs[i]
                j = indices[i]
                # If the index has been substituted no need to mutate.
                if text_ls[changed_indices[j]] == child[changed_indices[j]]:
                    population.append(child)
                    indices_done.append(j)
                    continue

                # Mutate the childs obtained after crossover on the random index.
                if classification_task:
                    txt = []
                    if max_replacements[changed_indices[j]] <= 25:
                        txt, mut_qrs = mutate(changed_indices[j], text_ls, pos_ls, child, child, changed_indices,
                                              synonyms_dict, old_syns, orig_label, sim_score_window,
                                              predictor_cache, sim_predictor, batch_size)
                else:
                    txt, mut_qrs = mutate(changed_indices[j], text_ls, pos_ls, child, child, changed_indices,
                                          synonyms_dict, old_syns, orig_label, sim_score_window,
                                          predictor_cache, sim_predictor, batch_size)
                qrs += mut_qrs
                indices_done.append(j)

                # If the input has been mutated successfully add to population for nest generation.
                if len(txt) != 0:
                    if classification_task:
                        max_replacements[changed_indices[j]] += 1
                    population.append(txt)
            if len(population) == 0:
                pop_count += 1
            else:
                pop_count = 0

            # If length of population is zero for 15 consecutive iterations return.
            if pop_count >= 15:
                sim = calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)[0]

                return ' '.join(best_attack), len(changed_indices), \
                       max_changes, orig_label, torch.argmax(predictor_cache.raw_out([best_attack])), qrs, sim, random_sim

            # Add best adversarial attack text also to next population.
            population.append(best_attack)
        sim = calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)[0]

        return ' '.join(best_attack), max_changes, len(changed_indices), \
               orig_label, torch.argmax(predictor_cache.raw_out([best_attack])), qrs, sim, random_sim

    else:
        # @ERROR:if attack fail, should return real query number instead of 0
        print("Not Found")
        return '', 0, 0, orig_label, orig_label, qrs, 0, 0