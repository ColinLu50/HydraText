import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# np.random.seed(1234)
np.random.seed(3334)
import random

random.seed(1223)
import csv
# from fuzzywuzzy import fuzz
import tensorflow.compat.v1 as tf

# To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.compat.v1.disable_eager_execution()

from dataloader import read_data_nli, read_nli_target
from local_models.NLI_config import NLI_LABEL_NUM2STR


# from local_models.nli_models import NLI_infer_InferSent, NLI_infer_BERT, NLI_infer_ESIM
from local_models.sim_models import USE


def get_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--attacker",
                        choices=['MO', 'GA', 'PSO', 'PWWS', 'TF'],
                        required=True,
                        type=str,
                        help="attacker")

    # substitute method
    parser.add_argument("--sub_method",
                        choices=['syno50', 'sememe', 'wordnet_NE', 'embedding_LM'],
                        required=True,
                        type=str,
                        help="Substitute method.")


    # dataset
    parser.add_argument("--target_dataset",
                        default="imdb",
                        type=str,
                        help="Dataset Name")
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True,
                        help="Which dataset to attack.")
    parser.add_argument("--preprocess_path",
                        type=str,
                        required=True,
                        help="Preprocessed data path.")

    parser.add_argument("--target_model",
                        type=str,
                        required=True,
                        choices=['infersent', 'esim', 'bert'],
                        help="Target models for text classification: fasttext, charcnn, word level lstm "
                             "For NLI: InferSent, ESIM, bert-base-uncased")

    parser.add_argument("--target_model_path",
                        type=str,
                        required=True,
                        help="pre-trained target model path")

    # goal function
    parser.add_argument("--goal_function",
                        choices=['decision', 'untarget', 'target'],
                        default='decision',
                        help="Goal function of attacking.")
    # attack setting
    parser.add_argument("--setting",
                        choices=['decision', 'score'],
                        default='decision',
                        help="Attack setting: decision-based or score-based")

    # target label file path
    parser.add_argument("--target_label_path",
                        type=str,
                        help="Path to the file of target labels")

    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default='',
                        help="path to the word embeddings for the target model")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        default="counter-fitted-vectors.txt",
                        help="path to the counter-fitting embeddings we used to find synonyms")
    parser.add_argument("--counter_fitting_cos_sim_path",
                        type=str,
                        default='',
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
    parser.add_argument("--USE_cache_path",
                        type=str,
                        # required=True,
                        help="Path to the USE encoder cache.")
    parser.add_argument("--output_dir",
                        type=str,
                        default='out/results/',
                        help="The output directory where the attack results will be written.")

    ## Model hyperparameters
    parser.add_argument("--sim_score_window",
                        default=40,
                        type=int,
                        help="Text length or token number to compute the semantic similarity score")
    parser.add_argument("--batch_size",
                        default=128,
                        type=int,
                        help="Batch size to get prediction")
    parser.add_argument("--attack_number",
                        default=5000,
                        type=int,
                        help="Data size to create adversaries")

    parser.add_argument("--qry_budget",
                        default=100000,
                        type=int,
                        help="Allowerd qrs")

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    # dataset
    # victim model
    if args.target_model == 'bert':
        args.max_seq_length = 256

    labeldict = NLI_LABEL_NUM2STR

    folder_path = os.path.join(args.output_dir, args.goal_function + '_' + args.setting,
                               args.target_dataset, args.target_model, args.attacker + '_' + args.sub_method)
    log_file = folder_path + "/log.txt"
    result_file = folder_path + "/results_final.csv"

    log_file_path = Path(log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # get data to attack
    data = read_data_nli(args.dataset_path)
    print("Data import finished!")

    # get the target label to attack
    if args.goal_function == 'target':
        target_labels = read_nli_target(args.target_label_path)

    # construct the model
    print("Building Model...")
    print('Load from', args.target_model_path)
    if args.target_model == 'esim':
        from local_models.ESIM_model_wrapper import ESIMWrapper
        model = ESIMWrapper(args.target_model_path, args.batch_size)
    elif args.target_model == 'infersent':
        from local_models.infersent_model import InfersentWrapper
        model = InfersentWrapper(args.target_model_path, args.batch_size)
    else:
        from local_models.nli_bert import NLI_infer_BERT
        model = NLI_infer_BERT(args.target_model_path)
    predictor = model.text_pred
    print("Model built!")

    # build the semantic similarity module
    use = USE(args.USE_cache_path)

    # ===================== init attacker ===================================
    if args.goal_function == 'target':
        is_targeted_goal = True
    elif args.goal_function == 'untarget':
        is_targeted_goal = False

    if args.setting == 'decision':
        from attack_wrapper.attack_wrapper_decision_based import AttackWrapperDecision
        attacker = AttackWrapperDecision(args, predictor, use, classification_task=False, is_targeted_goal=is_targeted_goal)
    else:
        from attack_wrapper.attack_wrapper_score_based import AttackWrapperScore
        attacker = AttackWrapperScore(args, predictor, use, classification_task=False, is_targeted_goal=is_targeted_goal)


    # start attacking
    # new
    attack_success = 0
    attack_number = 0
    all_number = 0

    # old
    orig_failures = 0.
    adv_failures = 0.
    changed_rates = []
    nums_queries = []
    orig_texts = []
    adv_texts = []
    true_labels = []
    new_labels = []
    wrds = []
    s_queries = []
    f_queries = []
    success = []
    results = []
    fails = []
    final_sims = []
    random_sims = []
    random_changed_rates = []
    real_qry_num_list = []

    converg_list = []
    change_num_list = []

    print('Start attacking!')

    test_t1 = time.time()


    for idx in range(len(data['premises'])):
        premise, hypothesis, true_label = data['premises'][idx], data['hypotheses'][idx], data['labels'][idx]
    # for idx, premise in enumerate(data['premises']):
    #     hypothesis, true_label = data['hypotheses'][idx], data['labels'][idx]

        target_label = None
        if args.goal_function == 'target':
            target_label = target_labels[idx]
            assert target_label != true_label
            if true_label == 0:
                assert target_label == 1
            elif true_label == 1:
                assert target_label == 0


        new_text, num_changed, random_changed, orig_label, \
        new_label, num_queries, sim, random_sim, is_converge, real_qry_num = attacker.feed_data(idx, premise, hypothesis, true_label, target_label)

        changed_rate = 1.0 * num_changed / len(hypothesis)
        random_changed_rate = 1.0 * random_changed / len(hypothesis)


        # if original failure
        if true_label != orig_label:
            orig_failures += 1
        else:
            nums_queries.append(num_queries)
            real_qry_num_list.append(real_qry_num)
            attack_number += 1
            converg_list.append(is_converge)

            if args.goal_function == 'untarget' and (new_label != orig_label) and changed_rate <= 0.25:
                attack_success += 1
            elif args.goal_function == 'target' and (new_label == target_label) and changed_rate <= 0.25:
                attack_success += 1

        if true_label != new_label:
            adv_failures += 1




        if args.goal_function == 'untarget':
            _success = true_label != new_label
        else:
            _success = target_label == new_label

        if true_label == orig_label and _success:

            orig_label = orig_label.to('cpu').numpy()[()]
            new_label = new_label.to('cpu').numpy()[()]

            temp = []
            temp.append(idx)
            temp.append(orig_label)
            temp.append(new_label)
            temp.append(labeldict[orig_label])
            temp.append(labeldict[new_label])
            temp.append(' '.join(premise))
            temp.append(' '.join(hypothesis))
            temp.append(new_text)
            temp.append(num_queries)
            temp.append(changed_rate * 100)
            temp.append(sim)
            temp.append(random_changed_rate * 100)
            temp.append(random_sim)
            temp.append(is_converge)
            temp.append(num_changed)
            temp.append(real_qry_num)

            results.append(temp)

        # filter out change rate > 25%
        if true_label == orig_label and _success and changed_rate <= 0.25:
            s_queries.append(num_queries)
            success.append(idx)
            changed_rates.append(changed_rate)
            orig_texts.append(' '.join(hypothesis))
            adv_texts.append(new_text)
            true_labels.append(true_label)
            new_labels.append(new_label)
            # random_changed_rates.append(random_changed_rate)
            # random_sims.append(random_sim)
            final_sims.append(sim)
            change_num_list.append(num_changed)

        all_number += 1

        tmp_t = time.time()
        if attack_number > 0:
            print(f'Attack {idx} end: Avg Time {(tmp_t - test_t1) / attack_number}, total time {tmp_t - test_t1}')
            print('=' * 100)

        if attack_number == args.attack_number:
            break

        sys.stdout.flush()

    message = f'Target Model: {args.target_model}\n' \
                          f'Dataset: {args.target_dataset}\n' \
                          f'Original Accuracy: {1 - orig_failures / all_number:.2%}\n' \
                          f'Attack Success Rate: {attack_success}/{attack_number} = {attack_success / attack_number:.2%}\n' \
                          f'Avg Change Rate: {np.mean(changed_rates):.2%}\n' \
                          f'Avg Change Num: {np.mean(change_num_list):.2f}\n' \
                          f'Avg Query Num: {np.mean(nums_queries):.1f}\n' \
                          f'Avg Real Query Num: {np.mean(real_qry_num_list):.1f}\n' \
                          f'Avg Similarity: {np.mean(final_sims):.3f}\n' \
                          f'Converge Rate: {np.mean(converg_list):.2%}\n'

    print(message)
    # print(orig_failures)
    sys.stdout.flush()

    # write logs
    log = open(log_file, 'w')
    log.write(message)

    with open(result_file, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # write the header
        csvwriter.writerow(['idx', 'orig label num', 'new label num', 'orig label', 'new label',
                            'premise', 'orig hypothesis', 'new hypothesis', 'query number',
                            'change rate', 'similarity', 'random change rate', 'random similarity', 'converge', 'change number', 'real query number'])

        # writing the data rows
        csvwriter.writerows(results)



if __name__ == "__main__":
    main()
