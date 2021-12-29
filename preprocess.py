import os

import argparse
import numpy as np



np.random.seed(1234)
import random

random.seed(1234)
# from fuzzywuzzy import fuzz
import tensorflow.compat.v1 as tf

# To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.compat.v1.disable_eager_execution()

from substitute_methods.syno50 import syno50_preprocess
from substitute_methods.sememe import sememe_preprocess
from substitute_methods.wordnet_NE import wordnet_NE_process
from substitute_methods.embedding_LM import build_dist_mat


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--target_dataset",
                        required=True,
                        type=str,
                        help="Dataset Name")
    parser.add_argument("--target_dataset_path",
                        type=str,
                        required=True,
                        help="Which dataset to attack.")
    parser.add_argument("--orig_dataset_path",
                        type=str,
                        required=True,
                        help="Original dataset folder path")
    parser.add_argument("--output_dir",
                        type=str,
                        default='./preprocess_data',
                        help="Which dataset to attack.")

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



    args = parser.parse_args()



    save_path = os.path.join(args.output_dir, "syno50")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    syno50_preprocess(save_path, args.target_dataset, args.target_dataset_path,
                      args.counter_fitting_embeddings_path, args.counter_fitting_cos_sim_path)

    save_path = os.path.join(args.output_dir, "sememe")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sememe_preprocess(save_path, args.target_dataset, args.target_dataset_path, args.orig_dataset_path)

    save_path = os.path.join(args.output_dir, "wordnet_NE")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    wordnet_NE_process(save_path, args.target_dataset, args.target_dataset_path)

    save_path = os.path.join(args.output_dir, "embedding_LM")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    build_dist_mat(save_path, args.target_dataset, args.orig_dataset_path, args.counter_fitting_embeddings_path)




if __name__ == "__main__":
    main()
