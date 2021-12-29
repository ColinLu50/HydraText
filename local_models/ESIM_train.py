"""
From https://github.com/coetaur0/ESIM
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import string
import numpy as np
import pickle
import time
import csv

import fnmatch
import json
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from local_models.esim.model import ESIM

from local_models.NLI_config import *

SNLI_FOLDER_PATH = 'path/to/snli_data_folder'
MNLI_FOLDER_PATH = 'path/to/mnli_data_folder'
GLOVE_EMBEDDING_PATH_64B300D = 'path/to/glove.840B.300d.txt'


class Preprocessor(object):
    """
    Preprocessor class for Natural Language Inference datasets.

    The class can be used to read NLI datasets, build worddicts for them
    and transform their premises, hypotheses and labels into lists of
    integer indices.
    """

    def __init__(self,
                 lowercase=False,
                 ignore_punctuation=False,
                 num_words=None,
                 stopwords=[]):
        """
        Args:
            lowercase: A boolean indicating whether the words in the datasets
                being preprocessed must be lowercased or not. Defaults to
                False.
            ignore_punctuation: A boolean indicating whether punctuation must
                be ignored or not in the datasets preprocessed by the object.
            num_words: An integer indicating the number of words to use in the
                worddict of the object. If set to None, all the words in the
                data are kept. Defaults to None.
            stopwords: A list of words that must be ignored when building the
                worddict for a dataset. Defaults to an empty list.
            bos: A string indicating the symbol to use for the 'beginning of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
            eos: A string indicating the symbol to use for the 'end of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
        """
        self.lowercase = lowercase
        self.ignore_punctuation = ignore_punctuation
        self.num_words = num_words
        self.stopwords = stopwords
        self.labeldict = NLI_LABEL_STR2NUM
        self.bos = BOS
        self.eos = EOS


    def read_data(self, filepath):
        """
        Read the premises, hypotheses and labels from some NLI dataset's
        file and return them in a dictionary. The file should be in the same
        form as SNLI's .txt files.

        Args:
            filepath: The path to a file containing some premises, hypotheses
                and labels that must be read. The file should be formatted in
                the same way as the SNLI (and MultiNLI) dataset.

        Returns:
            A dictionary containing three lists, one for the premises, one for
            the hypotheses, and one for the labels in the input data.
        """
        with open(filepath, "r", encoding="utf8") as input_data:
            ids, premises, hypotheses, labels = [], [], [], []

            # Translation tables to remove parentheses and punctuation from
            # strings.
            parentheses_table = str.maketrans({"(": None, ")": None})
            punct_table = str.maketrans({key: " "
                                         for key in string.punctuation})

            # Ignore the headers on the first line of the file.
            next(input_data)

            for line in input_data:
                line = line.strip().split("\t")

                # Ignore sentences that have no gold label.
                if line[0] == "-":
                    continue

                pair_id = line[7]
                premise = line[1]
                hypothesis = line[2]

                # Remove '(' and ')' from the premises and hypotheses.
                premise = premise.translate(parentheses_table)
                hypothesis = hypothesis.translate(parentheses_table)

                if self.lowercase:
                    premise = premise.lower()
                    hypothesis = hypothesis.lower()

                if self.ignore_punctuation:
                    premise = premise.translate(punct_table)
                    hypothesis = hypothesis.translate(punct_table)

                # Each premise and hypothesis is split into a list of words.
                premises.append([w for w in premise.rstrip().split()
                                 if w not in self.stopwords])
                hypotheses.append([w for w in hypothesis.rstrip().split()
                                   if w not in self.stopwords])
                labels.append(line[0])
                ids.append(pair_id)

            return {"ids": ids,
                    "premises": premises,
                    "hypotheses": hypotheses,
                    "labels": labels}

    def build_worddict(self, data):
        """
        Build a dictionary associating words to unique integer indices for
        some dataset. The worddict can then be used to transform the words
        in datasets to their indices.

        Args:
            data: A dictionary containing the premises, hypotheses and
                labels of some NLI dataset, in the format returned by the
                'read_data' method of the Preprocessor class.
        """
        words = []
        [words.extend(sentence) for sentence in data["premises"]]
        [words.extend(sentence) for sentence in data["hypotheses"]]

        counts = Counter(words)
        num_words = self.num_words
        if self.num_words is None:
            num_words = len(counts)

        self.worddict = {}

        # Special indices are used for padding, out-of-vocabulary words, and
        # beginning and end of sentence tokens.
        self.worddict[PAD] = 0
        self.worddict[OOV] = 1

        offset = 2
        if self.bos:
            self.worddict[BOS] = 2
            offset += 1
        if self.eos:
            self.worddict[EOS] = 3
            offset += 1

        for i, word in enumerate(counts.most_common(num_words)):
            self.worddict[word[0]] = i + offset

        if self.labeldict == {}:
            raise Exception('Wrong!')
            # label_names = set(data["labels"])
            # self.labeldict = {label_name: i
            #                   for i, label_name in enumerate(label_names)}

    def words_to_indices(self, sentence):
        """
        Transform the words in a sentence to their corresponding integer
        indices.

        Args:
            sentence: A list of words that must be transformed to indices.

        Returns:
            A list of indices.
        """
        indices = []
        # Include the beggining of sentence token at the start of the sentence
        # if one is defined.
        if self.bos:
            indices.append(self.worddict[BOS])

        for word in sentence:
            if word in self.worddict:
                index = self.worddict[word]
            else:
                # Words absent from 'worddict' are treated as a special
                # out-of-vocabulary word (OOV).
                index = self.worddict[OOV]
            indices.append(index)
        # Add the end of sentence token at the end of the sentence if one
        # is defined.
        if self.eos:
            indices.append(self.worddict[EOS])

        return indices

    def indices_to_words(self, indices):
        """
        Transform the indices in a list to their corresponding words in
        the object's worddict.

        Args:
            indices: A list of integer indices corresponding to words in
                the Preprocessor's worddict.

        Returns:
            A list of words.
        """
        return [list(self.worddict.keys())[list(self.worddict.values())
                                           .index(i)]
                for i in indices]

    def transform_to_indices(self, data):
        """
        Transform the words in the premises and hypotheses of a dataset, as
        well as their associated labels, to integer indices.

        Args:
            data: A dictionary containing lists of premises, hypotheses
                and labels, in the format returned by the 'read_data'
                method of the Preprocessor class.

        Returns:
            A dictionary containing the transformed premises, hypotheses and
            labels.
        """
        transformed_data = {"ids": [],
                            "premises": [],
                            "hypotheses": [],
                            "labels": []}

        for i, premise in enumerate(data["premises"]):
            # Ignore sentences that have a label for which no index was
            # defined in 'labeldict'.
            label = data["labels"][i]
            if label not in self.labeldict and label != "hidden":
                continue

            # TODO: modified
            # transformed_data["ids"].append(data["ids"][i])

            if label == "hidden":
                transformed_data["labels"].append(-1)
            else:
                transformed_data["labels"].append(self.labeldict[label])

            indices = self.words_to_indices(premise)
            transformed_data["premises"].append(indices)

            indices = self.words_to_indices(data["hypotheses"][i])
            transformed_data["hypotheses"].append(indices)

        return transformed_data

    def build_embedding_matrix(self, embeddings_file):
        """
        Build an embedding matrix with pretrained weights for object's
        worddict.

        Args:
            embeddings_file: A file containing pretrained word embeddings.

        Returns:
            A numpy matrix of size (num_words+n_special_tokens, embedding_dim)
            containing pretrained word embeddings (the +n_special_tokens is for
            the padding and out-of-vocabulary tokens, as well as BOS and EOS if
            they're used).
        """
        # Load the word embeddings in a dictionnary.
        embeddings = {}
        with open(embeddings_file, "r", encoding="utf8") as input_data:
            for line in input_data:
                line = line.split()

                try:
                    # Check that the second element on the line is the start
                    # of the embedding and not another word. Necessary to
                    # ignore multiple word lines.
                    float(line[1])
                    word = line[0]
                    if word in self.worddict:
                        embeddings[word] = line[1:]

                # Ignore lines corresponding to multiple words separated
                # by spaces.
                except ValueError:
                    continue

        num_words = len(self.worddict)
        embedding_dim = len(list(embeddings.values())[0])
        embedding_matrix = np.zeros((num_words, embedding_dim))

        # Actual building of the embedding matrix.
        missed = 0
        for word, i in self.worddict.items():
            if word in embeddings:
                embedding_matrix[i] = np.array(embeddings[word], dtype=float)
            else:
                if word == PAD:
                    continue
                missed += 1
                # Out of vocabulary words are initialised with random gaussian
                # samples.
                embedding_matrix[i] = np.random.normal(size=(embedding_dim))
        print("Missed words: ", missed)

        return embedding_matrix

class NLIDataset(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 data,
                 padding_idx=0,
                 max_premise_length=None,
                 max_hypothesis_length=None):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.premises_lengths = [len(seq) for seq in data["premises"]]
        self.max_premise_length = max_premise_length
        if self.max_premise_length is None:
            self.max_premise_length = max(self.premises_lengths)

        self.hypotheses_lengths = [len(seq) for seq in data["hypotheses"]]
        self.max_hypothesis_length = max_hypothesis_length
        if self.max_hypothesis_length is None:
            self.max_hypothesis_length = max(self.hypotheses_lengths)

        self.num_sequences = len(data["premises"])

        self.data = {"premises": torch.ones((self.num_sequences,
                                             self.max_premise_length),
                                            dtype=torch.long) * padding_idx,
                     "hypotheses": torch.ones((self.num_sequences,
                                               self.max_hypothesis_length),
                                              dtype=torch.long) * padding_idx,
                     "labels": torch.tensor(data["labels"], dtype=torch.long)}

        for i, premise in enumerate(data["premises"]):
            # self.data["ids"].append(data["ids"][i])
            end = min(len(premise), self.max_premise_length)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])

            hypothesis = data["hypotheses"][i]
            end = min(len(hypothesis), self.max_hypothesis_length)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {
            # "id": self.data["ids"][index],
                "premise": self.data["premises"][index],
                "premise_length": min(self.premises_lengths[index],
                                      self.max_premise_length),
                "hypothesis": self.data["hypotheses"][index],
                "hypothesis_length": min(self.hypotheses_lengths[index],
                                         self.max_hypothesis_length),
                "label": self.data["labels"][index]}

def preprocess_SNLI_data(output_dir):
    """
    Preprocess the data from the SNLI corpus so it can be used by the
    ESIM model.
    Compute a worddict from the train set, and transform the words in
    the sentences of the corpus to their indices, as well as the labels.
    Build an embedding matrix from pretrained word vectors.
    The preprocessed data is saved in pickled form in some target directory.

    Args:
        inputdir: The path to the directory containing the NLI corpus.
        embeddings_file: The path to the file containing the pretrained
            word vectors that must be used to build the embedding matrix.
        targetdir: The path to the directory where the preprocessed data
            must be saved.
        lowercase: Boolean value indicating whether to lowercase the premises
            and hypotheseses in the input data. Defautls to False.
        ignore_punctuation: Boolean value indicating whether to remove
            punctuation from the input data. Defaults to False.
        num_words: Integer value indicating the size of the vocabulary to use
            for the word embeddings. If set to None, all words are kept.
            Defaults to None.
        stopwords: A list of words that must be ignored when preprocessing
            the data. Defaults to an empty list.
        bos: A string indicating the symbol to use for beginning of sentence
            tokens. If set to None, bos tokens aren't used. Defaults to None.
        eos: A string indicating the symbol to use for end of sentence tokens.
            If set to None, eos tokens aren't used. Defaults to None.
    """
    inputdir = os.path.normpath(os.path.join(SNLI_FOLDER_PATH))
    embeddings_file = os.path.normpath(os.path.join(GLOVE_EMBEDDING_PATH_64B300D))
    targetdir = os.path.normpath(os.path.join(output_dir))
    lowercase = False
    ignore_punctuation = False
    num_words = None


    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    # Retrieve the train, dev and test data files from the dataset directory.
    train_file = ""
    dev_file = ""
    test_file = ""
    for file in os.listdir(inputdir):

        if fnmatch.fnmatch(file, "*_train.jsonl"):
            train_file = file
        elif fnmatch.fnmatch(file, "*_dev.jsonl"):
            dev_file = file
        elif fnmatch.fnmatch(file, "*_test.jsonl"):
            test_file = file

    # -------------------- Train data preprocessing -------------------- #
    preprocessor = Preprocessor(lowercase=lowercase,
                                ignore_punctuation=ignore_punctuation,
                                num_words=num_words)

    print(20*"=", " Preprocessing train set ", 20*"=")
    print("\t* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, train_file))

    print("\t* Computing worddict and saving it...")
    preprocessor.build_worddict(data)
    with open(os.path.join(targetdir, "word_dict.pkl"), "wb") as pkl_file:
        pickle.dump(preprocessor.worddict, pkl_file)

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = preprocessor.transform_to_indices(data)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "train_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Validation data preprocessing -------------------- #
    print(20*"=", " Preprocessing dev set ", 20*"=")
    print("\t* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, dev_file))

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = preprocessor.transform_to_indices(data)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "dev_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Test data preprocessing -------------------- #
    print(20*"=", " Preprocessing test set ", 20*"=")
    print("\t* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, test_file))

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = preprocessor.transform_to_indices(data)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "test_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Embeddings preprocessing -------------------- #
    print(20*"=", " Preprocessing embeddings ", 20*"=")
    print("\t* Building embedding matrix and saving it...")
    embed_matrix = preprocessor.build_embedding_matrix(embeddings_file)
    with open(os.path.join(targetdir, "embeddings.pkl"), "wb") as pkl_file:
        pickle.dump(embed_matrix, pkl_file)


def preprocess_MNLI_data(output_dir):
    """
    Preprocess the data from the MultiNLI corpus so it can be used by the
    ESIM model.
    Compute a worddict from the train set, and transform the words in
    the sentences of the corpus to their indices, as well as the labels.
    Build an embedding matrix from pretrained word vectors.
    The preprocessed data is saved in pickled form in some target directory.

    Args:
        inputdir: The path to the directory containing the NLI corpus.
        embeddings_file: The path to the file containing the pretrained
            word vectors that must be used to build the embedding matrix.
        targetdir: The path to the directory where the preprocessed data
            must be saved.
        lowercase: Boolean value indicating whether to lowercase the premises
            and hypotheseses in the input data. Defautls to False.
        ignore_punctuation: Boolean value indicating whether to remove
            punctuation from the input data. Defaults to False.
        num_words: Integer value indicating the size of the vocabulary to use
            for the word embeddings. If set to None, all words are kept.
            Defaults to None.
        stopwords: A list of words that must be ignored when preprocessing
            the data. Defaults to an empty list.
        bos: A string indicating the symbol to use for beginning of sentence
            tokens. If set to None, bos tokens aren't used. Defaults to None.
        eos: A string indicating the symbol to use for end of sentence tokens.
            If set to None, eos tokens aren't used. Defaults to None.
    """
    inputdir = MNLI_FOLDER_PATH

    embeddings_file = os.path.normpath(GLOVE_EMBEDDING_PATH_64B300D)
    targetdir = os.path.normpath(os.path.join(output_dir))
    lowercase = False
    ignore_punctuation = False
    num_words = None


    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    # Retrieve the train, dev and test data files from the dataset directory.
    train_file = ""
    matched_dev_file = ""
    mismatched_dev_file = ""
    matched_test_file = ""
    mismatched_test_file = ""
    # for file in os.listdir(inputdir):
    #     if fnmatch.fnmatch(file, "*_train.jsonl"):
    #         train_file = file
    #     elif fnmatch.fnmatch(file, "*_dev_matched.jsonl"):
    #         matched_dev_file = file
    #     elif fnmatch.fnmatch(file, "*_dev_mismatched.jsonl"):
    #         mismatched_dev_file = file
    #     elif fnmatch.fnmatch(file, "*_test_matched_unlabeled.jsonl"):
    #         matched_test_file = file
    #     elif fnmatch.fnmatch(file, "*_test_mismatched_unlabeled.jsonl"):
    #         mismatched_test_file = file

    for file in os.listdir(inputdir):
        if fnmatch.fnmatch(file, "*_train.txt"):
            train_file = file
        elif fnmatch.fnmatch(file, "*_dev_matched.txt"):
            matched_dev_file = file
        elif fnmatch.fnmatch(file, "*_dev_mismatched.txt"):
            mismatched_dev_file = file
        elif fnmatch.fnmatch(file, "*_test_matched_unlabeled.txt"):
            matched_test_file = file
        elif fnmatch.fnmatch(file, "*_test_mismatched_unlabeled.txt"):
            mismatched_test_file = file


    # -------------------- Train data preprocessing -------------------- #
    preprocessor = Preprocessor(lowercase=lowercase,
                                ignore_punctuation=ignore_punctuation,
                                num_words=num_words)

    print(20*"=", " Preprocessing train set ", 20*"=")
    print("\t* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, train_file))

    print("\t* Computing worddict and saving it...")
    preprocessor.build_worddict(data)
    with open(os.path.join(targetdir, "word_dict.pkl"), "wb") as pkl_file:
        pickle.dump(preprocessor.worddict, pkl_file)

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = preprocessor.transform_to_indices(data)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "train_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Validation data preprocessing -------------------- #
    print(20*"=", " Preprocessing dev sets ", 20*"=")
    print("\t* Reading matched dev data...")
    data = preprocessor.read_data(os.path.join(inputdir, matched_dev_file))

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = preprocessor.transform_to_indices(data)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "matched_dev_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    print("\t* Reading mismatched dev data...")
    data = preprocessor.read_data(os.path.join(inputdir, mismatched_dev_file))

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = preprocessor.transform_to_indices(data)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "mismatched_dev_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Test data preprocessing -------------------- #
    print(20*"=", " Preprocessing test sets ", 20*"=")
    print("\t* Reading matched test data...")
    data = preprocessor.read_data(os.path.join(inputdir, matched_test_file))

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = preprocessor.transform_to_indices(data)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "matched_test_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    print("\t* Reading mismatched test data...")
    data = preprocessor.read_data(os.path.join(inputdir, mismatched_test_file))

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = preprocessor.transform_to_indices(data)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "mismatched_test_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Embeddings preprocessing -------------------- #
    print(20*"=", " Preprocessing embeddings ", 20*"=")
    print("\t* Building embedding matrix and saving it...")
    embed_matrix = preprocessor.build_embedding_matrix(embeddings_file)
    with open(os.path.join(targetdir, "embeddings.pkl"), "wb") as pkl_file:
        pickle.dump(embed_matrix, pkl_file)


def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.

    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.

    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()

def train(model,
          dataloader,
          optimizer,
          criterion,
          epoch_number,
          max_gradient_norm):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.

    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        criterion: A loss criterion to use for training.
        epoch_number: The number of the epoch for which training is performed.
        max_gradient_norm: Max. norm for gradient norm clipping.

    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    """
    # Switch the model to train mode.
    model.train()
    device = model.device

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0

    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        batch_start = time.time()

        # Move input and output data to the GPU if it is used.
        premises = batch["premise"].to(device)
        premises_lengths = batch["premise_length"].to(device)
        hypotheses = batch["hypothesis"].to(device)
        hypotheses_lengths = batch["hypothesis_length"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits, probs = model(premises,
                              premises_lengths,
                              hypotheses,
                              hypotheses_lengths)
        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probs, labels)

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg/(batch_index+1),
                              running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)

    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, dataloader, criterion):
    """
    Compute the loss and accuracy of a model on some validation dataset.

    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is located.

    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()
    device = model.device

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            # Move input and output data to the GPU if one is used.
            premises = batch["premise"].to(device)
            premises_lengths = batch["premise_length"].to(device)
            hypotheses = batch["hypothesis"].to(device)
            hypotheses_lengths = batch["hypothesis_length"].to(device)
            labels = batch["label"].to(device)

            logits, probs = model(premises,
                                  premises_lengths,
                                  hypotheses,
                                  hypotheses_lengths)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            running_accuracy += correct_predictions(probs, labels)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))

    return epoch_time, epoch_loss, epoch_accuracy

# def train_snli_main(train_file,
#                valid_file,
#                embeddings_file,
#                target_dir,
#                hidden_size=300,
#                dropout=0.5,
#                num_classes=3,
#                epochs=64,
#                batch_size=32,
#                lr=0.0004,
#                patience=5,
#                max_grad_norm=10.0,
#                checkpoint=None):
#     """
#     Train the ESIM model on the SNLI dataset.
#
#     Args:
#         train_file: A path to some preprocessed data that must be used
#             to train the model.
#         valid_file: A path to some preprocessed data that must be used
#             to validate the model.
#         embeddings_file: A path to some preprocessed word embeddings that
#             must be used to initialise the model.
#         target_dir: The path to a directory where the trained model must
#             be saved.
#         hidden_size: The size of the hidden layers in the model. Defaults
#             to 300.
#         dropout: The dropout rate to use in the model. Defaults to 0.5.
#         num_classes: The number of classes in the output of the model.
#             Defaults to 3.
#         epochs: The maximum number of epochs for training. Defaults to 64.
#         batch_size: The size of the batches for training. Defaults to 32.
#         lr: The learning rate for the optimizer. Defaults to 0.0004.
#         patience: The patience to use for early stopping. Defaults to 5.
#         checkpoint: A checkpoint from which to continue training. If None,
#             training starts from scratch. Defaults to None.
#     """
#
#     best_save_path = os.path.join(target_dir, "best.pth.tar")
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     print(20 * "=", " Preparing for training ", 20 * "=")
#
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#
#     # -------------------- Data loading ------------------- #
#     print("\t* Loading training data...")
#     with open(train_file, "rb") as pkl:
#         train_data = NLIDataset(pickle.load(pkl))
#
#     train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
#
#     print("\t* Loading validation data...")
#     with open(valid_file, "rb") as pkl:
#         valid_data = NLIDataset(pickle.load(pkl))
#
#     valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)
#
#     # -------------------- Model definition ------------------- #
#     print("\t* Building model...")
#     with open(embeddings_file, "rb") as pkl:
#         embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float)\
#                      .to(device)
#
#     model = ESIM(embeddings.shape[0],
#                  embeddings.shape[1],
#                  hidden_size,
#                  embeddings=embeddings,
#                  dropout=dropout,
#                  num_classes=num_classes,
#                  device=device).to(device)
#
#     # -------------------- Preparation for training  ------------------- #
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                            mode="max",
#                                                            factor=0.5,
#                                                            patience=0)
#
#     best_score = 0.0
#     start_epoch = 1
#
#     # Data for loss curves plot.
#     epochs_count = []
#     train_losses = []
#     valid_losses = []
#
#     # Continuing training from a checkpoint if one was given as argument.
#     if checkpoint:
#         checkpoint = torch.load(checkpoint)
#         start_epoch = checkpoint["epoch"] + 1
#         best_score = checkpoint["best_score"]
#
#         print("\t* Training will continue on existing model from epoch {}..."
#               .format(start_epoch))
#
#         model.load_state_dict(checkpoint["model"])
#         optimizer.load_state_dict(checkpoint["optimizer"])
#         epochs_count = checkpoint["epochs_count"]
#         train_losses = checkpoint["train_losses"]
#         valid_losses = checkpoint["valid_losses"]
#
#     # Compute loss and accuracy before starting (or resuming) training.
#     _, valid_loss, valid_accuracy = validate(model,
#                                              valid_loader,
#                                              criterion)
#     print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%"
#           .format(valid_loss, (valid_accuracy*100)))
#
#     # -------------------- Training epochs ------------------- #
#     print("\n",
#           20 * "=",
#           "Training ESIM model on device: {}".format(device),
#           20 * "=")
#
#     patience_counter = 0
#     for epoch in range(start_epoch, epochs+1):
#         epochs_count.append(epoch)
#
#         print("* Training epoch {}:".format(epoch))
#         epoch_time, epoch_loss, epoch_accuracy = train(model,
#                                                        train_loader,
#                                                        optimizer,
#                                                        criterion,
#                                                        epoch,
#                                                        max_grad_norm)
#
#         train_losses.append(epoch_loss)
#         print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
#               .format(epoch_time, epoch_loss, (epoch_accuracy*100)))
#
#         print("* Validation for epoch {}:".format(epoch))
#         epoch_time, epoch_loss, epoch_accuracy = validate(model,
#                                                           valid_loader,
#                                                           criterion)
#
#         valid_losses.append(epoch_loss)
#         print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
#               .format(epoch_time, epoch_loss, (epoch_accuracy*100)))
#
#         # Update the optimizer's learning rate with the scheduler.
#         scheduler.step(epoch_accuracy)
#
#         # Early stopping on validation accuracy.
#         if epoch_accuracy < best_score:
#             patience_counter += 1
#         else:
#             best_score = epoch_accuracy
#             patience_counter = 0
#             # Save the best model. The optimizer is not saved to avoid having
#             # a checkpoint file that is too heavy to be shared. To resume
#             # training from the best model, use the 'esim_*.pth.tar'
#             # checkpoints instead.
#             torch.save({"epoch": epoch,
#                         "model": model.state_dict(),
#                         "best_score": best_score,
#                         "epochs_count": epochs_count,
#                         "train_losses": train_losses,
#                         "valid_losses": valid_losses},
#                        # os.path.join(target_dir, "best.pth.tar")
#                        best_save_path
#                        )
#
#             print(
#                 'Saving best checkpoint to {} at epoch {}'.format(os.path.abspath(best_save_path),
#                                                              epoch))
#
#         # # Save the model at each epoch.
#         # torch.save({"epoch": epoch,
#         #             "model": model.state_dict(),
#         #             "best_score": best_score,
#         #             "optimizer": optimizer.state_dict(),
#         #             "epochs_count": epochs_count,
#         #             "train_losses": train_losses,
#         #             "valid_losses": valid_losses},
#         #            os.path.join(target_dir, "esim_{}.pth.tar".format(epoch)))
#
#         if patience_counter >= patience:
#             print("-> Early stopping: patience limit reached, stopping...")
#             break
#
#     # save wrapper
#     ckpt = torch.load(best_save_path)
#     with open("../tmp/ESIM/snli/word_dict.pkl", 'rb') as f:
#         word_dict = pickle.load(f)
#
#     torch.save((ckpt, word_dict), os.path.join(target_dir, "snli"))
#
#     # # Plotting of the loss curves for the train and validation sets.
#     # plt.figure()
#     # plt.plot(epochs_count, train_losses, "-r")
#     # plt.plot(epochs_count, valid_losses, "-b")
#     # plt.xlabel("epoch")
#     # plt.ylabel("loss")
#     # plt.legend(["Training loss", "Validation loss"])
#     # plt.title("Cross entropy loss")
#     # plt.show()

def train_snli(outputdir):
    config = {
    "train_data": "../tmp/ESIM/snli/train_data.pkl",
    "valid_data": "../tmp/ESIM/snli/dev_data.pkl",
    "embeddings": "../tmp/ESIM/snli/embeddings.pkl",

    "target_dir": outputdir,

    "hidden_size": 300,
    "dropout": 0.5,
    "num_classes": 3,

    "epochs": 64, # 64
    "batch_size": 32, #32
    "lr": 0.0004,
    "patience": 5,
    "max_gradient_norm": 10.0
    }

    # train_snli_main(config["train_data"], config["valid_data"], config["embeddings"], config["target_dir"],
    #                 config["hidden_size"], config["dropout"], config["num_classes"], config["epochs"],
    #                 config["batch_size"], config["lr"], config["patience"], config["max_gradient_norm"], None)

    """
       Train the ESIM model on the SNLI dataset.

       Args:
           train_file: A path to some preprocessed data that must be used
               to train the model.
           valid_file: A path to some preprocessed data that must be used
               to validate the model.
           embeddings_file: A path to some preprocessed word embeddings that
               must be used to initialise the model.
           target_dir: The path to a directory where the trained model must
               be saved.
           hidden_size: The size of the hidden layers in the model. Defaults
               to 300.
           dropout: The dropout rate to use in the model. Defaults to 0.5.
           num_classes: The number of classes in the output of the model.
               Defaults to 3.
           epochs: The maximum number of epochs for training. Defaults to 64.
           batch_size: The size of the batches for training. Defaults to 32.
           lr: The learning rate for the optimizer. Defaults to 0.0004.
           patience: The patience to use for early stopping. Defaults to 5.
           checkpoint: A checkpoint from which to continue training. If None,
               training starts from scratch. Defaults to None.
       """
    train_file = config["train_data"]
    valid_file = config["valid_data"]
    embeddings_file = config["embeddings"]
    target_dir = config["target_dir"]
    hidden_size = config["hidden_size"]
    dropout = config["dropout"]
    num_classes = config["num_classes"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    patience = config["patience"]
    max_grad_norm = config["max_gradient_norm"]
    checkpoint = None


    best_save_path = os.path.join(target_dir, "snli_best.pth.tar")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for training ", 20 * "=")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    with open(train_file, "rb") as pkl:
        train_data = NLIDataset(pickle.load(pkl))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    print("\t* Loading validation data...")
    with open(valid_file, "rb") as pkl:
        valid_data = NLIDataset(pickle.load(pkl))

    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)

    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    with open(embeddings_file, "rb") as pkl:
        embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float) \
            .to(device)

    model = ESIM(embeddings.shape[0],
                 embeddings.shape[1],
                 hidden_size,
                 embeddings=embeddings,
                 dropout=dropout,
                 num_classes=num_classes,
                 device=device).to(device)

    # -------------------- Preparation for training  ------------------- #
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=0)

    best_score = 0.0
    start_epoch = 1

    # Data for loss curves plot.
    epochs_count = []
    train_losses = []
    valid_losses = []

    # Continuing training from a checkpoint if one was given as argument.
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]

    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy = validate(model,
                                             valid_loader,
                                             criterion)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%"
          .format(valid_loss, (valid_accuracy * 100)))

    # -------------------- Training epochs ------------------- #
    print("\n",
          20 * "=",
          "Training ESIM model on device: {}".format(device),
          20 * "=")

    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model,
                                                       train_loader,
                                                       optimizer,
                                                       criterion,
                                                       epoch,
                                                       max_grad_norm)

        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))

        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = validate(model,
                                                          valid_loader,
                                                          criterion)

        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)

        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            # Save the best model. The optimizer is not saved to avoid having
            # a checkpoint file that is too heavy to be shared. To resume
            # training from the best model, use the 'esim_*.pth.tar'
            # checkpoints instead.
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       # os.path.join(target_dir, "best.pth.tar")
                       best_save_path
                       )

            print(
                'Saving best checkpoint to {} at epoch {}'.format(os.path.abspath(best_save_path),
                                                                  epoch))

        # # Save the model at each epoch.
        # torch.save({"epoch": epoch,
        #             "model": model.state_dict(),
        #             "best_score": best_score,
        #             "optimizer": optimizer.state_dict(),
        #             "epochs_count": epochs_count,
        #             "train_losses": train_losses,
        #             "valid_losses": valid_losses},
        #            os.path.join(target_dir, "esim_{}.pth.tar".format(epoch)))

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break

    # save wrapper
    ckpt = torch.load(best_save_path)
    with open("../tmp/ESIM/snli/word_dict.pkl", 'rb') as f:
        word_dict = pickle.load(f)

    torch.save((ckpt, word_dict), os.path.join(target_dir, "snli"))

    # # Plotting of the loss curves for the train and validation sets.
    # plt.figure()
    # plt.plot(epochs_count, train_losses, "-r")
    # plt.plot(epochs_count, valid_losses, "-b")
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.legend(["Training loss", "Validation loss"])
    # plt.title("Cross entropy loss")
    # plt.show()

def train_mnli(outputdir):

    """
    Train the ESIM model on the SNLI dataset.

    Args:
        train_file: A path to some preprocessed data that must be used
            to train the model.
        valid_files: A dict containing the paths to the preprocessed matched
            and mismatched datasets that must be used to validate the model.
        embeddings_file: A path to some preprocessed word embeddings that
            must be used to initialise the model.
        target_dir: The path to a directory where the trained model must
            be saved.
        hidden_size: The size of the hidden layers in the model. Defaults
            to 300.
        dropout: The dropout rate to use in the model. Defaults to 0.5.
        num_classes: The number of classes in the output of the model.
            Defaults to 3.
        epochs: The maximum number of epochs for training. Defaults to 64.
        batch_size: The size of the batches for training. Defaults to 32.
        lr: The learning rate for the optimizer. Defaults to 0.0004.
        patience: The patience to use for early stopping. Defaults to 5.
        checkpoint: A checkpoint from which to continue training. If None,
            training starts from scratch. Defaults to None.
    """

    config = {
        "train_data": "../tmp/ESIM/mnli/train_data.pkl",
        "valid_data": {"matched": "../tmp/ESIM/mnli/matched_dev_data.pkl",
                       "mismatched": "../tmp/ESIM/mnli/mismatched_dev_data.pkl"},
        "embeddings": "../tmp/ESIM/mnli/embeddings.pkl",

        # "target_dir": outputdir,

        "hidden_size": 300,
        "dropout": 0.5,
        "num_classes": 3,

        "epochs": 64,
        "batch_size": 32,
        "lr": 0.0004,
        "patience": 5,
        "max_gradient_norm": 10.0
    }

    train_file = config["train_data"]
    valid_files = config["valid_data"]
    embeddings_file = config["embeddings"]
    target_dir = outputdir
    hidden_size = config["hidden_size"]
    dropout = config["dropout"]
    num_classes = config["num_classes"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    patience = config["patience"]
    max_grad_norm = config["max_gradient_norm"]
    checkpoint = None

    best_save_path = os.path.join(target_dir, "esim_best.pth.tar")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for training ", 20 * "=")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    with open(train_file, "rb") as pkl:
        train_data = NLIDataset(pickle.load(pkl))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    print("\t* Loading validation data...")
    with open(os.path.normpath(valid_files["matched"]), "rb") as pkl:
        matched_valid_data = NLIDataset(pickle.load(pkl))

    with open(os.path.normpath(valid_files["mismatched"]), "rb") as pkl:
        mismatched_valid_data = NLIDataset(pickle.load(pkl))

    matched_valid_loader = DataLoader(matched_valid_data,
                                      shuffle=False,
                                      batch_size=batch_size)
    mismatched_valid_loader = DataLoader(mismatched_valid_data,
                                         shuffle=False,
                                         batch_size=batch_size)

    # -------------------- Model definition ------------------- #
    print('\t* Building model...')
    with open(embeddings_file, "rb") as pkl:
        embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float)\
                     .to(device)

    model = ESIM(embeddings.shape[0],
                 embeddings.shape[1],
                 hidden_size,
                 embeddings=embeddings,
                 dropout=dropout,
                 num_classes=num_classes,
                 device=device).to(device)

    # -------------------- Preparation for training  ------------------- #
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=0)

    best_score = 0.0
    start_epoch = 1

    # Data for loss curves plot.
    epochs_count = []
    train_losses = []
    matched_valid_losses = []
    mismatched_valid_losses = []

    # Continuing training from a checkpoint if one was given as argument.
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        matched_valid_losses = checkpoint["match_valid_losses"]
        mismatched_valid_losses = checkpoint["mismatch_valid_losses"]

    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy = validate(model,
                                             matched_valid_loader,
                                             criterion)
    print("\t* Validation loss before training on matched data: {:.4f}, accuracy: {:.4f}%"
          .format(valid_loss, (valid_accuracy*100)))

    _, valid_loss, valid_accuracy = validate(model,
                                             mismatched_valid_loader,
                                             criterion)
    print("\t* Validation loss before training on mismatched data: {:.4f}, accuracy: {:.4f}%"
          .format(valid_loss, (valid_accuracy*100)))

    # -------------------- Training epochs ------------------- #
    print("\n",
          20 * "=",
          "Training ESIM model on device: {}".format(device),
          20 * "=")



    patience_counter = 0
    for epoch in range(start_epoch, epochs+1):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model,
                                                       train_loader,
                                                       optimizer,
                                                       criterion,
                                                       epoch,
                                                       max_grad_norm)

        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))

        print("* Validation for epoch {} on matched data:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = validate(model,
                                                          matched_valid_loader,
                                                          criterion)
        matched_valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))

        print("* Validation for epoch {} on mismatched data:".format(epoch))
        epoch_time, epoch_loss, mis_epoch_accuracy = validate(model,
                                                              mismatched_valid_loader,
                                                              criterion)
        mismatched_valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(epoch_time, epoch_loss, (mis_epoch_accuracy*100)))

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)

        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            # Save the best model. The optimizer is not saved to avoid having
            # a checkpoint file that is too heavy to be shared. To resume
            # training from the best model, use the 'esim_*.pth.tar'
            # checkpoints instead.
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "match_valid_losses": matched_valid_losses,
                        "mismatch_valid_losses": mismatched_valid_losses},
                       # os.path.join(target_dir, "esim_best.pth.tar")
                       best_save_path
                       )

            print(
                'Saving best checkpoint to {} at epoch {}'.format(os.path.abspath(best_save_path),
                                                             epoch))

        # # Save the model at each epoch.
        # torch.save({"epoch": epoch,
        #             "model": model.state_dict(),
        #             "best_score": best_score,
        #             "optimizer": optimizer.state_dict(),
        #             "epochs_count": epochs_count,
        #             "train_losses": train_losses,
        #             "match_valid_losses": matched_valid_losses,
        #             "mismatch_valid_losses": mismatched_valid_losses},
        #            os.path.join(target_dir, "esim_{}.pth.tar".format(epoch)))

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break

    # save wrapper
    ckpt = torch.load(best_save_path)
    with open("../tmp/ESIM/mnli/word_dict.pkl", 'rb') as f:
        word_dict = pickle.load(f)

    torch.save((ckpt, word_dict), os.path.join(target_dir, "esim"))



def test(model, dataloader):
    """
    Test the accuracy of a model on some labelled test dataset.

    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.

    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device

    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            batch_start = time.time()

            # Move input and output data to the GPU if one is used.
            premises = batch["premise"].to(device)
            premises_lengths = batch["premise_length"].to(device)
            hypotheses = batch["hypothesis"].to(device)
            hypotheses_lengths = batch["hypothesis_length"].to(device)
            labels = batch["label"].to(device)

            _, probs = model(premises,
                             premises_lengths,
                             hypotheses,
                             hypotheses_lengths)

            accuracy += correct_predictions(probs, labels)
            batch_time += time.time() - batch_start

    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))

    return batch_time, total_time, accuracy

def run_test_snli_orig():
    """
    Test the ESIM model with pretrained weights on some dataset.

    Args:
        test_file: The path to a file containing preprocessed NLI data.
        pretrained_file: The path to a checkpoint produced by the
            'train_model' script.
        vocab_size: The number of words in the vocabulary of the model
            being tested.
        embedding_dim: The size of the embeddings in the model.
        hidden_size: The size of the hidden layers in the model. Must match
            the size used during training. Defaults to 300.
        num_classes: The number of classes in the output of the model. Must
            match the value used during training. Defaults to 3.
        batch_size: The size of the batches used for testing. Defaults to 32.
    """
    test_file = '../tmp/ESIM/snli/test_data.pkl'
    pretrained_file = "/home/workspace/big_data/hardlabel/models/ESIM/snli" # '../tmp/ESIM/snli/models/best.pth.tar'
    batch_size = 128

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for testing ", 20 * "=")

    checkpoint = torch.load(pretrained_file)

    # Retrieving model parameters from checkpoint.
    vocab_size = checkpoint["model"]["_word_embedding.weight"].size(0)
    embedding_dim = checkpoint["model"]['_word_embedding.weight'].size(1)
    hidden_size = checkpoint["model"]["_projection.0.weight"].size(0)
    num_classes = checkpoint["model"]["_classification.4.weight"].size(0)

    print("\t* Loading test data...")
    with open(test_file, "rb") as pkl:
        test_data = NLIDataset(pickle.load(pkl))
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    print("\t* Building model...")
    model = ESIM(vocab_size,
                 embedding_dim,
                 hidden_size,
                 num_classes=num_classes,
                 device=device).to(device)

    model.load_state_dict(checkpoint["model"])

    print(20 * "=",
          " Testing ESIM model on device: {} ".format(device),
          20 * "=")
    batch_time, total_time, accuracy = test(model, test_loader)

    print("-> Average batch processing time: {:.4f}s, total test time:\
 {:.4f}s, accuracy: {:.4f}%".format(batch_time, total_time, (accuracy*100)))

def run_test_snli_my(pretrained_model_path):
    from local_models.ESIM_model_wrapper import ESIMWrapper

    from dataloader import read_orig_snli
    import nltk
    test_data_list = read_orig_snli(SNLI_FOLDER_PATH, 'test')
    format_data = {'premises': [], "hypotheses": []}
    label_list = []

    for i, _data in enumerate(test_data_list):
        label, premise, hypo = _data
        premise_ = " ".join(nltk.word_tokenize(premise))
        hypo_ = " ".join(nltk.word_tokenize(hypo))

        format_data['premises'].append(premise_.rstrip().split())
        format_data["hypotheses"].append(hypo_.rstrip().split())
        label_list.append(NLI_LABEL_STR2NUM[label])
    m = ESIMWrapper(pretrained_model_path)

    pred_list = m.text_pred(format_data).data.max(dim=1)[1]
    target_list = torch.LongTensor(label_list).cuda()
    c = pred_list.long().eq(target_list.data.long()).cpu().sum().item()

    acc = float(c) / len(target_list)
    print(f'Test accuracy: {acc:.2%}')

def predict(model, dataloader, labeldict):
    """
    Predict the labels of an unlabelled test set with a pretrained model.

    Args:
        model: The torch module which must be used to make predictions.
        dataloader: A DataLoader object to iterate over some dataset.
        labeldict: A dictionary associating labels to integer values.

    Returns:
        A dictionary associating pair ids to predicted labels.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device

    # Revert the labeldict to associate integers to labels.
    labels = {index: label for label, index in labeldict.items()}
    predictions = {}

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:

            # Move input and output data to the GPU if one is used.
            ids = batch["id"]
            premises = batch['premise'].to(device)
            premises_lengths = batch['premise_length'].to(device)
            hypotheses = batch['hypothesis'].to(device)
            hypotheses_lengths = batch['hypothesis_length'].to(device)

            _, probs = model(premises,
                             premises_lengths,
                             hypotheses,
                             hypotheses_lengths)

            _, preds = probs.max(dim=1)

            for i, pair_id in enumerate(ids):
                predictions[pair_id] = labels[int(preds[i])]

    return predictions

def run_test_mnli_my(pretrained_model_path):
    from local_models.ESIM_model_wrapper import ESIMWrapper

    m = ESIMWrapper(pretrained_model_path)

    from dataloader import read_orig_mnli
    import nltk
    dev_matched_data_list = read_orig_mnli(MNLI_FOLDER_PATH, 'dev_matched')
    format_data = {'premises': [], "hypotheses": []}
    label_list = []

    for i, _data in enumerate(dev_matched_data_list):
        label, premise, hypo, _ = _data
        premise_ = " ".join(nltk.word_tokenize(premise))
        hypo_ = " ".join(nltk.word_tokenize(hypo))

        format_data['premises'].append(premise_.rstrip().split())
        format_data["hypotheses"].append(hypo_.rstrip().split())
        label_list.append(NLI_LABEL_STR2NUM[label])
    # from dataloader import read_data_nli
    # format_data = read_data_nli("../data/snli")


    pred_list = m.text_pred(format_data).data.max(dim=1)[1]
    target_list = torch.LongTensor(label_list).cuda()
    c = pred_list.long().eq(target_list.data.long()).cpu().sum().item()

    acc = float(c) / len(target_list)
    print(f'Matched Dev Accuracy: {acc:.2%}')

    dev_mismatched_data_list = read_orig_mnli(MNLI_FOLDER_PATH, 'dev_mismatched')
    format_data = {'premises': [], "hypotheses": []}
    label_list = []

    for i, _data in enumerate(dev_mismatched_data_list):
        label, premise, hypo, _ = _data
        premise_ = " ".join(nltk.word_tokenize(premise))
        hypo_ = " ".join(nltk.word_tokenize(hypo))

        format_data['premises'].append(premise_.rstrip().split())
        format_data["hypotheses"].append(hypo_.rstrip().split())
        label_list.append(NLI_LABEL_STR2NUM[label])
    # from dataloader import read_data_nli
    # format_data = read_data_nli("../data/snli")


    pred_list = m.text_pred(format_data).data.max(dim=1)[1]
    target_list = torch.LongTensor(label_list).cuda()
    c = pred_list.long().eq(target_list.data.long()).cpu().sum().item()

    acc = float(c) / len(target_list)
    print(f'Mismatched Dev Accuracy: {acc:.2%}')

    # ============== generate test csv
    test_matched_data_list = read_orig_mnli(MNLI_FOLDER_PATH, 'test_matched')
    format_data = {'premises': [], "hypotheses": []}
    pairID_list = []

    for i, _data in enumerate(test_matched_data_list):
        _, premise, hypo, pair_id = _data
        premise_ = " ".join(nltk.word_tokenize(premise))
        hypo_ = " ".join(nltk.word_tokenize(hypo))

        format_data['premises'].append(premise_.rstrip().split())
        format_data["hypotheses"].append(hypo_.rstrip().split())
        pairID_list.append(pair_id)

    pred_list = m.text_pred(format_data).data.max(dim=1)[1].cpu().numpy()
    pred_str_list = [NLI_LABEL_NUM2STR[label_idx] for label_idx in pred_list]

    pred_results = zip(pairID_list, pred_str_list)

    save_path_1 = "../tmp/esim_mnli_matched.csv"

    with open(save_path_1, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        # write the header
        csvwriter.writerow(['pairID', 'gold_label'])

        # writing the data rows
        csvwriter.writerows(pred_results)

    print('Test matched prediction result save to', save_path_1)

    # mismatched test

    test_mismatched_data_list = read_orig_mnli(MNLI_FOLDER_PATH, 'test_mismatched')
    format_data = {'premises': [], "hypotheses": []}
    pairID_list = []

    for i, _data in enumerate(test_mismatched_data_list):
        _, premise, hypo, pair_id = _data
        premise_ = " ".join(nltk.word_tokenize(premise))
        hypo_ = " ".join(nltk.word_tokenize(hypo))

        format_data['premises'].append(premise_.rstrip().split())
        format_data["hypotheses"].append(hypo_.rstrip().split())
        pairID_list.append(pair_id)

    pred_list = m.text_pred(format_data).data.max(dim=1)[1].cpu().numpy()
    pred_str_list = [NLI_LABEL_NUM2STR[label_idx] for label_idx in pred_list]

    pred_results = zip(pairID_list, pred_str_list)

    save_path_2 = "../tmp/esim_mnli_mismatched.csv"

    with open(save_path_2, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        # write the header
        csvwriter.writerow(['pairID', 'gold_label'])

        # writing the data rows
        csvwriter.writerows(pred_results)

    print('Test mismatched prediction result save to', save_path_2)



if __name__ == '__main__':
    # SNLI
    output_dir_snli = ""
    preprocess_SNLI_data(output_dir_snli)
    train_snli(output_dir_snli)

    # MNLI
    output_dir_mnli = ''
    preprocess_MNLI_data(output_dir_mnli)
    train_mnli(output_dir_mnli)
