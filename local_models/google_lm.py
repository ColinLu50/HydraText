import os
import sys

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()
from google.protobuf import text_format





GOOGLE_LM_DIR = 'path/to/google_lm'


import random

import numpy as np
import tensorflow as tf

# lm_data_utils.py

class Vocabulary(object):
  """Class that holds a vocabulary for the dataset."""

  def __init__(self, filename):
    """Initialize vocabulary.
    Args:
      filename: Vocabulary file name.
    """

    self._id_to_word = []
    self._word_to_id = {}
    self._unk = -1
    self._bos = -1
    self._eos = -1

    with tf.compat.v1.gfile.Open(filename) as f:
      idx = 0
      for line in f:
        word_name = line.strip()
        if word_name == '<S>':
          self._bos = idx
        elif word_name == '</S>':
          self._eos = idx
        elif word_name == 'UNK':
          self._unk = idx
        if word_name == '!!!MAXTERMID':
          continue

        self._id_to_word.append(word_name)
        self._word_to_id[word_name] = idx
        idx += 1

  @property
  def bos(self):
    return self._bos

  @property
  def eos(self):
    return self._eos

  @property
  def unk(self):
    return self._unk

  @property
  def size(self):
    return len(self._id_to_word)

  def word_to_id(self, word):
    if word in self._word_to_id:
      return self._word_to_id[word]
    return self.unk

  def id_to_word(self, cur_id):
    if cur_id < self.size:
      return self._id_to_word[cur_id]
    return 'ERROR'

  def decode(self, cur_ids):
    """Convert a list of ids to a sentence, with space inserted."""
    return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

  def encode(self, sentence):
    """Convert a sentence to a list of ids, with special tokens added."""
    word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split()]
    return np.array([self.bos] + word_ids + [self.eos], dtype=np.int32)


class CharsVocabulary(Vocabulary):
  """Vocabulary containing character-level information."""

  def __init__(self, filename, max_word_length):
    super(CharsVocabulary, self).__init__(filename)
    self._max_word_length = max_word_length
    chars_set = set()

    for word in self._id_to_word:
      chars_set |= set(word)

    free_ids = []
    for i in range(256):
      if chr(i) in chars_set:
        continue
      free_ids.append(chr(i))

    if len(free_ids) < 5:
      raise ValueError('Not enough free char ids: %d' % len(free_ids))

    self.bos_char = free_ids[0]  # <begin sentence>
    self.eos_char = free_ids[1]  # <end sentence>
    self.bow_char = free_ids[2]  # <begin word>
    self.eow_char = free_ids[3]  # <end word>
    self.pad_char = free_ids[4]  # <padding>

    chars_set |= {self.bos_char, self.eos_char, self.bow_char, self.eow_char,
                  self.pad_char}

    self._char_set = chars_set
    num_words = len(self._id_to_word)

    self._word_char_ids = np.zeros([num_words, max_word_length], dtype=np.int32)

    self.bos_chars = self._convert_word_to_char_ids(self.bos_char)
    self.eos_chars = self._convert_word_to_char_ids(self.eos_char)

    for i, word in enumerate(self._id_to_word):
      self._word_char_ids[i] = self._convert_word_to_char_ids(word)

  @property
  def word_char_ids(self):
    return self._word_char_ids

  @property
  def max_word_length(self):
    return self._max_word_length

  def _convert_word_to_char_ids(self, word):
    code = np.zeros([self.max_word_length], dtype=np.int32)
    code[:] = ord(self.pad_char)

    if len(word) > self.max_word_length - 2:
      word = word[:self.max_word_length-2]
    cur_word = self.bow_char + word + self.eow_char
    for j in range(len(cur_word)):
      code[j] = ord(cur_word[j])
    return code

  def word_to_char_ids(self, word):
    if word in self._word_to_id:
      return self._word_char_ids[self._word_to_id[word]]
    else:
      return self._convert_word_to_char_ids(word)

  def encode_chars(self, sentence):
    chars_ids = [self.word_to_char_ids(cur_word)
                 for cur_word in sentence.split()]
    return np.vstack([self.bos_chars] + chars_ids + [self.eos_chars])


def get_batch(generator, batch_size, num_steps, max_word_length, pad=False):
  """Read batches of input."""
  cur_stream = [None] * batch_size

  inputs = np.zeros([batch_size, num_steps], np.int32)
  char_inputs = np.zeros([batch_size, num_steps, max_word_length], np.int32)
  global_word_ids = np.zeros([batch_size, num_steps], np.int32)
  targets = np.zeros([batch_size, num_steps], np.int32)
  weights = np.ones([batch_size, num_steps], np.float32)

  no_more_data = False
  while True:
    inputs[:] = 0
    char_inputs[:] = 0
    global_word_ids[:] = 0
    targets[:] = 0
    weights[:] = 0.0

    for i in range(batch_size):
      cur_pos = 0

      while cur_pos < num_steps:
        if cur_stream[i] is None or len(cur_stream[i][0]) <= 1:
          try:
            cur_stream[i] = list(generator.next())
          except StopIteration:
            # No more data, exhaust current streams and quit
            no_more_data = True
            break

        how_many = min(len(cur_stream[i][0]) - 1, num_steps - cur_pos)
        next_pos = cur_pos + how_many

        inputs[i, cur_pos:next_pos] = cur_stream[i][0][:how_many]
        char_inputs[i, cur_pos:next_pos] = cur_stream[i][1][:how_many]
        global_word_ids[i, cur_pos:next_pos] = cur_stream[i][2][:how_many]
        targets[i, cur_pos:next_pos] = cur_stream[i][0][1:how_many+1]
        weights[i, cur_pos:next_pos] = 1.0

        cur_pos = next_pos
        cur_stream[i][0] = cur_stream[i][0][how_many:]
        cur_stream[i][1] = cur_stream[i][1][how_many:]
        cur_stream[i][2] = cur_stream[i][2][how_many:]

        if pad:
          break

    if no_more_data and np.sum(weights) == 0:
      # There is no more data and this is an empty batch. Done!
      break
    yield inputs, char_inputs, global_word_ids, targets, weights


class LM1BDataset(object):
  """Utility class for 1B word benchmark dataset.
  The current implementation reads the data from the tokenized text files.
  """

  def __init__(self, filepattern, vocab):
    """Initialize LM1BDataset reader.
    Args:
      filepattern: Dataset file pattern.
      vocab: Vocabulary.
    """
    self._vocab = vocab
    self._all_shards = tf.compat.v1.gfile.Glob(filepattern)
    tf.logging.info('Found %d shards at %s', len(self._all_shards), filepattern)

  def _load_random_shard(self):
    """Randomly select a file and read it."""
    return self._load_shard(random.choice(self._all_shards))

  def _load_shard(self, shard_name):
    """Read one file and convert to ids.
    Args:
      shard_name: file path.
    Returns:
      list of (id, char_id, global_word_id) tuples.
    """
    tf.logging.info('Loading data from: %s', shard_name)
    with tf.gfile.Open(shard_name) as f:
      sentences = f.readlines()
    chars_ids = [self.vocab.encode_chars(sentence) for sentence in sentences]
    ids = [self.vocab.encode(sentence) for sentence in sentences]

    global_word_ids = []
    current_idx = 0
    for word_ids in ids:
      current_size = len(word_ids) - 1  # without <BOS> symbol
      cur_ids = np.arange(current_idx, current_idx + current_size)
      global_word_ids.append(cur_ids)
      current_idx += current_size

    tf.logging.info('Loaded %d words.', current_idx)
    tf.logging.info('Finished loading')
    return zip(ids, chars_ids, global_word_ids)

  def _get_sentence(self, forever=True):
    while True:
      ids = self._load_random_shard()
      for current_ids in ids:
        yield current_ids
      if not forever:
        break

  def get_batch(self, batch_size, num_steps, pad=False, forever=True):
    return get_batch(self._get_sentence(forever), batch_size, num_steps,
                     self.vocab.max_word_length, pad=pad)

  @property
  def vocab(self):
    return self._vocab

# lm_utils.py

def LoadModel(gd_file, ckpt_file):
    """Load the model from GraphDef and Checkpoint.

    Args:
      gd_file: GraphDef proto text file.
      ckpt_file: TensorFlow Checkpoint file.

    Returns:
      TensorFlow session and tensors dict.
    """
    with tf.Graph().as_default():
        sys.stderr.write('Recovering graph.\n')
        with tf.compat.v1.gfile.FastGFile(gd_file, 'r') as f:
            s = f.read()
            gd = tf.compat.v1.GraphDef()
            text_format.Merge(s, gd)

        tf.compat.v1.logging.info('Recovering Graph %s', gd_file)
        t = {}
        [t['states_init'], t['lstm/lstm_0/control_dependency'],
         t['lstm/lstm_1/control_dependency'], t['softmax_out'], t['class_ids_out'],
         t['class_weights_out'], t['log_perplexity_out'], t['inputs_in'],
         t['targets_in'], t['target_weights_in'], t['char_inputs_in'],
         t['all_embs'], t['softmax_weights'], t['global_step']
         ] = tf.import_graph_def(gd, {}, ['states_init',
                                          'lstm/lstm_0/control_dependency:0',
                                          'lstm/lstm_1/control_dependency:0',
                                          'softmax_out:0',
                                          'class_ids_out:0',
                                          'class_weights_out:0',
                                          'log_perplexity_out:0',
                                          'inputs_in:0',
                                          'targets_in:0',
                                          'target_weights_in:0',
                                          'char_inputs_in:0',
                                          'all_embs_out:0',
                                          'Reshape_3:0',
                                          'global_step:0'], name='')

        sys.stderr.write('Recovering checkpoint %s\n' % ckpt_file)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
        sess.run('save/restore_all', {'save/Const:0': ckpt_file})
        sess.run(t['states_init'])

    return sess, t

class LM(object):
    def __init__(self):
        # self.PBTXT_PATH = os.path.join(GOOGLE_LM_DIR, 'graph-2016-09-10.pbtxt')
        self.PBTXT_PATH = os.path.join(GOOGLE_LM_DIR, 'graph-gpu.pbtxt')
        self.CKPT_PATH = os.path.join(GOOGLE_LM_DIR, 'ckpt-*')
        self.VOCAB_PATH = os.path.join(GOOGLE_LM_DIR, 'vocab-2016-09-10.txt')

        self.BATCH_SIZE = 1
        self.NUM_TIMESTEPS = 1
        self.MAX_WORD_LEN = 50

        self.vocab = CharsVocabulary(self.VOCAB_PATH, self.MAX_WORD_LEN)
        print('LM vocab loading done')
        # with tf.device("/gpu:0"):
        #     self.graph = tf.Graph()
        #     self.sess = tf.compat.v1.Session(graph=self.graph)
        # with self.graph.as_default():
        self.sess, self.t = LoadModel(self.PBTXT_PATH, self.CKPT_PATH)



    def get_words_probs(self, prefix_words, list_words, suffix=None):
        targets = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.int32)
        weights = np.ones([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.float32)

        if prefix_words.find('<S>') != 0:
            prefix_words = '<S> ' + prefix_words
        prefix = [self.vocab.word_to_id(w) for w in prefix_words.split()]
        prefix_char_ids = [self.vocab.word_to_char_ids(w) for w in prefix_words.split()]

        inputs = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.int32)
        char_ids_inputs = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS, self.vocab.max_word_length], np.int32)

        samples = prefix[:]
        char_ids_samples = prefix_char_ids[:]
        inputs = [ [samples[-1]]]
        char_ids_inputs[0, 0, :] = char_ids_samples[-1]
        softmax = self.sess.run(self.t['softmax_out'],
        feed_dict={
            self.t['char_inputs_in']: char_ids_inputs,
            self.t['inputs_in']: inputs,
            self.t['targets_in']: targets,
            self.t['target_weights_in']: weights
        })
        # print(list_words)
        words_ids = [self.vocab.word_to_id(w) for w in list_words]
        word_probs =[softmax[0][w_id] for w_id in words_ids]
        word_probs = np.array(word_probs)

        if suffix == None:
            suffix_probs = np.ones(word_probs.shape)
        else:
            suffix_id = self.vocab.word_to_id(suffix)
            suffix_probs = []
            for idx, w_id in enumerate(words_ids):
                # print('..', list_words[idx])
                inputs = [[w_id]]
                w_char_ids = self.vocab.word_to_char_ids(list_words[idx])
                char_ids_inputs[0, 0, :] = w_char_ids
                softmax = self.sess.run(self.t['softmax_out'],
                                         feed_dict={
                                             self.t['char_inputs_in']: char_ids_inputs,
                                             self.t['inputs_in']: inputs,
                                             self.t['targets_in']: targets,
                                             self.t['target_weights_in']: weights
                                         })
                suffix_probs.append(softmax[0][suffix_id])
            suffix_probs = np.array(suffix_probs)
        # print(word_probs, suffix_probs)
        return suffix_probs * word_probs


if __name__ == '__main__':
   # my_lm = LM()
   # list_words = 'play will playing played afternoon'.split()
   # prefix = 'i'
   # suffix = 'yesterday'
   # probs = (my_lm.get_words_probs(prefix, list_words, suffix))
   # for i, w in enumerate(list_words):
   #     print(w, ' - ', probs[i])

   my_lm = LM()
   list_words = 'play will playing played yesterday'.split()
   prefix = 'i'
   suffix = 'afternoon'
   probs = (my_lm.get_words_probs(prefix, list_words, suffix))
   for i, w in enumerate(list_words):
       print(w, ' - ', probs[i])