import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import time
import argparse
import re
import inspect

import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch import optim
import nltk
from tqdm import tqdm

from local_models.NLI_config import *
from dataloader import read_orig_snli, clean_str, read_orig_mnli

GLOVE_EMBEDDING_PATH = "path/to/glove.6B.200d.txt"
GLOVE_EMBEDDING_PATH_64B300D = 'path/to/glove.840B.300d.txt'
SNLI_FOLDER_PATH = 'path/to/snli_data_folder'
MNLI_FOLDER_PATH = 'path/to/mnli_data_folder'



class InferSent(nn.Module):

    def __init__(self, config):
        super(InferSent, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']
        self.version = 1 if 'version' not in config else config['version']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1,
                                bidirectional=True, dropout=self.dpout_model)

        assert self.version in [1, 2]
        if self.version == 1:
            self.bos = '<s>'
            self.eos = '</s>'
            self.max_pad = True
            self.moses_tok = False
        elif self.version == 2:
            self.bos = '<p>'
            self.eos = '</p>'
            self.max_pad = False
            self.moses_tok = True

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return self.enc_lstm.bias_hh_l0.data.is_cuda

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: Variable(seqlen x bsize x worddim)
        sent, sent_len = sent_tuple

        # Sort by length (keep idx)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1].copy(), np.argsort(-sent_len)
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda() if self.is_cuda() \
            else torch.from_numpy(idx_sort)
        sent = sent.index_select(1, Variable(idx_sort))

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.is_cuda() \
            else torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(1, Variable(idx_unsort))

        # Pooling
        if self.pool_type == "mean":
            sent_len = Variable(torch.FloatTensor(sent_len.copy())).unsqueeze(1).cuda()
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            if not self.max_pad:
                sent_output[sent_output == 0] = -1e9
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2

        return emb

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch = Variable(self.get_batch(
                        sentences[stidx:stidx + bsize]), volatile=True)
            if self.is_cuda():
                batch = batch.cuda()
            batch = self.forward(
                (batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

    # def visualize(self, sent, tokenize=True):
    #
    #     sent = sent.split() if not tokenize else self.tokenize(sent)
    #     sent = [[self.bos] + [word for word in sent if word in self.word_vec] + [self.eos]]
    #
    #     if ' '.join(sent[0]) == '%s %s' % (self.bos, self.eos):
    #         import warnings
    #         warnings.warn('No words in "%s" have w2v vectors. Replacing \
    #                        by "%s %s"..' % (sent, self.bos, self.eos))
    #     batch = Variable(self.get_batch(sent), volatile=True)
    #
    #     if self.is_cuda():
    #         batch = batch.cuda()
    #     output = self.enc_lstm(batch)[0]
    #     output, idxs = torch.max(output, 0)
    #     # output, idxs = output.squeeze(), idxs.squeeze()
    #     idxs = idxs.data.cpu().numpy()
    #     argmaxs = [np.sum((idxs == k)) for k in range(len(sent[0]))]
    #
    #     # # visualize model
    #     # import matplotlib.pyplot as plt
    #     # x = range(len(sent[0]))
    #     # y = [100.0 * n / np.sum(argmaxs) for n in argmaxs]
    #     # plt.xticks(x, sent[0], rotation=45)
    #     # plt.bar(x, y)
    #     # plt.ylabel('%')
    #     # plt.title('Visualisation of words importance')
    #     # plt.show()
    #     #
    #     # return output, idxs

class NLINet(nn.Module):
    def __init__(self, config):
        super(NLINet, self).__init__()

        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']

        self.encoder = eval(self.encoder_type)(config)
        self.inputdim = 4*2*self.enc_lstm_dim
        self.inputdim = 4*self.inputdim if self.encoder_type in \
                        ["ConvNetEncoder", "InnerAttentionMILAEncoder"] else self.inputdim
        self.inputdim = self.inputdim/2 if self.encoder_type == "LSTMEncoder" \
                                        else self.inputdim
        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
                )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
                )

    def forward(self, s1, s2):
        # s1 : (s1, s1_len)
        u = self.encoder(s1)
        v = self.encoder(s2)

        features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        output = self.classifier(features)
        return output

    def encode(self, s1):
        emb = self.encoder(s1)
        return emb


class InfersentWrapper(object):
    def __init__(self, pretrained_model_path,  batch_size=64):
        config_nli_model = {
            'word_emb_dim': 200,  # 300,
            'enc_lstm_dim': 2048,
            'n_enc_layers': 1,
            'dpout_model': 0.,
            'dpout_fc': 0.,
            'fc_dim': 512,
            'bsize': batch_size,
            'n_classes': 3,
            'pool_type': 'max',
            'nonlinear_fc': 0,
            'encoder_type': 'InferSent',
            'use_cuda': True,
            'use_target': False,
            'version': 1,
        }

        state_dict, word2vec = torch.load(pretrained_model_path)

        self.nli_net = NLINet(config_nli_model)
        print('load from', pretrained_model_path)
        self.nli_net.load_state_dict(state_dict)
        self.nli_net.cuda()
        print(self.nli_net)
        print('Load Network Complete')

        self.word2vec = word2vec

        self.bos = BOS
        self.eos = EOS
        self.oov = OOV
        self.word_emb_dim = len(self.word2vec[self.oov])
        self.batch_size = batch_size

    def transform_text(self, data):
        # transform data into seq of embeddings
        premises = data['premises']
        hypotheses = data['hypotheses']

        # add bos and eos
        premises = [['<s>'] + premise + ['</s>'] for premise in premises]
        hypotheses = [['<s>'] + hypothese + ['</s>'] for hypothese in hypotheses]

        batches = []
        for stidx in range(0, len(premises), self.batch_size):
            # prepare batch
            s1_batch, s1_len = get_batch(premises[stidx:stidx + self.batch_size],
                                         self.word2vec, self.word_emb_dim)
            s2_batch, s2_len = get_batch(hypotheses[stidx:stidx + self.batch_size],
                                         self.word2vec, self.word_emb_dim)
            batches.append(((s1_batch, s1_len), (s2_batch, s2_len)))

        return batches

    def text_pred(self, text_data):
        self.nli_net.eval()

        data_batches = self.transform_text(text_data)

        # Deactivate autograd for evaluation.
        probs_all = []
        with torch.no_grad():
            for batch in data_batches:
                # Move input and output data to the GPU if one is used.
                (s1_batch, s1_len), (s2_batch, s2_len) = batch
                s1_batch, s2_batch = s1_batch.cuda(), s2_batch.cuda()
                logits = self.nli_net((s1_batch, s1_len), (s2_batch, s2_len))
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params


def get_batch(batch, word_vec, word_emb_dim=200):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    #         print(max_len)
    embed = np.zeros((max_len, len(batch), word_emb_dim))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            if batch[i][j] in word_vec:
                embed[j, i, :] = word_vec[batch[i][j]]
            else:
                embed[j, i, :] = word_vec[OOV]

    return torch.from_numpy(embed).float(), lengths

def get_embedding(sentences, embedding_path, word_emb_dim=200):
    # create word_vec with glove vectors
    # create vocab of words
    word_dict = {}
    for sent in sentences:
        for word in sent:
            if word not in word_dict:
                word_dict[word] = ''
    word_dict[BOS] = ''
    word_dict[EOS] = ''
    word_dict[OOV] = ''

    word_vec = {}
    word_vec[OOV] = np.random.normal(size=(word_emb_dim))
    with open(embedding_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            # if word in word_dict:
            word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}(/{1}) words with embedding vectors'.format(
        len(word_vec), len(word_dict)))
    return word_vec


def _process_one_text(text):
    text_ = clean_str(" ".join(nltk.word_tokenize(text)))
    text_ = text_.rstrip().split()
    # text_ = ['<s>'] + text_ + ['</s>']

    return text_


def _process_data(data_list):

    new_data = {'premises': [], 'hypotheses': [], 'labels': []}

    for _data in tqdm(data_list):
        label = _data[0]
        premise = _data[1]
        hypo = _data[2]

        premise_ = _process_one_text(premise)
        hypo_ = _process_one_text(hypo)
        label_ = NLI_LABEL_STR2NUM.get(label, -1)
        
        new_data['premises'].append(premise_)
        new_data['hypotheses'].append(hypo_)
        new_data['labels'].append(label_)

    return new_data


def get_params():
    parser = argparse.ArgumentParser(description='NLI training')
    # paths
    parser.add_argument("--dataset", type=str, default='snli', choices=['snli', 'mnli'], help="NLI dataset (SNLI or MultiNLI)")
    parser.add_argument("--dataset_path", type=str, default='dataset/SNLI/', help="NLI data path (SNLI or MultiNLI)")
    parser.add_argument("--output_dir", type=str, default='/home/workspace/big_data/hardlabel/models',
                        help="Output directory")
    parser.add_argument("--model_name", type=str, default='model_nli.pickle')
    parser.add_argument("--word_emb_path", type=str, default="dataset/GloVe/glove.840B.300d.txt",
                        help="word embedding file path")

    # training
    parser.add_argument("--n_epochs", type=int, default=25) # 20
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
    parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
    parser.add_argument("--decay", type=float, default=0.99, help="lr decay") # 0.99
    parser.add_argument("--minlr", type=float, default=1e-6, help="minimum lr") # 1e-5
    parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

    # model
    # parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
    # parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
    # parser.add_argument("--nonlinear_fc", type=int, default=1, help="use nonlinearity in fc")
    # parser.add_argument("--encoder_type", type=str, default='InferSent', help="see list of encoders")
    # parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
    # parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
    # parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
    parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
    # parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
    # parser.add_argument("--sememe_dim", type=int, default=300, help="encoder sememe dimension") # NO USE?

    # gpu
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--seed", type=int, default=2122, help="seed")

    # data
    parser.add_argument("--word_emb_dim", type=int, default=200, help="word embedding dimension")

    params, _ = parser.parse_known_args()

    # TODO: remove giving parameters
    params.dataset = 'mnli'
    params.dataset_path = MNLI_FOLDER_PATH if params.dataset == 'mnli' else SNLI_FOLDER_PATH
    params.output_dir = '/home/workspace/big_data/hardlabel/models/infersent'
    params.model_name = params.dataset
    params.word_emb_path = GLOVE_EMBEDDING_PATH
    # params.optimizer = 'adam'
    params.word_emb_dim = 200

    return params


def train_snli(params):
    # set gpu device
    torch.cuda.set_device(params.gpu_id)
    device = torch.device("cuda:0")
    # print parameters passed, and all parameters
    print('\ntogrep : {0}\n'.format(sys.argv[1:]))
    print(params)

    """
    SEED
    """
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    """
    DATA
    """

    cache_path = f'../tmp/infersent/{params.dataset}_data.pkl'
    save_model_path = os.path.join(params.output_dir, params.model_name)
    save_emd_path = os.path.join(params.output_dir, f'{params.model_name}_word2vec.pkl')

    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))


    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            train_data, valid_data, test_data = pickle.load(f)
    else:
        train_data_list = read_orig_snli(params.dataset_path, 'train')
        dev_data_list = read_orig_snli(params.dataset_path, 'dev')
        test_data_list = read_orig_snli(params.dataset_path, 'test')

        train_data = _process_data(train_data_list)
        valid_data = _process_data(dev_data_list)
        test_data = _process_data(test_data_list)

        with open(cache_path, 'wb') as f:
            pickle.dump((train_data, valid_data, test_data), f)



    word2vec = get_embedding(train_data['premises'] + train_data['hypotheses'],
                             embedding_path=params.word_emb_path, word_emb_dim=params.word_emb_dim)
    # with open(save_emd_path, 'wb') as f:
    #     pickle.dump(word2vec, f)


    print('Load dataset complete')

    """
    MODEL
    """
    # model
    config_nli_model = {
        'word_emb_dim': params.word_emb_dim, #200 TODO: 300,
        'enc_lstm_dim': 2048,
        'n_enc_layers': 1,
        'dpout_model': 0.,
        'dpout_fc': 0.,
        'fc_dim': 512,
        'bsize': params.batch_size,
        'n_classes': 3,
        'pool_type': 'max',
        'nonlinear_fc': 0,
        'encoder_type': 'InferSent',
        'use_cuda': True,
        'use_target': False,
        'version': 1,
    }

    nli_net = NLINet(config_nli_model)
    # nli_net.emb_sememe.weight.data.copy_(emb_s)
    print(nli_net)
    print('Load Network Complete')

    # loss
    weight = torch.FloatTensor(params.n_classes).fill_(1)
    loss_fn = nn.CrossEntropyLoss(weight=weight, reduction='sum')

    # optimizer
    optim_fn, optim_params = get_optimizer(params.optimizer)
    optimizer = optim_fn(nli_net.parameters(), **optim_params)

    # cuda by default
    nli_net.cuda()
    loss_fn.cuda()

    """
    TRAIN
    """
    val_acc_best = -1e10
    adam_stop = False
    stop_training = False
    lr = optim_params['lr'] if 'sgd' in params.optimizer else None

    def trainepoch(epoch):
        print('\nTRAINING : Epoch ' + str(epoch))
        nli_net.train()
        all_costs = []
        logs = []
        words_count = 0

        last_time = time.time()
        correct = 0.


        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch > 1 \
                                                                                            and 'sgd' in params.optimizer else \
        optimizer.param_groups[0]['lr']
        print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

        # shuffle the data
        permutation = np.random.permutation(len(train_data['premises']))

        # add bos and eos
        s1 = [['<s>'] + train_data['premises'][i] + ['</s>'] for i in permutation]
        s2 = [['<s>'] + train_data['hypotheses'][i] + ['</s>'] for i in permutation]
        target = [train_data['labels'][i] for i in permutation]

        for stidx in range(0, len(s1), params.batch_size):

            # prepare batch
            s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size], word2vec, params.word_emb_dim)
            s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size], word2vec, params.word_emb_dim)

            s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
            tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()
            k = s1_batch.size(1)  # actual batch size

            # model forward
            output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
            pred = output.data.max(1)[1]
            correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()
            assert len(pred) == len(s1[stidx:stidx + params.batch_size])

            # loss
            loss = loss_fn(output, tgt_batch)
            all_costs.append(loss.item())
            words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

            # backward
            optimizer.zero_grad()
            loss.backward()

            # # gradient clipping (off by default)
            shrink_factor = 1
            total_norm = 0

            for p in nli_net.parameters():
                if p.requires_grad:
                    if p.grad is not None:
                        p.grad.data.div_(k)  # divide by the actual batch size
                        total_norm += (p.grad.data.norm() ** 2).item()
            total_norm = np.sqrt(total_norm)

            if total_norm > params.max_norm:
                shrink_factor = params.max_norm / total_norm
            current_lr = optimizer.param_groups[0]['lr']  # current lr (no external "lr", for adam)
            optimizer.param_groups[0]['lr'] = current_lr * shrink_factor  # just for update

            # optimizer step
            optimizer.step()
            optimizer.param_groups[0]['lr'] = current_lr

            if len(all_costs) == 100:
                logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                    stidx, np.round(np.mean(all_costs), 2),
                    int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                    int(words_count * 1.0 / (time.time() - last_time)),
                    100. * float(correct) / (stidx + k)))
                print(logs[-1])
                last_time = time.time()
                words_count = 0
                all_costs = []

        train_acc = 100 * float(correct) / len(s1)
        print('results : epoch {0} ; mean accuracy train : {1}'
              .format(epoch, train_acc))
        return train_acc

    def evaluate(val_acc_best, lr, stop_training, adam_stop, epoch, eval_type='valid', final_eval=False):
        nli_net.eval()
        correct = 0.

        if eval_type == 'valid':
            print('\nVALIDATION : Epoch {0}'.format(epoch))

        s1 = valid_data['premises'] if eval_type == 'valid' else test_data['premises']
        s2 = valid_data['hypotheses'] if eval_type == 'valid' else test_data['hypotheses']
        target = valid_data['labels'] if eval_type == 'valid' else test_data['labels']

        # add bos and eos
        s1 = [['<s>'] + premise + ['</s>'] for premise in s1]
        s2 = [['<s>'] + hypothesis + ['</s>'] for hypothesis in s2]

        # s1 = valid['s1'] if eval_type == 'valid' else test['s1']
        # s2 = valid['s2'] if eval_type == 'valid' else test['s2']
        # target = valid['label'] if eval_type == 'valid' else test['label']

        for i in range(0, len(s1), params.batch_size):
            # prepare batch
            s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word2vec, params.word_emb_dim)
            s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word2vec, params.word_emb_dim)
            s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
            tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()

            # model forward
            output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

            pred = output.data.max(1)[1]
            correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()

        # save model
        eval_acc = 100 * float(correct) / len(s1)
        if final_eval:
            print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
        else:
            print('togrep : results : epoch {0} ; mean accuracy {1} :\
                  {2}'.format(epoch, eval_type, eval_acc))

        if eval_type == 'valid' and epoch <= params.n_epochs:
            if eval_acc > val_acc_best:
                print('saving model at epoch {0}'.format(epoch))
                if not os.path.exists(params.output_dir):
                    os.makedirs(params.output_dir)
                torch.save((nli_net.state_dict(), word2vec), save_model_path)
                val_acc_best = eval_acc
            else:
                if 'sgd' in params.optimizer:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                    print('Shrinking lr by : {0}. New lr = {1}'
                          .format(params.lrshrink,
                                  optimizer.param_groups[0]['lr']))
                    if optimizer.param_groups[0]['lr'] < params.minlr:
                        stop_training = True
                if 'adam' in params.optimizer:
                    # early stopping (at 2nd decrease in accuracy)
                    stop_training = adam_stop
                    adam_stop = True
        return eval_acc, val_acc_best, lr, stop_training, adam_stop

    """
    Train model on Natural Language Inference task
    """
    epoch = 1

    while not stop_training and epoch <= params.n_epochs:
        train_acc = trainepoch(epoch)
        eval_acc, val_acc_best, lr, stop_training, adam_stop = evaluate(val_acc_best, lr, stop_training, adam_stop, epoch, 'valid')
        epoch += 1

    # Run best model on test set.
    checkpoint, word2vec = torch.load(save_model_path)
    nli_net.load_state_dict(checkpoint)
    nli_net.cuda()

    print('\nTEST : Epoch {0}'.format(epoch))
    evaluate(val_acc_best, lr, stop_training, adam_stop, 1e6, 'valid', True)
    evaluate(val_acc_best, lr, stop_training, adam_stop, 0, 'test', True)

    # Save encoder instead of full model
    # torch.save(nli_net.encoder.state_dict(), os.path.join(params.output_dir, params.model_name + '.encoder.pkl'))


def train_mnli(params):

    print('Start training on', params.dataset)

    # set gpu device
    torch.cuda.set_device(params.gpu_id)
    device = torch.device("cuda:0")
    # print parameters passed, and all parameters
    print('\ntogrep : {0}\n'.format(sys.argv[1:]))
    print(params)

    """
    SEED
    """
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    """
    DATA
    """

    cache_path = f'../tmp/infersent/{params.model_name}_data.pkl'
    save_model_path = os.path.join(params.output_dir, params.model_name)
    save_emd_path = os.path.join(params.output_dir, f'{params.model_name}_word2vec.pkl')

    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))


    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            train_data, matched_data, mismatched_data = pickle.load(f)
    else:
        train_data_list = read_orig_mnli(params.dataset_path, 'train')
        matched_data_list = read_orig_mnli(params.dataset_path, 'dev_matched')
        mismatched_data_list = read_orig_mnli(params.dataset_path, 'dev_mismatched')

        train_data = _process_data(train_data_list)
        matched_data = _process_data(matched_data_list)
        mismatched_data = _process_data(mismatched_data_list)

        with open(cache_path, 'wb') as f:
            pickle.dump((train_data, matched_data, mismatched_data), f)


    word2vec = get_embedding(train_data['premises'] + train_data['hypotheses'],
                             embedding_path=params.word_emb_path, word_emb_dim=params.word_emb_dim)


    print('Load dataset complete')

    """
    MODEL
    """
    # model
    config_nli_model = {
        'word_emb_dim': 200,
        'enc_lstm_dim': 2048,
        'n_enc_layers': 1,
        'dpout_model': 0.,
        'dpout_fc': 0.,
        'fc_dim': 512,
        'bsize': params.batch_size,
        'n_classes': 3,
        'pool_type': 'max',
        'nonlinear_fc': 0,
        'encoder_type': 'InferSent',
        'use_cuda': True,
        'use_target': False,
        'version': 1,
    }

    nli_net = NLINet(config_nli_model)
    # nli_net.emb_sememe.weight.data.copy_(emb_s)
    print(nli_net)
    print('Load Network Complete')

    # loss
    weight = torch.FloatTensor(params.n_classes).fill_(1)
    loss_fn = nn.CrossEntropyLoss(weight=weight, reduction='sum')

    # optimizer
    optim_fn, optim_params = get_optimizer(params.optimizer)
    optimizer = optim_fn(nli_net.parameters(), **optim_params)

    # cuda by default
    nli_net.cuda()
    loss_fn.cuda()

    """
    TRAIN
    """
    val_acc_best = -1e10
    adam_stop = False
    stop_training = False
    lr = optim_params['lr'] if 'sgd' in params.optimizer else None

    def trainepoch(epoch):
        print('\nTRAINING : Epoch ' + str(epoch))
        nli_net.train()
        all_costs = []
        logs = []
        words_count = 0

        last_time = time.time()
        correct = 0.


        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch > 1 \
                                                                                            and 'sgd' in params.optimizer else \
        optimizer.param_groups[0]['lr']
        print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

        # shuffle the data
        permutation = np.random.permutation(len(train_data['premises']))

        # add bos and eos
        s1 = [['<s>'] + train_data['premises'][i] + ['</s>'] for i in permutation]
        s2 = [['<s>'] + train_data['hypotheses'][i] + ['</s>'] for i in permutation]
        target = [train_data['labels'][i] for i in permutation]

        for stidx in range(0, len(s1), params.batch_size):

            # prepare batch
            s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size], word2vec, params.word_emb_dim)
            s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size], word2vec, params.word_emb_dim)

            s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
            tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()
            k = s1_batch.size(1)  # actual batch size

            # model forward
            output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
            pred = output.data.max(1)[1]
            correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()
            assert len(pred) == len(s1[stidx:stidx + params.batch_size])

            # loss
            loss = loss_fn(output, tgt_batch)
            all_costs.append(loss.item())
            words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping (off by default)
            shrink_factor = 1
            total_norm = 0

            for p in nli_net.parameters():
                if p.requires_grad:
                    if p.grad is not None:
                        p.grad.data.div_(k)  # divide by the actual batch size
                        total_norm += (p.grad.data.norm() ** 2).item()
            total_norm = np.sqrt(total_norm)

            if total_norm > params.max_norm:
                shrink_factor = params.max_norm / total_norm
            current_lr = optimizer.param_groups[0]['lr']  # current lr (no external "lr", for adam)
            optimizer.param_groups[0]['lr'] = current_lr * shrink_factor  # just for update

            # optimizer step
            optimizer.step()
            optimizer.param_groups[0]['lr'] = current_lr

            if len(all_costs) == 100:
                logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                    stidx, np.round(np.mean(all_costs), 2),
                    int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                    int(words_count * 1.0 / (time.time() - last_time)),
                    100. * float(correct) / (stidx + k)))
                print(logs[-1])
                last_time = time.time()
                words_count = 0
                all_costs = []

        train_acc = 100 * float(correct) / len(s1)
        print('results : epoch {0} ; mean accuracy train : {1}'
              .format(epoch, train_acc))
        return train_acc

    def evaluate(val_acc_best, lr, stop_training, adam_stop, epoch, eval_type='valid', final_eval=False):
        nli_net.eval()
        correct = 0.

        if eval_type == 'valid':
            print('\nVALIDATION : Epoch {0}'.format(epoch))

        s1 = matched_data['premises'] if eval_type == 'matched' else mismatched_data['premises']
        s2 = matched_data['hypotheses'] if eval_type == 'matched' else mismatched_data['hypotheses']
        target = matched_data['labels'] if eval_type == 'matched' else mismatched_data['labels']

        # add bos and eos
        s1 = [['<s>'] + premise + ['</s>'] for premise in s1]
        s2 = [['<s>'] + hypothesis + ['</s>'] for hypothesis in s2]

        # s1 = valid['s1'] if eval_type == 'valid' else test['s1']
        # s2 = valid['s2'] if eval_type == 'valid' else test['s2']
        # target = valid['label'] if eval_type == 'valid' else test['label']

        for i in range(0, len(s1), params.batch_size):
            # prepare batch
            s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word2vec, params.word_emb_dim)
            s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word2vec, params.word_emb_dim)
            s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
            tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()

            # model forward
            output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

            pred = output.data.max(1)[1]
            correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()

        # save model
        eval_acc = 100 * float(correct) / len(s1)
        if final_eval:
            print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
        else:
            print('togrep : results : epoch {0} ; mean accuracy {1} :\
                  {2}'.format(epoch, eval_type, eval_acc))

        if eval_type == 'matched' and epoch <= params.n_epochs:
            if eval_acc > val_acc_best:
                print('saving model at epoch {0}'.format(epoch))
                if not os.path.exists(params.output_dir):
                    os.makedirs(params.output_dir)
                torch.save((nli_net.state_dict(), word2vec), save_model_path)
                val_acc_best = eval_acc
            else:
                if 'sgd' in params.optimizer:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                    print('Shrinking lr by : {0}. New lr = {1}'
                          .format(params.lrshrink,
                                  optimizer.param_groups[0]['lr']))
                    if optimizer.param_groups[0]['lr'] < params.minlr:
                        stop_training = True
                if 'adam' in params.optimizer:
                    # early stopping (at 2nd decrease in accuracy)
                    stop_training = adam_stop
                    adam_stop = True
        return eval_acc, val_acc_best, lr, stop_training, adam_stop

    """
    Train model on Natural Language Inference task
    """
    epoch = 1

    while not stop_training and epoch <= params.n_epochs:
        train_acc = trainepoch(epoch)
        eval_acc, val_acc_best, lr, stop_training, adam_stop = evaluate(val_acc_best, lr, stop_training, adam_stop, epoch, 'matched')
        _, _, _, _, _ = evaluate(val_acc_best, lr, stop_training, adam_stop, epoch, 'mismatched')
        epoch += 1

    # Run best model on test set.
    checkpoint, word2vec = torch.load(save_model_path)
    nli_net.load_state_dict(checkpoint)
    nli_net.cuda()

    print('\nTEST : Epoch {0}'.format(epoch))
    evaluate(val_acc_best, lr, stop_training, adam_stop, 1e6, 'valid', True)
    evaluate(val_acc_best, lr, stop_training, adam_stop, 0, 'test', True)

    # Save encoder instead of full model
    # torch.save(nli_net.encoder.state_dict(), os.path.join(params.output_dir, params.model_name + '.encoder.pkl'))


def test_snli(params):
    # DATA
    cache_path = '../tmp/infersent/snli_data.pkl'
    save_model_path = os.path.join(params.output_dir, params.model_name)

    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            train_data, valid_data, test_data = pickle.load(f)
    else:
        train_data_list = read_orig_snli(SNLI_FOLDER_PATH, 'train')
        dev_data_list = read_orig_snli(SNLI_FOLDER_PATH, 'dev')
        test_data_list = read_orig_snli(SNLI_FOLDER_PATH, 'test')

        train_data = _process_data(train_data_list)
        valid_data = _process_data(dev_data_list)
        test_data = _process_data(test_data_list)

        with open(cache_path, 'wb') as f:
            pickle.dump((train_data, valid_data, test_data), f)



    """
    MODEL
    """
    infersent_net = InfersentWrapper(save_model_path, batch_size=128)


    pred_list = infersent_net.text_pred(test_data).data.max(dim=1)[1]
    target_list = Variable(torch.LongTensor(test_data['labels'])).cuda()

    c = pred_list.long().eq(target_list.data.long()).cpu().sum().item()

    acc = float(c) / len(target_list)

    print(f'Test accuracy: {acc:.2%}')

def test_mnli(params):
    # DATA
    cache_path = f'../tmp/infersent/{params.model_name}_data.pkl'
    save_model_path = os.path.join(params.output_dir, params.model_name)
    # save_emd_path = os.path.join(params.output_dir, f'{params.model_name}_word2vec.pkl')

    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            train_data, matched_data, mismatched_data = pickle.load(f)
    else:
        train_data_list = read_orig_mnli(params.dataset_path, 'train')
        matched_data_list = read_orig_mnli(params.dataset_path, 'dev_matched')
        mismatched_data_list = read_orig_mnli(params.dataset_path, 'dev_mismatched')

        train_data = _process_data(train_data_list)
        matched_data = _process_data(matched_data_list)
        mismatched_data = _process_data(mismatched_data_list)

        with open(cache_path, 'wb') as f:
            pickle.dump((train_data, matched_data, mismatched_data), f)



    """
    MODEL
    """
    infersent_net = InfersentWrapper(save_model_path, batch_size=128)

    pred_list = infersent_net.text_pred(matched_data).data.max(dim=1)[1]
    target_list = Variable(torch.LongTensor(matched_data['labels'])).cuda()

    c = pred_list.long().eq(target_list.data.long()).cpu().sum().item()

    acc = float(c) / len(target_list)

    print(f'Mathced accuracy: {acc:.2%}')

    pred_list = infersent_net.text_pred(mismatched_data).data.max(dim=1)[1]
    target_list = Variable(torch.LongTensor(mismatched_data['labels'])).cuda()

    c = pred_list.long().eq(target_list.data.long()).cpu().sum().item()

    acc = float(c) / len(target_list)

    print(f'Mismathced accuracy: {acc:.2%}')

    # generate test submision csv file
    import csv


    matched_test_data_list = read_orig_mnli(params.dataset_path, 'test_matched')
    mismatched_test_data_list = read_orig_mnli(params.dataset_path, 'test_mismatched')

    matched_test_data = _process_data(matched_test_data_list)
    mismatched_test_data = _process_data(mismatched_test_data_list)

    # matched test data

    pred_list = infersent_net.text_pred(matched_test_data).data.max(dim=1)[1].cpu().numpy()
    pred_str_list = [NLI_LABEL_NUM2STR[label_idx] for label_idx in pred_list]
    pairID_list = [_data[-1] for _data in matched_test_data_list]

    pred_results = zip(pairID_list, pred_str_list)



    with open("../tmp/infersent_mnli_matched.csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        # write the header
        csvwriter.writerow(['pairID', 'gold_label'])

        # writing the data rows
        csvwriter.writerows(pred_results)

    # mismatched test data


    pred_list = infersent_net.text_pred(mismatched_test_data).data.max(dim=1)[1].cpu().numpy()
    pred_str_list = [NLI_LABEL_NUM2STR[label_idx] for label_idx in pred_list]
    pairID_list = [_data[-1] for _data in mismatched_test_data_list]

    pred_results = zip(pairID_list, pred_str_list)

    with open("../tmp/infersent_mnli_mismatched.csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        # write the header
        csvwriter.writerow(['pairID', 'gold_label'])

        # writing the data rows
        csvwriter.writerows(pred_results)




if __name__ == '__main__':
    params = get_params()
    if params.dataset == 'snli':
        train_snli(params)
        test_snli(params)
    else:
        train_mnli(params)
        test_mnli(params)

