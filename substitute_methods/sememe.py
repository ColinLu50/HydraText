'''
Reimplement of subsitute words used by PSO (Word-level Textual Adversarial Attacking as Combinatorial Optimization)
WordNet Synonyms + Named Entities (NE)
'''

import os
# import nltk

from tqdm import tqdm
from multiprocessing import Pool

import pickle

from tensorflow.keras.preprocessing.text import Tokenizer

# import sys
#
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import nltk
from nltk.tag import StanfordPOSTagger
from nltk.stem import WordNetLemmatizer
import OpenHowNet

from utils import my_file
import dataloader

# download from https://nlp.stanford.edu/software/tagger.shtml
STANFORD_POS_TAGGER_JAR = 'path/to/stanford-postagger-2018-10-16/stanford-postagger.jar'
STANFORD_POS_TAGGER_MODEL = 'path/to/stanford-postagger-2018-10-16/models/english-left3words-distsim.tagger'


stanford_pos_set = {'JJ', 'NN', 'RB', 'VB'}
# ================================== preprocess dataset ====================================

def build_dataset(dataset, dataset_path, output_dir):

    cache_path = os.path.join(output_dir, dataset + "_candidates.pkl")
    if os.path.exists(cache_path):
        return my_file.load_pkl(cache_path)

    if dataset == 'imdb':
        max_vocab_size = 50000
        is_lower = True
        train_texts, _labels = dataloader.read_orig_imdb(dataset_path, 'train')
    elif dataset == 'agnews':
        max_vocab_size = None
        is_lower = True
        train_texts, _labels = dataloader.read_orig_agnews(dataset_path, 'train')
    elif dataset == 'yahoo':
        max_vocab_size = None
        is_lower = True
        train_texts, _labels = dataloader.read_orig_yahoo(dataset_path, 'train')
    elif dataset == 'mr':
        max_vocab_size = None
        is_lower = True
        train_texts, _labels = dataloader.read_orig_mr(dataset_path)
    elif dataset == 'snli':
        max_vocab_size = None
        is_lower = False
        train_data_list = dataloader.read_orig_snli(dataset_path, 'train')

        train_texts = []
        for label, s1, s2 in train_data_list:
            train_texts.append(s1)
            train_texts.append(s2)
    elif dataset[:4] == 'mnli':
        max_vocab_size = None
        is_lower = False
        train_data_list = dataloader.read_orig_mnli(dataset_path, 'train')

        train_texts = []
        for label, s1, s2, _ in train_data_list:
            train_texts.append(s1)
            train_texts.append(s2)


    print('tokenizing...')
    tokenizer = Tokenizer(lower=is_lower)
    tokenizer.fit_on_texts(train_texts)

    dict_ = {}
    inv_dict = {}
    full_dict = {}
    inv_full_dict = {}

    if not max_vocab_size:
        max_vocab_size = len(tokenizer.word_index.items())
    else:
        dict_[max_vocab_size] = 'UNK'
        inv_dict['UNK'] = max_vocab_size

    for word, idx in tokenizer.word_index.items():
        if idx < max_vocab_size:
            inv_dict[idx] = word
            dict_[word] = idx
        full_dict[word] = idx
        inv_full_dict[idx] = word

    train_seqs = tokenizer.texts_to_sequences(train_texts)

    print('Dataset built !')

    my_file.save_pkl((train_seqs, dict_, inv_dict, full_dict, inv_full_dict, max_vocab_size), cache_path)
    print(len(train_seqs))
    return train_seqs, dict_, inv_dict, full_dict, inv_full_dict, max_vocab_size


# # ==================== generate pos tag ==============================

def _pos_tag_single(args):
    text_list, pos_tagger = args
    cur_pos_tags = []
    for text_ in tqdm(text_list):
        pos_tags = pos_tagger.tag(text_)
        cur_pos_tags.append(pos_tags)
    return cur_pos_tags


def generate_pos(dataset, train_seqs, inv_full_dict):
    print('Generate POS list ...')
    cache_path = f'tmp/sememe/{dataset}_pos_tags.pkl'
    if os.path.exists(cache_path):
        return my_file.load_pkl(cache_path)

    jar = STANFORD_POS_TAGGER_JAR
    model = STANFORD_POS_TAGGER_MODEL
    pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')

    train_texts = [[inv_full_dict[t] for t in tt] for tt in train_seqs]

    all_pos_tags = []
    # for text_ in tqdm(train_texts):
    #     pos_tags = pos_tagger.tag(text_)
    #     all_pos_tags.append(pos_tags)

    arg_list = []
    worker_num = 10

    a, b = divmod(len(train_texts), worker_num)
    if b > 0:
        a += 1
    each_worker_task_num = a

    for worker_idx in range(worker_num):
        cur_texts = train_texts[worker_idx * each_worker_task_num: (worker_idx + 1) * each_worker_task_num]
        print(len(cur_texts))
        arg_list.append((cur_texts, pos_tagger))

    with Pool(worker_num) as p:
        res_list = p.map(_pos_tag_single, arg_list)

    for res in res_list:
        all_pos_tags.extend(res)

    my_file.save_pkl(all_pos_tags, cache_path)

    return all_pos_tags



# # ================================== Lemma ===================================
def lemma(dataset, all_pos_tag):

    print('Start Lemma!')

    cache_path = f'tmp/sememe/{dataset}_sss_dict.pkl'
    if os.path.exists(cache_path):
        print('Use cache')
        return my_file.load_pkl(cache_path)


    wnl = WordNetLemmatizer()

    # train_text=[[dataset.inv_full_dict[t] for t in tt] for tt in dataset.train_seqs]
    NNS={}
    NNPS={}
    JJR={}
    JJS={}
    RBR={}
    RBS={}
    VBD={}
    VBG={}
    VBN={}
    VBP={}
    VBZ={}
    inv_NNS={}
    inv_NNPS={}
    inv_JJR={}
    inv_JJS={}
    inv_RBR={}
    inv_RBS={}
    inv_VBD={}
    inv_VBG={}
    inv_VBN={}
    inv_VBP={}
    inv_VBZ={}
    s_ls=['NNS','NNPS','JJR','JJS','RBR','RBS','VBD','VBG','VBN','VBP','VBZ']
    s_noun=['NNS','NNPS']
    s_verb=['VBD','VBG','VBN','VBP','VBZ']
    s_adj=['JJR','JJS']
    s_adv=['RBR','RBS']
    # f=open('pos_tags.pkl','rb')
    # all_pos_tag=pickle.load(f)
    print(len(all_pos_tag))
    for idx in tqdm(range(len(all_pos_tag))):

        # print(idx)
        #text=train_text[idx]
        pos_tags = all_pos_tag[idx]
        for i in range(len(pos_tags)):
            pair=pos_tags[i]
            if pair[1] in s_ls:
                if pair[1][:2]=='NN':
                    w=wnl.lemmatize(pair[0],pos='n')
                elif pair[1][:2]=='VB':
                    w = wnl.lemmatize(pair[0], pos='v')
                elif pair[1][:2]=='JJ':
                    w = wnl.lemmatize(pair[0], pos='a')
                else:
                    w = wnl.lemmatize(pair[0], pos='r')
                eval('inv_'+pair[1])[w]=pair[0]
                eval(pair[1])[pair[0]]=w

    ret = (NNS,NNPS,JJR,JJS,RBR,RBS,VBD,VBG,VBN,VBP,VBZ,inv_NNS,inv_NNPS,inv_JJR,inv_JJS,inv_RBR,inv_RBS,inv_VBD,inv_VBG,inv_VBN,inv_VBP,inv_VBZ)

    my_file.save_pkl(ret, cache_path)

    return ret
    # f=open('sss_dict.pkl','wb')
    # pickle.dump((NNS,NNPS,JJR,JJS,RBR,RBS,VBD,VBG,VBN,VBP,VBZ,inv_NNS,inv_NNPS,inv_JJR,inv_JJS,inv_RBR,inv_RBS,inv_VBD,inv_VBG,inv_VBN,inv_VBP,inv_VBZ),f)


# # ========================== Generate Candidates =========================================

def _add_w1(w1, i1, dict_, word_candidate, word_influction, word_pos, word_sem):
    s_ls, s_noun, s_verb, s_adj, s_adv, NNS, NNPS, JJR, JJS, RBR, RBS, VBD, VBG, VBN, VBP, VBZ, inv_NNS, inv_NNPS, inv_JJR, inv_JJS, inv_RBR, inv_RBS, \
    inv_VBD, inv_VBG, inv_VBN, inv_VBP, inv_VBZ = all_tmp


    word_candidate[i1] = {}
    w1_pos_sem=word_influction[i1] # stanford POS(str) of w1

    w1_pos = set(word_pos[i1]) # sememe POS of w1
    for pos in pos_set:
        word_candidate[i1][pos] = []
    valid_pos_w1 = w1_pos & pos_set  # {sememe POS of w1} & {'noun', 'verb', 'adj', 'adv'}

    if len(valid_pos_w1) == 0:
        return

    new_w1_sememes = word_sem[i1] # list of sememe syno(str) of w1
    if len(new_w1_sememes) == 0:
        return

    for w2, i2 in dict_.items():
        if i2 > 50000:
            continue
        if i1 == i2:
            continue
        w2_pos_sem=word_influction[i2] # stanford POS(str) of w2

        w2_pos = set(word_pos[i2]) # sememe POS of w2
        all_pos = w2_pos & w1_pos & pos_set # w1, w2 same pos
        if len(all_pos) == 0:
            continue

        new_w2_sememes = word_sem[i2] # list of sememe syno(str) of w1
        # print(w2)
        # print(new_w1_sememes)
        # print(new_w2_sememes)
        if len(new_w2_sememes) == 0:
            continue
        # not_in_num1 = count(w1_sememes, w2_sememes)
        # not_in_num2 = count(w2_sememes,w1_sememes)
        # not_in_num=not_in_num1+not_in_num2
        w_flag=0

        for s1_id in range(len(new_w1_sememes)):
            if w_flag == 1:
                break
            pos_w1 = word_pos[i1][s1_id]
            s1 = set(new_w1_sememes[s1_id])
            if pos_w1 not in pos_set:
                continue
            for s2_id in range(len(new_w2_sememes)):
                if w_flag==1:
                    break
                pos_w2 = word_pos[i2][s2_id]
                s2 = set(new_w2_sememes[s2_id])
                if pos_w1 == pos_w2 and s1 == s2:
                    if w1_pos_sem == 'orig':
                        if w2_pos_sem == 'orig':
                            word_candidate[i1][pos_w1].append(i2)
                            w_flag=1
                            break
                    else:
                        for p in eval('s_' + pos_w1):
                            if w1 in eval(p) and w2 in eval(p):
                                word_candidate[i1][pos_w1].append(i2)
                                w_flag=1
                                break


def generate_candidates(dataset_name, lemma_results, dict_, output_dir):
    print('Generate Candidates')
    save_path = os.path.join(output_dir, dataset_name + "_candidates.pkl")
    if os.path.exists(save_path):
        print('Already generate candidates of', dataset_name)
        return save_path


    word_candidate = {}
    hownet_dict = OpenHowNet.HowNetDict()

    NNS, NNPS, JJR, JJS, RBR, RBS, VBD, VBG, VBN, VBP, VBZ, inv_NNS, inv_NNPS, inv_JJR, inv_JJS, inv_RBR, inv_RBS,\
    inv_VBD, inv_VBG, inv_VBN, inv_VBP, inv_VBZ = lemma_results

    pos_list = ['noun', 'verb', 'adj', 'adv']
    global pos_set
    pos_set = set(pos_list)

    s_ls = ['NNS', 'NNPS', 'JJR', 'JJS', 'RBR', 'RBS', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    s_noun = ['NNS', 'NNPS']
    s_verb = ['VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    s_adj = ['JJR', 'JJS']
    s_adv = ['RBR', 'RBS']
    global all_tmp
    all_tmp = (s_ls, s_noun, s_verb, s_adj, s_adv, NNS, NNPS, JJR, JJS, RBR, RBS, VBD, VBG, VBN, VBP, VBZ, inv_NNS, inv_NNPS, inv_JJR, inv_JJS, inv_RBR, inv_RBS,\
    inv_VBD, inv_VBG, inv_VBN, inv_VBP, inv_VBZ)

    word_pos = {} # {token_idx(int) : sememe POS?}
    word_sem = {} # {token_idx(int) : list of sememe syno(str)}
    word_influction={} # token_idx(int) : stanford POS(str)
    print('Prepare Sememe ...')

    inner_path = os.path.join(output_dir, dataset_name + "_candidates_inner.pkl")
    if os.path.exists(inner_path):
        word_pos, word_sem, word_influction = my_file.load_pkl(inner_path)
    else:
        for w1, i1 in tqdm(dict_.items()):
            w1_s_flag = 0
            w1_orig = None
            for s in s_ls:
                if w1 in eval(s):
                    w1_s_flag = 1
                    w1_orig = eval(s)[w1] # lemma of w1
                    word_influction[i1]=s
                    break
            if w1_s_flag == 0:
                w1_orig = w1
                word_influction[i1] = 'orig'
            try:
                tree = hownet_dict.get_sememes_by_word(w1_orig, merge=False, structured=True, lang="en")
                w1_sememes = hownet_dict.get_sememes_by_word(w1_orig, structured=False, lang="en", merge=False)
                new_w1_sememes = [t['sememes'] for t in w1_sememes]
                # print(tree)

                w1_pos_list = [x['word']['en_grammar'] for x in tree]
                word_pos[i1] = w1_pos_list
                word_sem[i1] = new_w1_sememes
                main_sememe_list = hownet_dict.get_sememes_by_word(w1_orig, merge=False, structured=False, lang='en',
                                                                   expanded_layer=2)
            except:
                word_pos[i1] = []
                word_sem[i1] = []
                main_sememe_list = []
            # assert len(w1_pos_list)==len(new_w1_sememes)
            # assert len(w1_pos_list)==len(main_sememe_list)

        my_file.save_pkl((word_pos, word_sem, word_influction), inner_path)

    print('Generate Candidates')
    print(len(dict_))
    for w1, i1 in tqdm(dict_.items()):
        if i1>50000:
            continue

        _add_w1(w1, i1, dict_, word_candidate, word_influction, word_pos, word_sem)



    my_file.save_pkl(word_candidate, save_path)
    # f = open('word_candidates_sense.pkl', 'wb')
    # pickle.dump(word_candidate, f)
    print('Full Sememe Candidates Generated')

    return word_candidate


# def load_raw_sememe(preprocess_dir, target_dataset):
#     processed_dataset_path = os.path.join(preprocess_dir, 'sememe', target_dataset + '_dataset.pkl')
#     processed_dataset = my_file.load_pkl(processed_dataset_path)
#
#     sememe_candidates_path = os.path.join(preprocess_dir, 'sememe', target_dataset + '_candidates.pkl')
#     sememe_candidates = my_file.load_pkl(sememe_candidates_path)
#
#     return processed_dataset, sememe_candidates


def _pos_convert(stanford_pos):
    check_pos = stanford_pos[:2]

    if check_pos not in stanford_pos_set:
        return None

    if check_pos == 'JJ':
        pos = 'adj'
    elif check_pos == 'NN':
        pos = 'noun'
    elif check_pos == 'RB':
        pos = 'adv'
    elif check_pos == 'VB':
        pos = 'verb'
    else:
        raise Exception('Pos: ', stanford_pos)


    return pos


def _process_one(one_sent, word2idx, idx2word, sememe_candidates):
    jar = STANFORD_POS_TAGGER_JAR
    model = STANFORD_POS_TAGGER_MODEL
    pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')

    pos_tags = pos_tagger.tag(one_sent)

    idx_word_perturb_list = []
    sememe_dict = {}

    for idx, orig_word in enumerate(one_sent):
        # if hasattr(processed_dataset, 'full_dict'):
        #     word2idx = processed_dataset.full_dict
        #     idx2word = processed_dataset.inv_full_dict
        # else:
        #     word2idx = {w: i for (w, i) in processed_dataset.word_index.items()}
        #     idx2word = {i: w for (w, i) in processed_dataset.word_index.items()}

        if orig_word not in word2idx:
            continue

        orig_word_token_idx = word2idx[orig_word]
        if orig_word_token_idx >= 50000:
            continue

        # get pos
        _pos = pos_tags[idx][1]
        c_pos = _pos_convert(_pos)
        if c_pos is None or c_pos not in sememe_candidates[orig_word_token_idx]:
            continue

        # get sememe candidates
        sememe_token_idx_list = sememe_candidates[orig_word_token_idx][c_pos]
        if len(sememe_token_idx_list) > 0:
            sememe_word_list = [idx2word[sememe_idx] for sememe_idx in sememe_token_idx_list]

            idx_word_perturb_list.append((idx, orig_word))
            sememe_dict[(idx, orig_word)] = sememe_word_list

    return idx_word_perturb_list, sememe_dict


def sememe_preprocess(output_dir, dataset, process_data_path, orig_dataset_path):

    cand_path = os.path.join(output_dir, dataset + "_candidates.pkl")
    processed_data_pkl_path = os.path.join(output_dir, dataset + "_dataset.pkl")
    if os.path.exists(cand_path) and os.path.exists(processed_data_pkl_path):
        sememe_candidates = my_file.load_pkl(cand_path)
        train_seqs, dict_, inv_dict, full_dict, inv_full_dict, max_vocab_size = my_file.load_pkl(processed_data_pkl_path)
    else:
        # prepare full candidates dict
        train_seqs, dict_, inv_dict, full_dict, inv_full_dict, max_vocab_size =\
            build_dataset(dataset, orig_dataset_path, output_dir)
        all_pos_tag = generate_pos(dataset, train_seqs, inv_full_dict)
        lemma_results = lemma(dataset, all_pos_tag)
        sememe_candidates = generate_candidates(dataset, lemma_results, dict_, output_dir)



    res_list = []
    if dataset in ['snli', 'mnli', 'mnli_matched', 'mnli_mismatched']:

        from dataloader import read_data_nli

        data = read_data_nli(process_data_path, data_size=999999999)

        for idx, premise in tqdm(enumerate(data['premises'])):

            hypothesis, true_label = data['hypotheses'][idx], data['labels'][idx]
            idx_word_list, syno_dict = _process_one(hypothesis, full_dict, inv_full_dict, sememe_candidates)
            res_list.append((idx_word_list, syno_dict))


    elif dataset in ['imdb', 'agnews', 'yahoo', 'mr']:

        import dataloader
        texts, labels = dataloader.read_corpus(process_data_path, csvf=False)
        data = list(zip(texts, labels))

        for idx, (text, true_label) in tqdm(enumerate(data)):
            idx_word_list, syno_dict = _process_one(text, full_dict, inv_full_dict, sememe_candidates)
            res_list.append((idx_word_list, syno_dict))


    save_path = os.path.join(output_dir, process_data_path.split('/')[-1])
    with open(save_path, 'wb') as f:
        pickle.dump(res_list, f)

    print(f'Sememe on {dataset} preprocess finished!')
    print('Save to', save_path)


