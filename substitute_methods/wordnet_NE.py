'''
Reimplement of subsitute words used by PWWS (Generating natural language adversarial examples through probability weighted word saliency)
WordNet Synonyms + Named Entities (NE)
'''
import os
from functools import partial
import pickle
import re
from collections import Counter, defaultdict
import copy
import string

from tqdm import tqdm
from nltk.corpus import wordnet as wn
import spacy

from utils import my_file
from local_models.NLI_config import NLI_LABEL_NUM2STR, NLI_LABEL_STR2NUM

def read_nli_data(filepath, lowercase=False, ignore_punctuation=False, stopwords=[]):
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

    with open(filepath, 'r', encoding='utf8') as input_data:
        premises, hypotheses, labels = [], [], []

        # Translation tables to remove punctuation from strings.
        punct_table = str.maketrans({key: ' '
                                     for key in string.punctuation})

        for idx, line in enumerate(input_data):

            line = line.strip().split('\t')

            # Ignore sentences that have no gold label.
            if line[0] == '-':
                continue

            premise = line[1]
            hypothesis = line[2]

            if lowercase:
                premise = premise.lower()
                hypothesis = hypothesis.lower()

            if ignore_punctuation:
                premise = premise.translate(punct_table)
                hypothesis = hypothesis.translate(punct_table)

            # Each premise and hypothesis is split into a list of words.
            premises.append([w for w in premise.rstrip().split()
                             if w not in stopwords])
            hypotheses.append([w for w in hypothesis.rstrip().split()
                               if w not in stopwords])
            labels.append(line[0])

        return {"premises": premises,
                "hypotheses": hypotheses,
                "labels": labels}


nlp_ne_generate = spacy.load('en')
nlp = spacy.load('en')

num_classes = {'imdb': 2, 'yahoo': 10, 'agnews': 4, 'snli': 3, 'mr': 2, 'mnli': 3}

# snli_label_idx2str = {0: "contradiction", 1: "entailment", 2: "neutral"}
# snli_label_str2idx = {'contradiction': 0, 'entailment': 1, 'neutral': 2}

supported_pos_tags = {
    'CC',  # coordinating conjunction, like "and but neither versus whether yet so"
    # 'CD',   # Cardinal number, like "mid-1890 34 forty-two million dozen"
    # 'DT',   # Determiner, like all "an both those"
    # 'EX',   # Existential there, like "there"
    # 'FW',   # Foreign word
    # 'IN',   # Preposition or subordinating conjunction, like "among below into"
    'JJ',  # Adjective, like "second ill-mannered"
    'JJR',  # Adjective, comparative, like "colder"
    'JJS',  # Adjective, superlative, like "cheapest"
    # 'LS',   # List item marker, like "A B C D"
    # 'MD',   # Modal, like "can must shouldn't"
    'NN',  # Noun, singular or mass
    'NNS',  # Noun, plural
    'NNP',  # Proper noun, singular
    'NNPS',  # Proper noun, plural
    # 'PDT',  # Predeterminer, like "all both many"
    # 'POS',  # Possessive ending, like "'s"
    # 'PRP',  # Personal pronoun, like "hers herself ours they theirs"
    # 'PRP$',  # Possessive pronoun, like "hers his mine ours"
    'RB',  # Adverb
    'RBR',  # Adverb, comparative, like "lower heavier"
    'RBS',  # Adverb, superlative, like "best biggest"
    # 'RP',   # Particle, like "board about across around"
    # 'SYM',  # Symbol
    # 'TO',   # to
    # 'UH',   # Interjection, like "wow goody"
    'VB',  # Verb, base form
    'VBD',  # Verb, past tense
    'VBG',  # Verb, gerund or present participle
    'VBN',  # Verb, past participle
    'VBP',  # Verb, non-3rd person singular present
    'VBZ',  # Verb, 3rd person singular present
    # 'WDT',  # Wh-determiner, like "that what whatever which whichever"
    # 'WP',   # Wh-pronoun, like "that who"
    # 'WP$',  # Possessive wh-pronoun, like "whose"
    # 'WRB',  # Wh-adv  erb, like "however wherever whenever"
}

NE_type_dict = {
    'PERSON': defaultdict(int),  # People, including fictional.
    'NORP': defaultdict(int),  # Nationalities or religious or political groups.
    'FAC': defaultdict(int),  # Buildings, airports, highways, bridges, etc.
    'ORG': defaultdict(int),  # Companies, agencies, institutions, etc.
    'GPE': defaultdict(int),  # Countries, cities, states.
    'LOC': defaultdict(int),  # Non-GPE locations, mountain ranges, bodies of water.
    'PRODUCT': defaultdict(int),  # Object, vehicles, foods, etc.(Not services)
    'EVENT': defaultdict(int),  # Named hurricanes, battles, wars, sports events, etc.
    'WORK_OF_ART': defaultdict(int),  # Titles of books, songs, etc.
    'LAW': defaultdict(int),  # Named documents made into laws.
    'LANGUAGE': defaultdict(int),  # Any named language.
    'DATE': defaultdict(int),  # Absolute or relative dates or periods.
    'TIME': defaultdict(int),  # Times smaller than a day.
    'PERCENT': defaultdict(int),  # Percentage, including "%".
    'MONEY': defaultdict(int),  # Monetary values, including unit.
    'QUANTITY': defaultdict(int),  # Measurements, as of weight or distance.
    'ORDINAL': defaultdict(int),  # "first", "second", etc.
    'CARDINAL': defaultdict(int),  # Numerals that do not fall under another type.
}

# Hard Code NE
# Warning: NE lists of {IMBD, Yahoo, AG News} are directly copied from raw codes of PWWS (https://github.com/JHL-HUST/PWWS).
#          The results of running the generation codes are not same as the following.
class NameEntityList(object):
    # If the original input in IMDB belongs to class 0 (negative)
    imdb_0 = {'PERSON': 'David',
              'NORP': 'Australian',
              'FAC': 'Hound',
              'ORG': 'Ford',
              'GPE': 'India',
              'LOC': 'Atlantic',
              'PRODUCT': 'Highly',
              'EVENT': 'Depression',
              'WORK_OF_ART': 'Casablanca',
              'LAW': 'Constitution',
              'LANGUAGE': 'Portuguese',
              'DATE': '2001',
              'TIME': 'hours',
              'PERCENT': '98%',
              'MONEY': '4',
              'QUANTITY': '70mm',
              'ORDINAL': '5th',
              'CARDINAL': '7',
              }
    # If the original input in IMDB belongs to class 1 (positive)
    imdb_1 = {'PERSON': 'Lee',
              'NORP': 'Christian',
              'FAC': 'Shannon',
              'ORG': 'BAD',
              'GPE': 'Seagal',
              'LOC': 'Malta',
              'PRODUCT': 'Cat',
              'EVENT': 'Hugo',
              'WORK_OF_ART': 'Jaws',
              'LAW': 'RICO',
              'LANGUAGE': 'Sebastian',
              'DATE': 'Friday',
              'TIME': 'minutes',
              'PERCENT': '75%',
              'MONEY': '$',
              'QUANTITY': '9mm',
              'ORDINAL': 'sixth',
              'CARDINAL': 'zero',
              }
    imdb = [imdb_0, imdb_1]
    agnews_0 = {'PERSON': 'Williams',
                'NORP': 'European',
                'FAC': 'Olympic',
                'ORG': 'Microsoft',
                'GPE': 'Australia',
                'LOC': 'Earth',
                'PRODUCT': '#',
                'EVENT': 'Cup',
                'WORK_OF_ART': 'PowerBook',
                'LAW': 'Pacers-Pistons',
                'LANGUAGE': 'Chinese',
                'DATE': 'third-quarter',
                'TIME': 'Tonight',
                'MONEY': '#39;t',
                'QUANTITY': '#39;t',
                'ORDINAL': '11th',
                'CARDINAL': '1',
                }
    agnews_1 = {'PERSON': 'Bush',
                'NORP': 'Iraqi',
                'FAC': 'Outlook',
                'ORG': 'Microsoft',
                'GPE': 'Iraq',
                'LOC': 'Asia',
                'PRODUCT': '#',
                'EVENT': 'Series',
                'WORK_OF_ART': 'Nobel',
                'LAW': 'Constitution',
                'LANGUAGE': 'French',
                'DATE': 'third-quarter',
                'TIME': 'hours',
                'MONEY': '39;Keefe',
                'ORDINAL': '2nd',
                'CARDINAL': 'Two',
                }
    agnews_2 = {'PERSON': 'Arafat',
                'NORP': 'Iraqi',
                'FAC': 'Olympic',
                'ORG': 'AFP',
                'GPE': 'Baghdad',
                'LOC': 'Earth',
                'PRODUCT': 'Soyuz',
                'EVENT': 'Cup',
                'WORK_OF_ART': 'PowerBook',
                'LAW': 'Constitution',
                'LANGUAGE': 'Filipino',
                'DATE': 'Sunday',
                'TIME': 'evening',
                'MONEY': '39;m',
                'QUANTITY': '20km',
                'ORDINAL': 'eighth',
                'CARDINAL': '6',
                }
    agnews_3 = {'PERSON': 'Arafat',
                'NORP': 'Iraqi',
                'FAC': 'Olympic',
                'ORG': 'AFP',
                'GPE': 'Iraq',
                'LOC': 'Kashmir',
                'PRODUCT': 'Yukos',
                'EVENT': 'Cup',
                'WORK_OF_ART': 'Gazprom',
                'LAW': 'Pacers-Pistons',
                'LANGUAGE': 'Hebrew',
                'DATE': 'Saturday',
                'TIME': 'overnight',
                'MONEY': '39;m',
                'QUANTITY': '#39;t',
                'ORDINAL': '11th',
                'CARDINAL': '6',
                }
    agnews = [agnews_0, agnews_1, agnews_2, agnews_3]
    yahoo_0 = {'PERSON': 'Fantasy',
               'NORP': 'Russian',
               'FAC': 'Taxation',
               'ORG': 'Congress',
               'GPE': 'U.S.',
               'LOC': 'Sea',
               'PRODUCT': 'Variable',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Stopping',
               'LAW': 'Constitution',
               'LANGUAGE': 'Hebrew',
               'DATE': '2004-05',
               'TIME': 'morning',
               'MONEY': '$ale',
               'QUANTITY': 'Hiberno-English',
               'ORDINAL': 'Tertiary',
               'CARDINAL': 'three',
               }
    yahoo_1 = {'PERSON': 'Equine',
               'NORP': 'Japanese',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'UK',
               'LOC': 'Sea',
               'PRODUCT': 'RuneScape',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Stopping',
               'LAW': 'Strap-',
               'LANGUAGE': 'Spanish',
               'DATE': '2004-05',
               'TIME': 'night',
               'PERCENT': '100%',
               'MONEY': 'five-dollar',
               'QUANTITY': 'Hiberno-English',
               'ORDINAL': 'Sixth',
               'CARDINAL': '5',
               }
    yahoo_2 = {'PERSON': 'Equine',
               'NORP': 'Canadian',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'California',
               'LOC': 'Atlantic',
               'PRODUCT': 'Variable',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Weight',
               'LANGUAGE': 'Filipino',
               'DATE': '2004-05',
               'TIME': 'night',
               'PERCENT': '100%',
               'MONEY': 'ten-dollar',
               'QUANTITY': '$ale',
               'ORDINAL': 'Tertiary',
               'CARDINAL': 'two',
               }
    yahoo_3 = {'PERSON': 'Equine',
               'NORP': 'Irish',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'California',
               'LOC': 'Sea',
               'PRODUCT': 'RuneScape',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Weight',
               'LAW': 'Strap-',
               'LANGUAGE': 'Spanish',
               'DATE': '2004-05',
               'TIME': 'tonight',
               'PERCENT': '100%',
               'MONEY': 'five-dollar',
               'QUANTITY': 'Hiberno-English',
               'ORDINAL': 'Sixth',
               'CARDINAL': '5',
               }
    yahoo_4 = {'PERSON': 'Equine',
               'NORP': 'Irish',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'Canada',
               'LOC': 'Sea',
               'PRODUCT': 'Variable',
               'WORK_OF_ART': 'Stopping',
               'LAW': 'Constitution',
               'LANGUAGE': 'Spanish',
               'DATE': '2004-05',
               'TIME': 'seconds',
               'PERCENT': '100%',
               'MONEY': 'hundred-dollar',
               'QUANTITY': 'Hiberno-English',
               'ORDINAL': 'Tertiary',
               'CARDINAL': '100',
               }
    yahoo_5 = {'PERSON': 'Equine',
               'NORP': 'English',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'Australia',
               'LOC': 'Sea',
               'PRODUCT': 'Variable',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Weight',
               'LAW': 'Strap-',
               'LANGUAGE': 'Filipino',
               'DATE': '2004-05',
               'TIME': 'seconds',
               'MONEY': 'hundred-dollar',
               'ORDINAL': 'Tertiary',
               'CARDINAL': '2000',

               }
    yahoo_6 = {'PERSON': 'Fantasy',
               'NORP': 'Islamic',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'California',
               'LOC': 'Sea',
               'PRODUCT': 'Variable',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Stopping',
               'LANGUAGE': 'Filipino',
               'DATE': '2004-05',
               'TIME': 'seconds',
               'PERCENT': '100%',
               'MONEY': '$ale',
               'QUANTITY': '$ale',
               'ORDINAL': 'Tertiary',
               'CARDINAL': '100',
               }
    yahoo_7 = {'PERSON': 'Fantasy',
               'NORP': 'Canadian',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'UK',
               'LOC': 'West',
               'PRODUCT': 'Variable',
               'EVENT': 'Watergate',
               'WORK_OF_ART': 'Stopping',
               'LAW': 'Constitution',
               'LANGUAGE': 'Filipino',
               'DATE': '2004-05',
               'TIME': 'tonight',
               'PERCENT': '100%',
               'MONEY': '$ale',
               'QUANTITY': '$ale',
               'ORDINAL': 'Tertiary',
               'CARDINAL': '2000',
               }
    yahoo_8 = {'PERSON': 'Equine',
               'NORP': 'Japanese',
               'FAC': 'Music',
               'ORG': 'Congress',
               'GPE': 'Chicago',
               'LOC': 'Sea',
               'PRODUCT': 'Variable',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Stopping',
               'LAW': 'Strap-',
               'LANGUAGE': 'Spanish',
               'DATE': '2004-05',
               'TIME': 'night',
               'PERCENT': '100%',
               'QUANTITY': '$ale',
               'ORDINAL': 'Sixth',
               'CARDINAL': '2',

               }
    yahoo_9 = {'PERSON': 'Equine',
               'NORP': 'Chinese',
               'FAC': 'Music',
               'ORG': 'Digital',
               'GPE': 'U.S.',
               'LOC': 'Atlantic',
               'PRODUCT': 'Variable',
               'EVENT': 'Series',
               'WORK_OF_ART': 'Weight',
               'LAW': 'Constitution',
               'LANGUAGE': 'Spanish',
               'DATE': '1918-1945',
               'TIME': 'night',
               'PERCENT': '100%',
               'MONEY': 'ten-dollar',
               'QUANTITY': 'Hiberno-English',
               'ORDINAL': 'Tertiary',
               'CARDINAL': '5'
               }
    yahoo = [yahoo_0, yahoo_1, yahoo_2, yahoo_3, yahoo_4, yahoo_5, yahoo_6, yahoo_7, yahoo_8, yahoo_9]

    mr_0 = {
        'PERSON': 'spielberg',
        'NORP': 'iranian',
        'FAC': 'auschwitz',
        'ORG': 'miller',
        'GPE': 'china',
        'LOC': 'atlantic',
        'PRODUCT': 'classic',
        'WORK_OF_ART': 'moretti',
        'LANGUAGE': 'japanese',
        'DATE': 'years',
        'TIME': 'hours',
        'QUANTITY': 'tons',
        'ORDINAL': 'tenth',
        'CARDINAL': '13'
    }
    mr_1 = {
        'PERSON': 'madonna',
        'NORP': 'irish',
        'FAC': 'mummy',
        'ORG': 'harvard',
        'GPE': 'turkey',
        'LOC': 'mars',
        'PRODUCT': 'pg-13',
        'WORK_OF_ART': 'bible',
        'LAW': 'vapid',
        'DATE': 'friday',
        'TIME': '88-minute',
        'PERCENT': '100%',
        'MONEY': '451',
        'QUANTITY': '102-minute',
        'ORDINAL': 'eighth',
        'CARDINAL': '51'
    }
    mr = [mr_0, mr_1]

    snli_contradiction = {
        'PERSON': 'Jenga',
        'NORP': 'indian',
        'FAC': 'metro',
        'ORG': 'Army',
        'GPE': 'Paris',
        'LOC': 'Alps',
        'PRODUCT': 'YouTube',
        'EVENT': 'Thanksgiving',
        'WORK_OF_ART': 'Crocs',
        'LAW': 'Ray-Ban',
        'LANGUAGE': 'Chinese',
        'DATE': 'halloween',
        'TIME': 'dusk',
        'PERCENT': '50%',
        'MONEY': '500',
        'QUANTITY': 'tons',
        'ORDINAL': 'fifth',
        'CARDINAL': 'six'
    }
    snli_entailment = {
        'PERSON': 'Legos',
        'NORP': 'hispanic',
        'FAC': 'Broadway',
        'ORG': 'NYC',
        'GPE': 'Hawaii',
        'LOC': 'Mars',
        'PRODUCT': 'Gatorade',
        'EVENT': 'olympics',
        'WORK_OF_ART': 'Bible',
        'LAW': 'Ray-Ban',
        'LANGUAGE': 'russian',
        'DATE': 'halloween',
        'TIME': 'midnight',
        'PERCENT': '100%',
        'MONEY': '100',
        'QUANTITY': 'tons',
        'ORDINAL': '2nd',
        'CARDINAL': '4'
    }
    snli_neutral = {
        'PERSON': 'Jesus',
        'NORP': 'indian',
        'FAC': 'broadway',
        'ORG': 'cat',
        'GPE': 'turkey',
        'LOC': 'Mars',
        'PRODUCT': 'Saturn',
        'WORK_OF_ART': 'YMCA',
        'LANGUAGE': 'Arabic',
        'DATE': 'day',
        'TIME': 'noon',
        'PERCENT': '100%',
        'MONEY': '75',
        'QUANTITY': 'five',
        'ORDINAL': 'Third',
        'CARDINAL': '4'
    }

    snli = {'neutral': snli_neutral, 'entailment': snli_entailment, 'contradiction': snli_contradiction}

    mnli_contradiction = {
        'PERSON': 'Brown',
        'NORP': 'European',
        'FAC': 'Stick',
        'ORG': 'Lincoln',
        'GPE': 'Israel',
        'LOC': 'Kashmir',
        'PRODUCT': 'Holocaust',
        'EVENT': 'Series',
        'WORK_OF_ART': 'Child',
        'LAW': 'Act',
        'LANGUAGE': 'Malay',
        'DATE': 'daily',
        'TIME': 'afternoon',
        'PERCENT': '25%',
        'MONEY': '2',
        'QUANTITY': 'gallon',
        'ORDINAL': '3rd',
        'CARDINAL': 'Two'
    }

    mnli_entailment = {
        'PERSON': 'Brown',
        'NORP': 'Christian',
        'FAC': 'Disneyland',
        'ORG': 'Clinton',
        'GPE': 'Egypt',
        'LOC': 'Mars',
        'PRODUCT': 'Windows',
        'EVENT': 'Series',
        'WORK_OF_ART': 'Crossfire',
        'LANGUAGE': 'American',
        'DATE': 'tomorrow',
        'TIME': 'noon',
        'PERCENT': '40%',
        'MONEY': '500',
        'QUANTITY': 'ten',
        'ORDINAL': '3rd',
        'CARDINAL': 'ten'
    }

    mnli_neutral = {
        'PERSON': 'Brown',
        'NORP': 'European',
        'FAC': 'Dome',
        'ORG': 'OMB',
        'GPE': 'Israel',
        'LOC': 'East',
        'PRODUCT': 'Galileo',
        'EVENT': 'WWII',
        'WORK_OF_ART': 'Crossfire',
        'LANGUAGE': 'Malay',
        'DATE': '1996',
        'TIME': 'afternoon',
        'PERCENT': '1%',
        'MONEY': '10',
        'QUANTITY': '6pm',
        'ORDINAL': 'tenth',
        'CARDINAL': 'Two'
    }
    mnli = {'contradiction': mnli_contradiction, 'entailment': mnli_entailment, 'neutral': mnli_neutral}

    L = {'imdb': imdb, 'agnews': agnews, 'yahoo': yahoo, 'snli': snli, 'mr': mr, 'mnli': mnli}

NE_list = NameEntityList()

def recognize_named_entity(texts):
    '''
    Returns all NEs in the input texts and their corresponding types
    '''
    NE_freq_dict = copy.deepcopy(NE_type_dict)

    for text in tqdm(texts):
        doc = nlp_ne_generate(text)
        for word in doc.ents:
            NE_freq_dict[word.label_][word.text] += 1
    return NE_freq_dict

def find_adv_NE(D_true, D_other, dataset, class_idx):
    '''
    find NE_adv in D-D_y_true which is defined in the end of section 3.1
    '''
    output_file_path = my_file.real_path_of('preprocess_data/wordnet_NE', f'{dataset}_NE.txt')
    print('\nfind adv_NE_list in class', class_idx)
    with open(output_file_path, 'a', encoding='utf-8') as f:
        f.write('\nfind adv_NE_list in class' + str(class_idx))

    # adv_NE_list = []
    for type in NE_type_dict.keys():
        # find the most frequent true and other NEs of the same type
        true_NE_list = [NE_tuple[0] for (i, NE_tuple) in enumerate(D_true[type]) if i < 15]
        other_NE_list = [NE_tuple[0] for (i, NE_tuple) in enumerate(D_other[type]) if i < 30]

        for other_NE in other_NE_list:
            if other_NE not in true_NE_list and len(other_NE.split()) == 1:
                # adv_NE_list.append((type, other_NE))
                print("'" + type + "': '" + other_NE + "',")
                with open(output_file_path, 'a', encoding='utf-8') as f:
                    f.write("'" + type + "': '" + other_NE + "',\n")
                break

def generate_NE(dataset, dataset_path):
    class_num = num_classes[dataset]

    if dataset == 'imdb':
        # use NE candidates provided by source code
        return

    elif dataset == 'agnews':
        # use NE candidates provided by source code
        return
    elif dataset == 'yahoo':
        # use NE candidates provided by source code
        return
    elif dataset == 'mr':
        from dataloader import read_orig_mr
        text_list, label_list = read_orig_mr(dataset_path)

        texts = [[] for i in range(class_num)]
        for i in range(len(text_list)):
            texts[label_list[i]].append(text_list[i])
    elif dataset == 'snli':
        from dataloader import read_orig_snli
        data_list = read_orig_snli(dataset_path, 'train')
        data_list.extend(read_orig_snli(dataset_path, 'dev'))
        data_list.extend(read_orig_snli(dataset_path, 'test'))

        texts = [[] for i in range(class_num)]

        for label_str, premise, hypo in data_list:
            texts[NLI_LABEL_STR2NUM[label_str]].append(hypo)
    elif dataset == 'mnli':
        from dataloader import read_orig_mnli
        data_list = read_orig_mnli(dataset_path, 'train')
        data_list.extend(read_orig_mnli(dataset_path, 'dev_matched'))
        data_list.extend(read_orig_mnli(dataset_path, 'dev_mismatched'))

        texts = [[] for i in range(class_num)]

        for label_str, premise, hypo, _ in data_list:
            texts[NLI_LABEL_STR2NUM[label_str]].append(hypo)

    D_true_list = []
    for i in range(class_num):
        D_true = recognize_named_entity(texts[i])  # D_true contains the NEs in input texts with the label y_true
        D_true_list.append(D_true)

    for i in range(class_num):
        D_true = copy.deepcopy(D_true_list[i])
        D_other = copy.deepcopy(NE_type_dict)
        for j in range(class_num):
            if i == j:
                continue
            for type in NE_type_dict.keys():
                # combine D_other[type] and D_true_list[j][type]
                for key in D_true_list[j][type].keys():
                    D_other[type][key] += D_true_list[j][type][key]
        for type in NE_type_dict.keys():
            D_other[type] = sorted(D_other[type].items(), key=lambda k_v: k_v[1], reverse=True)
            D_true[type] = sorted(D_true[type].items(), key=lambda k_v: k_v[1], reverse=True)

        class_idx = i
        if dataset == 'snli' or dataset == 'mnli':
            class_idx = NLI_LABEL_NUM2STR[i]
        find_adv_NE(D_true, D_other, dataset, class_idx)

def _get_wordnet_pos(spacy_token):
    '''Wordnet POS tag'''
    pos = spacy_token.tag_[0].lower()
    if pos in ['r', 'n', 'v']:  # adv, noun, verb
        return pos
    elif pos == 'j':
        return 'a'  # adj

def _synonym_prefilter_fn(token, synonym):
    '''
    Similarity heuristics go here
    '''
    if (len(synonym.text.split()) > 2 or (  # the synonym produced is a phrase
            synonym.lemma == token.lemma) or (  # token and synonym are the same
            synonym.tag != token.tag) or (  # the pos of the token synonyms are different
            token.text.lower() == 'be')):  # token is be
        return False
    else:
        return True

def _generate_synonym_candidates(token, _token_position=None):
    '''
    Generate synonym candidates.
    For each token in the doc, the list of WordNet synonyms is expanded.
    :return candidates, a list, whose type of element is <class '__main__.SubstitutionCandidate'>
            like SubstitutionCandidate(token_position=0, similarity_rank=10, original_token=Soft, candidate_word='subdued')
    '''
    candidates = []
    if token.tag_ in supported_pos_tags:
        wordnet_pos = _get_wordnet_pos(token)  # 'r', 'a', 'n', 'v' or None
        wordnet_synonyms = []

        synsets = wn.synsets(token.text, pos=wordnet_pos)
        for synset in synsets:
            wordnet_synonyms.extend(synset.lemmas())

        synonyms = []
        for wordnet_synonym in wordnet_synonyms:
            spacy_synonym = nlp(wordnet_synonym.name().replace('_', ' '))[0]
            synonyms.append(spacy_synonym)

        synonyms = filter(partial(_synonym_prefilter_fn, token), synonyms)

        candidate_set = set()
        for _, synonym in enumerate(synonyms):
            candidate_word = synonym.text
            if candidate_word in candidate_set:  # avoid repetition
                continue
            candidate_set.add(candidate_word)
            candidates.append(candidate_word)
    return candidates


def _process_one(orig_sent, true_label, dataset_NE_list):
    NE_candidates = dataset_NE_list[true_label]
    NE_tags = set(NE_candidates.keys())

    idx_word_perturb_list = []
    sub_words_dict = {}

    orig_text = " ".join(orig_sent)
    orig_doc = nlp(orig_text)

    spacy_token_list = []
    for token in orig_doc:
        spacy_token_list.append(token)

    # align
    idx_map_orig2spacy = {}
    if len(spacy_token_list) != len(orig_sent):
        spacy_idx = 0
        orig_idx = 0
        while orig_idx < len(orig_sent):
            orig_word = orig_sent[orig_idx]

            if orig_word == spacy_token_list[spacy_idx].text:
                idx_map_orig2spacy[orig_idx] = [spacy_idx]

            else:
                spacy_str = ""
                st_idx = spacy_idx
                while spacy_idx < len(spacy_token_list):
                    spacy_str += spacy_token_list[spacy_idx].text
                    if orig_word == spacy_str:
                        idx_map_orig2spacy[orig_idx] = list(range(st_idx, spacy_idx + 1))
                        break
                    spacy_idx += 1
            spacy_idx += 1
            orig_idx += 1

        valid_token_list = []
        for i, orig_word in enumerate(orig_sent):
            cur_spacy_list = idx_map_orig2spacy[i]
            if len(cur_spacy_list) > 1:
                valid_token_list.append(None)
            else:
                spacy_token = spacy_token_list[idx_map_orig2spacy[i][0]]
                assert orig_word == spacy_token.text
                valid_token_list.append(spacy_token)
        spacy_token_list = valid_token_list

    # generate candidates
    for word_idx, orig_word in enumerate(orig_sent):
        spacy_token = spacy_token_list[word_idx]

        if not spacy_token:
            continue

        NER_tag = spacy_token.ent_type_
        if NER_tag in NE_tags: #and spacy_token.tag_ in supported_NE_pos_tags:
            NE_sub_word = NE_candidates[NER_tag]
            if orig_word.islower():
                NE_sub_word = NE_sub_word.lower()
            elif orig_word.isupper():
                NE_sub_word = NE_sub_word.upper()

            if NE_sub_word == orig_word:
                continue

            cur_sub_words = [NE_sub_word]
            # print(orig_word, 'use NE:', cur_sub_words)
        else:
            cur_sub_words = _generate_synonym_candidates(spacy_token)


        if len(cur_sub_words) > 0:
            idx_word_perturb_list.append((word_idx, orig_word))
            sub_words_dict[(word_idx, orig_word)] = cur_sub_words



    return idx_word_perturb_list, sub_words_dict

def _process_one_sub(args):
    orig_sent, true_label, dataset_NE_list = args
    return _process_one(orig_sent, true_label, dataset_NE_list)

def _process_mp(data, dataset_NE_list):
    from multiprocessing import Pool
    arg_list = [(text, true_label, dataset_NE_list) for idx, (text, true_label) in enumerate(data)]
    num_workers = 20

    with Pool(num_workers) as p:
        outputs = p.map(_process_one_sub, arg_list)

    return outputs




def wordnet_NE_process(output_dir, dataset, dataset_path):
    print('Start WordNet + NameEntity Preprocessing')
    res_list = []
    if dataset[:4] == 'mnli':
        dataset_NE_list = NE_list.L['mnli']
    else:
        dataset_NE_list = NE_list.L[dataset]

    if dataset in ['snli', 'mnli', 'mnli_matched', 'mnli_mismatched']:

        data = read_nli_data(dataset_path)
        data_ = list(zip(data['hypotheses'], data['labels']))
        res_list = _process_mp(data_, dataset_NE_list)

    elif dataset in ['imdb', 'agnews', 'yahoo', 'mr']:
        import dataloader
        texts, labels = dataloader.read_corpus(dataset_path, csvf=False)
        data = list(zip(texts, labels))
        res_list = _process_mp(data, dataset_NE_list)

    save_path = os.path.join(output_dir, dataset)
    with open(save_path, 'wb') as f:
        pickle.dump(res_list, f)

    print(f'Sememe on {dataset} preprocess finished!')
    print('Save to', save_path)

if __name__ == '__main__':
    orig_dataset_path = 'path/to/orignal/dataset'
    dataset = 'dataset name'
    # generate Name Entity
    generate_NE(dataset, orig_dataset_path)


