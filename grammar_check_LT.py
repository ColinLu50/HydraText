import os
import glob

import language_tool_python
import pandas as pd
from tqdm import tqdm

tool = language_tool_python.LanguageTool('en-US')

ORIG_GRAMMER_ERROR_TITLE = 'orig grammar error'
ADV_GRAMMER_ERROR_TITLE = 'new grammar error'

GRAMMER_RESULT_FILENAME = 'grammar_result.txt'


def _grammar_check_one_text(_text):
    match = tool.check(_text)
    return len(match)

def grammar_check_df(df):

    if 'orig text' in df.columns:
        orig_text_title = 'orig text'
        adv_text_title = 'new text'
    elif 'orig hypothesis' in df.columns:
        orig_text_title = 'orig hypothesis'
        adv_text_title = 'new hypothesis'
    else:
        raise Exception('Wrong format of csv')

    orig_error_list = []
    adv_error_list = []

    orig_error_total_num = 0
    adv_error_total_num = 0

    for idx, row in tqdm(df.iterrows()):
        orig_text = row[orig_text_title]
        adv_text = row[adv_text_title]

        if row['change rate'] > 25:
            orig_error_num = -1
            adv_error_num = -1
        else:
            orig_error_num = _grammar_check_one_text(orig_text)
            adv_error_num = _grammar_check_one_text(adv_text)
            orig_error_total_num += orig_error_num
            adv_error_total_num += adv_error_num

        orig_error_list.append(orig_error_num)
        adv_error_list.append(adv_error_num)

    df[ORIG_GRAMMER_ERROR_TITLE] = orig_error_list
    df[ADV_GRAMMER_ERROR_TITLE] = adv_error_list


    return df, orig_error_total_num, adv_error_total_num


def grammar_check(result_folder_dir):
    print('Start grammar checking on', result_folder_dir)

    csv_path = os.path.join(result_folder_dir, 'results_final.csv')
    df = pd.read_csv(csv_path)
    df, orig_error_total_num, adv_error_total_num = grammar_check_df(df)
    I = adv_error_total_num / orig_error_total_num - 1

    # write to csv
    df.to_csv(csv_path, index=False)

    # save summary to txt
    txt_path = os.path.join(result_folder_dir, GRAMMER_RESULT_FILENAME)
    with open(txt_path, 'w') as f:
        f.write('=== Grammer Check Result ===\n')
        f.write(f'Orig Grammer Error: {orig_error_total_num}\n')
        f.write(f'Adv Grammer Error: {adv_error_total_num}\n')
        f.write(f'Grammar Increase: {I:.2%}\n')
    print(f'{result_folder_dir} Grammar Increase: {I:.2%}\n')


if __name__ == '__main__':
    grammar_check('out/results/decision_untarget/imdb/wordCNN/MO_syno50')





