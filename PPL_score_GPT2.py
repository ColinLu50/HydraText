import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm
import pandas as pd

PPL_TITLE = 'Perplexity'
PPL_RESULT_FILENAME = 'PPL_result.txt'

device = torch.device('cuda')

model_name = 'gpt2-large'
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

def calc_PPL_text(text):
    encoded_input = tokenizer(text, return_tensors='pt').to(device)
    input_ids = encoded_input.input_ids
    output = model(input_ids, labels=input_ids)
    loss = output[0]
    PPL = torch.exp(loss).detach().cpu().numpy()
    # print(PPL)

    return PPL

def calc_PPL_df(df):

    if 'orig text' in df.columns:
        adv_text_title = 'new text'
    elif 'orig hypothesis' in df.columns:
        adv_text_title = 'new hypothesis'
    else:
        raise Exception('Wrong format of csv')

    PPL_list = []
    real_PPL_list = []

    for idx, row in tqdm(df.iterrows()):
        adv_text = row[adv_text_title]

        if row['change rate'] > 25:
            adv_PPL = -1
        else:
            adv_PPL = calc_PPL_text(adv_text)
            real_PPL_list.append(adv_PPL)

        PPL_list.append(adv_PPL)


    df[PPL_TITLE] = PPL_list

    return df, np.mean(real_PPL_list)

def calc_PPL(result_folder_dir):
    print('Calculate Perplexity on', result_folder_dir)

    csv_path = os.path.join(result_folder_dir, 'results_final.csv')
    df = pd.read_csv(csv_path)
    df, avg_PPL = calc_PPL_df(df)

    # write to csv
    df.to_csv(csv_path, index=False)

    # save summary to txt
    txt_path = os.path.join(result_folder_dir, PPL_RESULT_FILENAME)
    with open(txt_path, 'w') as f:
        f.write('=== Perplexity Result ===\n')
        f.write(f'Calculated by {model_name}\n')
        f.write(f'PPL : {avg_PPL}\n')

    print(f'{result_folder_dir} PPL: {avg_PPL}\n')


# if __name__ == '__main__':
#     calc_PPL('out/results/untarget_decision/imdb/bert/GA_syno50')


