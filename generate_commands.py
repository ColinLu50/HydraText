
MODEL_NAME = {
    'bert': 'BERT',
    'wordLSTM': 'LSTM',
    'wordCNN': 'CNN',
    'infersent': 'INFST',
    'esim': 'ESIM'
}

CLASSIFY_MODELS = {'imdb', 'agnews', 'mr', 'yahoo', 'imdb_train', 'mr_train'}

def generate_command(gpu_id, search_method, substitute, victim_model, dataset, setting, goal_func, attack_num, record_out):

    # assert dataset in {'imdb', 'snli', }

    classification = dataset in CLASSIFY_MODELS

    script_name = 'classification_attack.py' if classification else 'nli_attack.py'

    cmd_str = f'CUDA_VISIBLE_DEVICES={gpu_id} python3 {script_name} '\
        f'--attacker {search_method} ' \
        f'--sub_method {substitute} ' \
        f'--target_model {victim_model} ' \
        f'--target_dataset {dataset} ' \
          f'--setting {setting} ' \
          f'--goal_function {goal_func} '

    if attack_num:
        cmd_str += f'--attack_number {attack_num} '

    if record_out:
        cmd_str += f'> ./out/outfiles/{dataset.upper()}_{MODEL_NAME[victim_model]}_{search_method.upper()}_{substitute}_{setting}_{goal_func}.out 2>&1 &'

    return cmd_str

def adv_cmds():
    l = [('MO', 'syno50'), ('GA', 'syno50'), ('MO', 'sememe'), ('PSO', 'sememe')]
    dataset = 'mr'

    for attacker, sub in l:
        if sub == 'sememe':
            setting = 'score'
        else:
            setting = 'decision'

        for adv_attacker, adv_sub in l:
            cmd_str = f'CUDA_VISIBLE_DEVICES=3 ' \
                      f'python3 classification_attack.py ' \
                      f'--attacker {attacker} ' \
                      f'--sub_method {sub} ' \
                      f'--target_model bert ' \
                      f'--target_model_path adv_training/data/{dataset}_{adv_attacker}_{adv_sub} ' \
                      f'--output_dir out/adv_results/advT_{adv_attacker}_{adv_sub} ' \
                      f'--target_dataset {dataset} ' \
                      f'--setting {setting} ' \
                      f'--goal_function untarget ' \
                      f'> ./out/outfiles/{dataset.upper()}_{attacker}_{sub}_on_advT_{adv_attacker}_{adv_sub}.out 2>&1 &\n'
            print(cmd_str)

if __name__ == '__main__':
    gpu_id = 0
    search_method = 'GA'
    substitute = 'syno50'
    victim_model = 'wordCNN'
    dataset = 'snli'
    goal_func = 'target'
    setting = 'decision'
    attack_num = None
    # attack_num = 100
    # record_out = False
    record_out = True

    if dataset in CLASSIFY_MODELS:
        victim_models = ['bert', 'wordCNN', 'wordLSTM']
    elif dataset in {'snli', 'mnli', 'mnli_matched', 'mnli_mismatched'}:
        victim_models = ['bert', 'infersent', 'esim']

    for victim_model in victim_models:
        cmd_str = generate_command(gpu_id, search_method, substitute, victim_model, dataset, setting, goal_func, attack_num, record_out)

        print(cmd_str)
        print()

    # adv_cmds()