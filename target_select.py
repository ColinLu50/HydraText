import random
random.seed(1234)

def target_label_select():
    filepath = 'path/to/target_dataset_nli'
    target_label_list = []


    with open(filepath, 'r', encoding='utf8') as input_data:

        for idx, line in enumerate(input_data):

            line = line.strip().split('\t')

            # Ignore sentences that have no gold label.
            if line[0] == '-':
                continue

            label_str = line[0]

            if label_str == 'neutral':
                if random.random() < 0.5:
                    target_label = "contradiction"
                else:
                    target_label = "entailment"
            elif label_str == "entailment":
                target_label = "contradiction"
            elif label_str == "contradiction":
                target_label = "entailment"
            else:
                raise Exception('Wrong')

            print(f'{label_str} -> {target_label}')

            target_label_list.append(target_label)


    output_filepath = filepath + '_target'
    with open(output_filepath, 'w') as f:
        _str = "\n".join(target_label_list)
        f.write(_str)

if __name__ == '__main__':
    target_label_select()