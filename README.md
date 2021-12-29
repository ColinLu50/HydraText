# HydraText: Rethinking the Objectives in Adversarial Textual Attack

## Requirements

- Python == 3.7
- Pytorch == 1.8
- Tensorflow == 2.1
- NLTK == 3.5
- Spacy == 2.3.5
- OpenHowNet == 0.0.1a

## Download Dependencies

- Download top 50 synonym file of counter-fitted-vectors from [here](https://drive.google.com/file/d/1AIz8Imvv8OmHxVwY5kx10iwKAUzD6ODx/view) for Decision-Based attack.
- Download the glove 200 dimensional vectors from [here](https://nlp.stanford.edu/projects/glove/) unzip it.
- Download [counter-fitted-vectors](https://raw.githubusercontent.com/nmrksic/counter-fitting/master/word_vectors/counter-fitted-vectors.txt.zip) unzip it.
- Download USE model from [tensorflow model hub](https://tfhub.dev/google/universal-sentence-encoder-large/3).
- Download the Google language model according to `./local_models/download_google_lm.sh`, change the model path in `local_models/google_lm.py`


## Preprocess data

run `preprocess.py` to generate substitute words of each examples in each dataset, the generated files is placed in: 
- `output_folder/sememe`:  for **PSO, HydraText(score-based)** attack
- `output_folder/syno50`:  for **HydraText(decisoin-based), TextFooler, GADe** attack
- `output_folder/embedding_LM`: for **GA** attack
- `output_folder/wordnet_NE`: for **PWWS** attack

####  Run Preprocess Instructions: 

```
python preprocess.py \
        --target_dataset datast_name \
        --target_dataset_path path/to/attack_dataset \
        --orig_dataset_path /path/to/orig_data_folder \
        --output_dir output_folder \
        --word_embeddings_path /path/to/glove.6B.200d.txt \
        --counter_fitting_embeddings_path /path/to/counter-fitted-vectors.txt \
        --counter_fitting_cos_sim_path /path/to/mat.txt
```


## Victim Models

#### Direct Download
Download pretrained target models provided by [hard-label-attack](https://github.com/RishabhMaheshwary/hard-label-attack) for each dataset [bert](https://drive.google.com/file/d/1UChkyjrSJAVBpb3DcPwDhZUE4FuL0J25/view?usp=sharing), [lstm](https://drive.google.com/drive/folders/1nnf3wrYBrSt6F3Ms10wsDTTGFodrRFEW?usp=sharing), [cnn](https://drive.google.com/drive/folders/149Y5R6GIGDpBIaJhgG8rRaOslM21aA0Q?usp=sharing) unzip it.

#### Train

- BERT: use the commands provided in the `BERT` directory. 
- LSTM and CNN:  run the `train_classifier.py --<model_name> --<dataset>`.
- Infersent : run `infersent_model.py` script
- ESIM :  run `ESIM_train.py` script

## Attack

#### Command

run `classification_attack.py` to attack on **IMDB, MR, AG's News** datasets.
run `nli_attack.py` to attack on **SNLI, MNLI** datasets.

Here we explain each required argument in details:

- `--attacker`: Attack algorithm, including `[MO(HydraText), GA, PWWS, TF(TextFooler), PSO]`
- `--sub_method`: Substitute methods, including `[syno50, sememe, wordnet_NE, embedding_LM]`
- `--goal_function`: Goal function type: `[target, untarget]`
- `--setting`: Attack environment setting: `[score, decision]`
- `--target_model`: Name of target mode type: including `[bert, wordLSTM, wordCNN]`
- `--target_model_path`: The path to the trained parameters of the target model. 
- `--target_dataset`:  Name of target dataset.
- `--dataset_path`: The path to target dataset file.
- `--counter_fitting_cos_sim_path`: The path to pre-computed cosine similarity scores based on the counter-fitting word embeddings.
- `--preprocess_path`: Path to the output folder of `preprocess.py`.
- `--counter_fitting_embeddings_path`: The path to the counter-fitting word embeddings.
- `--word_embeddings_path`: The path to the  glove word embeddings.
- `--USE_cache_path`: The path to the USE model downloaded from tensorflow hub.
- `--target_label_path`: The path to the target label file, if the goal function is 'target'
- `--attack_number`: The number of attacking.
- `--nclasses`: Number of model output. For example, 4 for AG's News.

#### Examples

- Run **HydraText** attack on **BERT** model with **IMDB** dataset, using **untargeted** goal function in **decision-based** environment.

```
python3 classification_attack.py \
        --attacker MO \
        --sub_method syno50 \
        --goal_function untarget \
        --setting decision \
        --target_model bert \
        --target_model_path /path/to/imdb_bert \
        --target_dataset imdb \
        --dataset_path /path/to/imdb_dataset \
        --target_label_path '' \
        --attack_number 100 \
        --USE_cache_path /path/to/use_model \
        --counter_fitting_cos_sim_path /path/to/mat.txt \
        --preprocess_path /path/to/preprocess_data/syno50/imdb \
        --word_embeddings_path /path/to/glove.6B.200d.txt \
        --counter_fitting_embeddings_path /path/to/counter-fitted-vectors.txt \
        --nclasses 2
```



- Run **HydraText** attack on **ESIM** model with **SNLI** dataset, using **targeted** goal function in **score-based** environment.
```
python3 classification_attack.py \
        --attacker MO \
        --sub_method sememe \
        --goal_function target \
        --setting score \
        --target_model esim \
        --target_model_path /path/to/snli_esim \
        --target_dataset snli \
        --dataset_path /path/to/snli_dataset \
        --target_label_path /path/to/snli_target_label \
        --attack_number 100 \
        --USE_cache_path /path/to/use_model \
        --preprocess_path /path/to/preprocess_data/sememe/snli \
        --word_embeddings_path /path/to/glove.6B.200d.txt \
        --counter_fitting_embeddings_path /path/to/counter-fitted-vectors.txt \
        --nclasses 3
```



#### Check Results

The result is placed in the `/out/results/(goal_function)_(setting)/dataset/model/attack` folder, which includes:

- `log.txt`: Summary of attack results
- `results_final.csv`: Detailed adversarial results of each examples.



## Quality Check

#### Grammar Error

- Install [`language_tool_python`](https://github.com/jxmorris12/language_tool_python) package
- Run `grammar_check_LT.py` 
- Check result in the `result_folder/grammar_result.txt`

#### Perplexity 

- Download `gpt2-large`  model
- Run `PPL_score_GPT2.py` 
- Check result in the `result_folder/PPL_result.txt`

