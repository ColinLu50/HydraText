classification_attack.py
python3 classification_attack.py \
        --dataset_path data/imdb  \
        --word_embeddings_path glove.6B.200d.txt \
        --target_model wordLSTM \
        --counter_fitting_cos_sim_path mat.txt \
        --target_dataset imdb \
        --target_model_path lstm/imdb \
        --counter_fitting_embeddings_path  counter-fitted-vectors.txt \
        --USE_cache_path " " \
        --max_seq_length 256 \
        --sim_score_window 40 \
        --nclasses 2 &&
python3 classification_attack.py \
        --dataset_path data/yelp  \
        --word_embeddings_path glove.6B.200d.txt \
        --target_model wordLSTM \
        --counter_fitting_cos_sim_path mat.txt \
        --target_dataset yelp \
        --target_model_path lstm/yelp \
        --counter_fitting_embeddings_path  counter-fitted-vectors.txt \
        --USE_cache_path " " \
        --max_seq_length 256 \
        --sim_score_window 40 \
        --nclasses 2 &&
python3 attack_classification_yahoo.py \
        --dataset_path data/yahoo  \
        --word_embeddings_path glove.6B.200d.txt \
        --target_model wordLSTM \
        --counter_fitting_cos_sim_path mat.txt \
        --target_dataset yahoo \
        --target_model_path lstm/yahoo \
        --counter_fitting_embeddings_path  counter-fitted-vectors.txt \
        --USE_cache_path " " \
        --max_seq_length 256 \
        --sim_score_window 40 \
        --nclasses 10 &&
python3 classification_attack.py \
        --dataset_path data/ag  \
        --word_embeddings_path glove.6B.200d.txt \
        --target_model wordLSTM \
        --counter_fitting_cos_sim_path mat.txt \
        --target_dataset ag \
        --target_model_path lstm/ag \
        --counter_fitting_embeddings_path  counter-fitted-vectors.txt \
        --USE_cache_path " " \
        --max_seq_length 256 \
        --sim_score_window 40 \
        --nclasses 4 &&
python3 classification_attack.py \
        --dataset_path data/mr  \
        --word_embeddings_path glove.6B.200d.txt \
        --target_model wordLSTM \
        --counter_fitting_cos_sim_path mat.txt \
        --target_dataset mr \
        --target_model_path lstm/mr \
        --counter_fitting_embeddings_path  counter-fitted-vectors.txt \
        --USE_cache_path " " \
        --max_seq_length 256 \
        --sim_score_window 40 \
        --nclasses 2

# new
python3 classification_attack.py \
        --dataset_path data/imdb  \
        --word_embeddings_path /home/workspace/big_data/hardlabel/glove.6B.200d.txt \
        --target_model wordLSTM \
        --counter_fitting_cos_sim_path /home/workspace/big_data/hardlabel/mat.txt \
        --target_dataset imdb \
        --target_model_path /home/workspace/big_data/hardlabel/lstm/imdb \
        --counter_fitting_embeddings_path  /home/workspace/big_data/hardlabel/counter-fitted-vectors.txt \
        --USE_cache_path "/home/workspace/big_data/hardlabel/nli_cache/use" \
        --max_seq_length 256 \
        --sim_score_window 40 \
        --nclasses 2
