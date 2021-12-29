python3 nli_attack.py \
        --dataset_path data/snli  \
        --word_embeddings_path glove.6B.200d.txt \
        --target_model bert \
        --counter_fitting_cos_sim_path mat.txt \
        --target_dataset snli \
        --target_model_path BERT/results/snli \
        --USE_cache_path "nli_cache" \
        --sim_score_window 40 \ &&
python3 nli_attack.py \
        --dataset_path data/mnli_matched  \
        --word_embeddings_path glove.6B.200d.txt \
        --target_model bert \
        --counter_fitting_cos_sim_path mat.txt \
        --target_dataset mnli \
        --target_model_path BERT/results/mnli \
        --USE_cache_path "nli_cache" \
        --sim_score_window 40 \  &&
python3 nli_attack.py \
        --dataset_path data/mnli_mismatched  \
        --word_embeddings_path glove.6B.200d.txt \
        --target_model bert \
        --counter_fitting_cos_sim_path mat.txt \
        --target_dataset mis_mnli \
        --target_model_path BERT/results/mnli \
        --USE_cache_path "nli_cache" \
        --sim_score_window 40 \

# new
python3 nli_attack.py \
        --dataset_path data/snli  \
        --word_embeddings_path /home/workspace/big_data/hardlabel/glove.6B.200d.txt \
        --counter_fitting_embeddings_path /home/workspace/big_data/hardlabel/counter-fitted-vectors.txt\
        --target_model bert \
        --counter_fitting_cos_sim_path /home/workspace/big_data/hardlabel/mat.txt \
        --target_dataset snli \
        --target_model_path /home/workspace/big_data/hardlabel/bert/snli \
        --USE_cache_path "/home/workspace/big_data/hardlabel/nli_cache/use" \
        --sim_score_window 40
