# =================== IMDB MO =============================
python preprocess.py \
        --target_dataset imdb \
        --target_dataset_path data/imdb \
        --orig_dataset_path /home/workspace/big_data/datasets/imdb \
        --output_dir preprocess_data \
        --word_embeddings_path /home/workspace/big_data/hardlabel/glove.6B.200d.txt \
        --counter_fitting_embeddings_path /home/workspace/big_data/hardlabel/counter-fitted-vectors.txt \
        --counter_fitting_cos_sim_path /home/workspace/big_data/hardlabel/mat.txt


# Hardlabel
# syno50
CUDA_VISIBLE_DEVICES=0 python3 classification_attack.py \
        --attacker MO \
        --sub_method syno50 \
        --target_model bert \
        --target_model_path /home/workspace/big_data/hardlabel/models/bert/imdb \
        --target_dataset imdb \
        --dataset_path /home/workspace/hard_label_MO/data/imdb \
        --target_label_path '' \
        --attack_number 20 \
        --USE_cache_path /home/workspace/big_data/hardlabel/nli_cache/use \
        --counter_fitting_cos_sim_path /home/workspace/big_data/hardlabel/mat.txt \
        --preprocess_path /home/workspace/hard_label_MO/preprocess_data/syno50/imdb \
        --goal_function untarget \
        --setting decision \
        --word_embeddings_path /home/workspace/big_data/hardlabel/glove.6B.200d.txt \
        --counter_fitting_embeddings_path /home/workspace/big_data/hardlabel/counter-fitted-vectors.txt



CUDA_VISIBLE_DEVICES=0 python3 classification_attack.py \
        --attacker GA \
        --sub_method syno50 \
        --target_model bert \
        --target_model_path /home/workspace/big_data/hardlabel/models/bert/imdb \
        --target_dataset imdb \
        --dataset_path /home/workspace/hard_label_MO/data/imdb \
        --target_label_path '' \
        --attack_number 3 \
        --USE_cache_path /home/workspace/big_data/hardlabel/nli_cache/use \
        --counter_fitting_cos_sim_path /home/workspace/big_data/hardlabel/mat.txt \
        --preprocess_path /home/workspace/hard_label_MO/preprocess_data/syno50/imdb \
        --goal_function untarget \
        --setting decision \
        --word_embeddings_path /home/workspace/big_data/hardlabel/glove.6B.200d.txt \
        --counter_fitting_embeddings_path /home/workspace/big_data/hardlabel/counter-fitted-vectors.txt


CUDA_VISIBLE_DEVICES=0 python3 classification_attack.py \
        --attacker MO \
        --sub_method sememe \
        --target_model wordCNN \
        --target_model_path /home/workspace/big_data/hardlabel/models/wordcnn/imdb \
        --target_dataset imdb \
        --dataset_path /home/workspace/hard_label_MO/data/imdb \
        --target_label_path '' \
        --attack_number 3 \
        --USE_cache_path /home/workspace/big_data/hardlabel/nli_cache/use \
        --counter_fitting_cos_sim_path /home/workspace/big_data/hardlabel/mat.txt \
        --preprocess_path /home/workspace/hard_label_MO/preprocess_data/sememe/imdb \
        --goal_function untarget \
        --setting score \
        --word_embeddings_path /home/workspace/big_data/hardlabel/glove.6B.200d.txt \
        --counter_fitting_embeddings_path /home/workspace/big_data/hardlabel/counter-fitted-vectors.txt

CUDA_VISIBLE_DEVICES=0 python3 classification_attack.py \
        --attacker PSO \
        --sub_method sememe \
        --target_model wordCNN \
        --target_model_path /home/workspace/big_data/hardlabel/models/wordcnn/imdb \
        --target_dataset imdb \
        --dataset_path /home/workspace/hard_label_MO/data/imdb \
        --target_label_path '' \
        --attack_number 3 \
        --USE_cache_path /home/workspace/big_data/hardlabel/nli_cache/use \
        --counter_fitting_cos_sim_path /home/workspace/big_data/hardlabel/mat.txt \
        --preprocess_path /home/workspace/hard_label_MO/preprocess_data/sememe/imdb \
        --goal_function untarget \
        --setting score \
        --word_embeddings_path /home/workspace/big_data/hardlabel/glove.6B.200d.txt \
        --counter_fitting_embeddings_path /home/workspace/big_data/hardlabel/counter-fitted-vectors.txt


CUDA_VISIBLE_DEVICES=0 python3 nli_attack.py \
        --attacker MO \
        --sub_method syno50 \
        --target_model bert \
        --target_model_path /home/workspace/big_data/hardlabel/models/bert/snli \
        --target_dataset snli \
        --dataset_path /home/workspace/hard_label_MO/data/snli \
        --target_label_path '' \
        --attack_number 5 \
        --USE_cache_path /home/workspace/big_data/hardlabel/nli_cache/use \
        --counter_fitting_cos_sim_path /home/workspace/big_data/hardlabel/mat.txt \
        --preprocess_path /home/workspace/hard_label_MO/preprocess_data/syno50/snli \
        --goal_function untarget \
        --setting decision \
        --word_embeddings_path /home/workspace/big_data/hardlabel/glove.6B.200d.txt \
        --counter_fitting_embeddings_path /home/workspace/big_data/hardlabel/counter-fitted-vectors.txt


CUDA_VISIBLE_DEVICES=0 python3 nli_attack.py \
        --attacker PWWS \
        --sub_method wordnet_NE \
        --target_model bert \
        --target_model_path /home/workspace/big_data/hardlabel/models/bert/mnli \
        --target_dataset mnli_matched \
        --dataset_path /home/workspace/hard_label_MO/data/mnli_matched \
        --target_label_path '' \
        --attack_number 5 \
        --USE_cache_path /home/workspace/big_data/hardlabel/nli_cache/use \
        --counter_fitting_cos_sim_path /home/workspace/big_data/hardlabel/mat.txt \
        --preprocess_path /home/workspace/hard_label_MO/preprocess_data/wordnet_NE/mnli_matched \
        --goal_function untarget \
        --setting score \
        --word_embeddings_path /home/workspace/big_data/hardlabel/glove.6B.200d.txt \
        --counter_fitting_embeddings_path /home/workspace/big_data/hardlabel/counter-fitted-vectors.txt





