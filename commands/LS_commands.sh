# =================== IMDB =============================
# BERT
## Syno50
CUDA_VISIBLE_DEVICES=2 python3 classification_attack.py \
        --attacker LS\
        --sub_method syno50\
        --target_model wordCNN \
        --target_dataset imdb \
        --goal_function untarget\
         > ./out/outfiles/IMDB_CNN_LS_Syno50.out 2>&1 & &&

## Sememe
CUDA_VISIBLE_DEVICES=2 python3 classification_attack.py \
        --attacker LS\
        --sub_method sememe\
        --target_model bert \
        --target_dataset imdb\
        --goal_function untarget\
        > ./out/outfiles/IMDB_BERT_LS_Sememe.out 2>&1 &

# WordCNN
## Sememe
CUDA_VISIBLE_DEVICES=3 python3 classification_attack.py \
        --attacker LS\
        --sub_method sememe\
        --target_model wordCNN \
        --target_dataset imdb\
        --goal_function untarget\
        > ./out/outfiles/IMDB_CNN_LS_Sememe.out 2>&1 &

# LSTM
## Semem
CUDA_VISIBLE_DEVICES=0 python3 classification_attack.py \
        --attacker LS\
        --sub_method sememe\
        --target_model wordLSTM \
        --target_dataset imdb\
        --goal_function untarget\
        > ./out/outfiles/IMDB_LSTM_LS_Sememe.out 2>&1 &

# ================= SNLI ===============================
CUDA_VISIBLE_DEVICES=3 python3 nli_attack.py \
        --attacker LS \
        --sub_method syno50 \
        --target_model bert \
        --target_dataset snli \
        --goal_function untarget \
        > ./out/outfiles/SNLI_BERT_LS_Syno50.out 2>&1 & &&
        --data_size 20