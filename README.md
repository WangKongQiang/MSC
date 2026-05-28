# MSC
Project code for paper ---> MSC: Matrix Supporting Clustering Based on Sentence-Transformers with Contrastive Learning.

Function description of files and folders:

datasets   folder:   

                    textdatasets--------------------agnews.csv  biomedical.csv  googlenews_S.csv  googlenews_T.csv  googlenews_TS.csv  searchsnippets.csv  stackoverflow.csv  tweet.csv   It contains eight representative short text public datasets.

                    Neural-Machine-Translation-German------------------Corresponding data augmentation method: paraphrase via back-translation (PARA-German).
                    
                    Neural-Machine-Translation-Nepali------------------Corresponding data augmentation method: paraphrase via back-translation (PARA-Nepali).
                    
                    Paraphrase_French---------------------Corresponding data augmentation method: paraphrase via back-translation (PARA-French).
                    
                    nlpaug_explore.py-----------------Corresponding data augmentation methods: the WordNet augmenter (WNET) and the Context augmenter (CTXT).
                    
                    rtr_augment.py------------------Corresponding data augmentation method: Random Token Replacement (RTR).
                    
                    rtr_typo_augment.py---------------Corresponding data augmentation method: Character-level Perturbation (TYPO).
                    
                    swr_augment.py------------------Corresponding data augmentation method: Stop-Words Replacement (SWR).

Kmeans.ipynb-------------------It is used for the initial exploration of Standard KMeans (initialized randomly), K-Means++ (default) and K-Means-- Simple handwritten Implementation (Core Logic) in a randomly generated matrix of numbers, to facilitate the transfer to our experiments.

requirements.txt------------------This is the main environment configuration for the entire project. A virtual environment used for conveniently and quickly reproducing MSC: Matrix Supporting Clustering Based on Sentence-Transformers with Contrastive Learning method.

main.py----------------------This is the entry code for implementing MSC: Matrix Supporting Clustering Based on Sentence-Transformers with Contrastive Learning method on eight representative short text datasets.

#⚡Quick Start⚡#

The pre-trained models to be pulled, models--FacebookAI--roberta-base and models--google-bert--bert-base-uncased, are used for contextual_augment function in the nlpaug_explore.py script.

Hugging Face: https://huggingface.co/FacebookAI/roberta-base

Hugging Face: https://huggingface.co/google-bert/bert-base-uncased

The pre-trained models to be pulled, models--sentence-transformers--all-mpnet-base-v2, models--sentence-transformers--all-distilroberta-v1, models--sentence-transformers--distilbert-base-nli-stsb-mean-tokens serves as the backbone of the framework.

sentence-transformers/all-mpnet-base-v2 Hugging Face: https://huggingface.co/sentence-transformers/all-mpnet-base-v2

sentence-transformers/all-distilroberta-v1 Hugging Face: https://huggingface.co/sentence-transformers/all-distilroberta-v1

sentence-transformers/distilbert-base-nli-stsb-mean-tokens Hugging Face: https://huggingface.co/sentence-transformers/distilbert-base-nli-stsb-mean-tokens

The torch installation environment and the execution parameters for data augmentation can be found in the comments in nlpaug_explore.py

### setup torch
 pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

### Dataset Content (It contains datasets of the corresponding data augmentation type).
Perform corresponding data augmentation on the eight representative short text public datasets of folder datasets/textdatasets. Generate corresponding enhanced samples text1 and text2 based on the original text.

The script for running MSC contrastive learning can be found in main.py, and the hyperparameters for running MSC contrastive learning can be found in the script comments.

### Run the MSC clustering experiment script: (For example, take the experiment with RTR data augmentation as a positive example generation method).
 python main.py  --objective MSC --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname searchsnippets_RTR --num_classes 8
 
 python main.py  --objective MSC --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname stackoverflow_RTR --num_classes 20
 
 python main.py  --objective MSC --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname biomedical_RTR--num_classes 20
 
 python main.py  --objective MSC --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname agnews_RTR --num_classes 4
 
 python main.py  --objective MSC --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname tweet_RTR --num_classes 110
 
 python main.py  --objective MSC --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname googlenews_TS_RTR --num_classes 152
 
 python main.py  --objective MSC --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname googlenews_T_RTR --num_classes 152
 
 python main.py  --objective MSC --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname googlenews_S_RTR --num_classes 152
