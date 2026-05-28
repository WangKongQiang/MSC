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

