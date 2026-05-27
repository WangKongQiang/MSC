"""
Copyright https://www.ynu.edu.cn/ or its affiliated school. All Rights Reserved

Author: Kongqiang Wang (wangkongqiang60@gmail.com)
Date: 05/27/2026
"""

import torch
import numpy as np
from utils.metric import Confusion
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
import sklearn.cluster as sc
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
from sklearn_extra.cluster import KMedoids

def get_mean_embeddings(bert, input_ids, attention_mask):
    bert_output = bert.forward(input_ids=input_ids, attention_mask=attention_mask)
    attention_mask = attention_mask.unsqueeze(-1)
    mean_output = torch.sum(bert_output[0] * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
    mean_output=  bert_output[0][:,0,:]
    return mean_output

def get_batch_token(tokenizer, text, max_length):
    token_feat = tokenizer.batch_encode_plus(
        text,
        max_length=max_length,
        return_tensors='pt',
        padding='max_length',
        truncation=True
    )
    return token_feat

def get_kmeans_centers(bert, tokenizer, train_loader, num_classes, max_length):
    for i, batch in enumerate(train_loader):
        print("get kmeans centers : {} ".format(i))
        label = batch['label']

        # The original dataloader section

        # text, label = batch['text'], batch['label']
        # tokenized_features = get_batch_token(tokenizer, text, max_length)
        # corpus_embeddings0 = get_mean_embeddings(bert, **tokenized_features)

        # The new dataloader section

        corpus_embeddings0 = get_mean_embeddings(bert, batch['input_id'][:, 0, :], batch['attention_mask'][:, 0, :])

        if i == 0:
            all_labels = label
            all_embeddings = corpus_embeddings0.detach().numpy()
        else:
            all_labels = torch.cat((all_labels, label), dim=0)
            all_embeddings = np.concatenate((all_embeddings, corpus_embeddings0.detach().numpy()), axis=0)

    print("get kmeans centers : OK!")

    # Perform KMeans clustering
    confusion = Confusion(num_classes)

   #  ️⃣ 1 If you mean K-means + +（recommend）, In fact, you are already using it. In K-means, the default initialization method of sklearn is 👉 K-means + +（k - means + + initialization）
    # That is to say, this line of code of yours：
    # clustering_model = KMeans(n_clusters=num_classes)
    # it is equivalent to：
    # clustering_model = KMeans(n_clusters=num_classes, init='k-means++')
    # ✔ It is already one of the optimal initialization methods without any modification.
    #  ️⃣ 2 If you want to use MiniBatch K - means（big data is faster） It can be changed to:

    # ⭐ Optional but highly recommended: Do normalization first (very crucial)
    # all_embeddings = normalize(all_embeddings)
    # ✅ Conclusion (Directly giving you the most practical suggestions)
    # If you just want to "upgrade KMeans", it is most recommended to change it to👇：

    # clustering_model = MiniBatchKMeans(
    #     n_clusters=num_classes,
    #     batch_size=1024,
    #     random_state=42
    # )
    # it is apply to： The volume of data is very large.（You are currently embeddings and may grow larger and larger）
    #         GPU / Memory high pressure

    # ️⃣ 3 If you merely aim to "optimize the effect", what's actually more crucial is👇  Using the cosine distance
    # clustering_model = KMeans(n_clusters=num_classes, algorithm='elkan')  # The default European style is actually not suitable for BERT
    # 👉 BERT embedding is more suitable for cosine rather than Euclidean distance

    # The implementation and differences of K-means, K-means ++, and K-means in sklearn
    # sklearn KMeans、KMeans++、KMeans-- Implementation principle & Core differences
    # Let's clarify the concept first:

    # 1.  Common K-Means：Randomly initialize the clustering centers

    # 2.  K-Means++：The default initialization method of sklearn optimizes the initial center point to avoid local optimum.

    # 3.  K-Means--：It's a reverse approach. First, select all the points as the centers, and then gradually delete the redundant centers. It's a niche improved version.

    # I、 Core Principles of the Three

    # 1. Standard K-Means (Random Initialization)
    # Steps of Algorithm
    # 1.  Randomly select K samples from the sample as the initial clustering centers
    # 2.  Allocation: Each sample is assigned to the nearest cluster center
    # 3.  Update: The mean of each cluster is recalculated as the new center
    # 4.  Repeat iterations 2 and 3 until the center no longer changes/the maximum iteration is reached
    # drawback
    # The initial centers are completely random, which makes it easy to fall into local optimum. The clustering results fluctuate greatly and are prone to poor clustering.

    # 2.   K-Means++（sklearn default   init = 'k-means++'  ）
    # Only the "initial center selection" is changed. The subsequent iterations are exactly the same as those of a regular KMeans
    # Initialization rule
    # 1.  Randomly select the first clustering center from all samples
    # 2.  For each sample, calculate the minimum distance D(x) to the selected center
    # 3.  Select the next center by the probability of the square of the distance: P(x) \propto D(x) ^ 2 roulette
    # 4.  Repeat until K initial centers are selected
    # 5.  The subsequent iterations will still follow the standard KMeans
    # advantage
    # The initial centers are far apart from each other, making it less likely to fall into local optimum, and the clustering effect is more stable and converges faster.

    # 3. K-Means-- （K-Means minus minus）
    # The train of thought is completely opposite to the previous two
    # algorithm thought
    # 1.  At the beginning, all samples were regarded as clustering centers
    # 2.  Each time, delete the most redundant central point that is closest to the other centers
    # 3.  Keep deleting until only K centers remain
    # 4.  Then use these K centers to perform the standard KMeans iteration
    # characteristic
    # - Suitable for small samples and high-dimensional sparse data
    # - The initial center has a good global distribution, but the computational load is greater than that of kmeans + +
    # - sklearn does not have a native implementation and needs to be written by hand

    # Ⅱ、sklearn code implementation
    #
    # 1. Standard KMeans (Randomly initialized)
    # 2. K-Means++（Default）
    # 3. K-Means-- Simple handwritten Implementation (Core Logic)
    #
    # Ⅲ、List of Core Differences
    # Comparison dimension standard K-Means,K-Means++,K-Means--
    # initial center: Randomly select K / Choose the K ones that are far apart in probability / Select all first and then reduce to K
    # local optimum: It's very easy to fall into / Greatly avoided / Not easy to fall into
    # rate of convergence: Slow and highly volatile / Grow fast and steadily / Slow-paced
    # sklearn support: Native support / Default native / There is no originality; it needs to be handwritten
    # computational overhead: little / medium / rather big
    # applicable scene: Just run around, baseline / Industrial first choice, universal / Small sample size, high-dimensional data
    # Ⅳ、A one-sentence summary
    #
    # 1.  For daily use, simply default KMeans++，the effect is the most stable and no adjustment is needed；
    #
    # 2.  Common KMeans It is only suitable for simple tasks baseline；
    #
    # 3.  KMeans-- It is a niche improvement. It is mostly used in academic experiments and rarely in engineering.
    #
    # Do I need to write a visual comparison code for the clustering effects of the three methods for you? Can you see the differences by running it directly? You can refer to the Kmeans.ipynb script.

    ### KMEANS- -
    # class KMeansMinusMinus:
    #     def __init__(
    #             self,
    #             n_clusters,
    #             delete_batch=100,
    #             random_state=42
    #     ):
    #         self.k = n_clusters
    #         self.delete_batch = delete_batch
    #         self.random_state = random_state
    #
    #         self.labels_ = None
    #         self.cluster_centers_ = None
    #         self.n_iter_ = 0
    #
    #     def fit_predict(self, X):
    #
    #         np.random.seed(self.random_state)
    #
    #         # Initially, all are taken as the center
    #         centers = X.copy()
    #
    #         print(f"Initial centers: {len(centers)}")
    #
    #         # Constantly delete
    #         while len(centers) > self.k:
    #
    #             current_n = len(centers)
    #
    #             # How much is deleted in each round
    #             remove_n = min(
    #                 self.delete_batch,
    #                 current_n - self.k
    #             )
    #
    #             print(f"Current centers: {current_n}")
    #
    #             # Calculate the nearest neighbor distance in blocks
    #             nearest_dist = np.full(current_n, np.inf)
    #
    #             block_size = 512
    #
    #             for i in range(0, current_n, block_size):
    #
    #                 end_i = min(i + block_size, current_n)
    #
    #                 dist_block = euclidean_distances(
    #                     centers[i:end_i],
    #                     centers
    #                 )
    #
    #                 # Set the inf by yourself
    #                 for j in range(end_i - i):
    #                     dist_block[j, i + j] = np.inf
    #
    #                 nearest_dist[i:end_i] = np.min(
    #                     dist_block,
    #                     axis=1
    #                 )
    #
    #             # Delete the most redundant centers
    #             del_indices = np.argsort(nearest_dist)[:remove_n]
    #
    #             mask = np.ones(current_n, dtype=bool)
    #             mask[del_indices] = False
    #
    #             centers = centers[mask]
    #
    #             self.n_iter_ += 1
    #
    #         print(f"Final init centers: {len(centers)}")
    #
    #         # Initialize KMeans with the selected centers
    #         km = KMeans(
    #             n_clusters=self.k,
    #             init=centers,
    #             n_init=1,
    #             random_state=self.random_state
    #         )
    #
    #         self.labels_ = km.fit_predict(X)
    #         self.cluster_centers_ = km.cluster_centers_
    #
    #         return self.labels_
    #
    # clustering_model = KMeansMinusMinus(n_clusters=num_classes, delete_batch=100)

    # KMedoids  Help me achieve it
    # Sure. I'll directly help you seamlessly replace the K-Medoids (PAM) version into your code while keeping your original evaluation logic unchanged.

    # ⚠️Let's start with the key points:
    # KMedoids is more robust (less sensitive to outliers) than KMeans.
    # But it is slower (O(n²)). Be careful when there are many samples

    # ✅ I. Directly Available Version (Replacing KMeans)
    #
    # You just need to install:
    # pip install scikit-learn-extra
    # Then take your original:
    # clustering_model = KMeans(n_clusters=num_classes, init='random', n_init=10)
    # change for👇

    clustering_model = KMedoids(
        n_clusters=num_classes,
        metric='cosine',  # ⭐ Strongly recommended cosine（BERT embedding）
        method='pam',  # standard KMedoids
        init='k-medoids++',
        random_state=42
    )

    # clustering_model = AgglomerativeClustering(n_clusters=num_classes, linkage="ward")
    # clustering_model = AffinityPropagation(damping=0.5, max_iter=5, convergence_iter=30, preference=-50)

    # clustering_model = GaussianMixture(n_components=num_classes)
    # clustering_model = Birch(n_clusters=num_classes,threshold=0.2)

    # bw = sc.estimate_bandwidth(all_embeddings, n_samples=len(all_embeddings), quantile=0.1)
    # clustering_model = sc.MeanShift(bandwidth=bw, bin_seeding=True)
    # model = AffinityPropagation(damping=0.5, max_iter=500, convergence_iter=30, preference=-50)

    # clustering_model.fit(all_embeddings)
    # cluster_centers_indices =  clustering_model.cluster_centers_indices_
    # cluster_assignment = clustering_model.labels_

    cluster_assignment = clustering_model.fit_predict(all_embeddings)
    true_labels = all_labels
    pred_labels = torch.tensor(cluster_assignment)
    print("all_embeddings:{}, true_labels:{}, pred_labels:{}".format(all_embeddings.shape, len(true_labels),
                                                                     len(pred_labels)))

    confusion.add(pred_labels, true_labels)
    confusion.optimal_assignment(num_classes)
    print("Iterations:{}, Clustering ACC:{:.3f}, centers:{}".format(clustering_model.n_iter_, confusion.acc(),
                                                                    clustering_model.cluster_centers_.shape))
    # print(" Clustering ACC:{:.3f}, centers:{} ".format( confusion.acc(), clustering_model.means_))
    return clustering_model.cluster_centers_