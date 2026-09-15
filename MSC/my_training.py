"""
Copyright https://www.ynu.edu.cn/ or its affiliates school. All Rights Reserved

Author: Kongqiang Wang (wangkongqiang60@gmail.com)
Date: 05/27/2026
"""

import math
import os
import time
import numpy as np
from sklearn import cluster
from pandas import DataFrame
from utils.logger import statistics_log
from utils.metric import Confusion
from dataloader.dataloader import unshuffle_loader
import torch.utils.data as util_data
from tqdm import trange, tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from learner.cluster_utils import target_distribution
from learner.contrastive_utils import PairConLoss
from contrastive_loss import ClusterLoss
# from contrastive_loss import InstanceLoss
from contrastive_loss import SupConLoss
# from figure import draw,fig
from memory import MemoryBank,fill_memory_bank
from neighbor_dataset import NeighborsDataset

class SCCLvTrainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer,optimizer1, train_loader, args):
        super(SCCLvTrainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.optimizer1=optimizer1
        self.train_loader = train_loader
        self.args = args
        self.eta = self.args.eta
        self.cluster_num = args.num_classes
        self.cluster_loss = nn.KLDivLoss(size_average=False)  # the original clustering loss
        self.contrast_loss1 = PairConLoss(temperature=self.args.temperature)  # the original instance-level loss
        self.cluster_loss2 = ClusterLoss(class_num=self.cluster_num, temperature=self.args.temperature,
                                         device=torch.device("cuda"))  # Newly created cluster contrast loss
        self.gstep = 0
        # self.contrast_loss = SupConLoss()  # The original instance-level loss
        self.contrast_loss = SupConLoss(temperature=self.args.temperature)
        print(f"*****Intialize SCCLv, temp:{self.args.temperature}, eta:{self.args.eta}\n")

    def get_batch_token(self, text):
        token_feat = self.tokenizer.batch_encode_plus(
            text,
            max_length=self.args.max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        return token_feat

    def prepare_transformer_input(self, batch):
        if len(batch) == 4:
            text1, text2, text3 = batch['text'], batch['augmentation_1'], batch['augmentation_2']
            feat1 = self.get_batch_token(text1)
            feat2 = self.get_batch_token(text2)
            feat3 = self.get_batch_token(text3)
            label = batch['label']
            input_ids = torch.cat(
                [feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1), feat3['input_ids'].unsqueeze(1)],
                dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1),
                                        feat3['attention_mask'].unsqueeze(1)], dim=1)

        elif len(batch) == 2:
            text = batch['text']
            feat1 = self.get_batch_token(text)
            feat2 = self.get_batch_token(text)

            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)],
                                       dim=1)
        elif len(batch) == 3 :
            input_ids = batch['input_id']
            attention_mask = batch['attention_mask']
            label = batch['label']

        return input_ids.cuda(), attention_mask.cuda(), label.cuda()

    def train_step_virtual(self, input_ids, attention_mask):

        embd1, embd2 = self.model(input_ids, attention_mask, task_type="virtual")

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd1, embd2)
        losses = self.contrast_loss(feat1, feat2)
        loss = self.eta * losses["loss"]

        #cluster_level_contrastive
        # Clustering loss
        if self.args.objective == "SCCL":
            output = self.model.get_cluster_prob(embd1)
            target = target_distribution(output).detach()

            cluster_loss = self.cluster_loss((output + 1e-08).log(), target) / output.shape[0]
            loss += 0.5 * cluster_loss
            losses["cluster_loss"] = cluster_loss.item()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses

    def train_step_explicit(self, input_ids, attention_mask, adjacency, i, L):
        a = 0.5 * math.cos(i / L * math.pi) + 0.5
        losses = {}
        embd1, embd2, embd3 = self.model(input_ids, attention_mask, task_type="explicit")
       
        feat1, feat2 = self.model.contrast_logits(embd2, embd3)
        
        feat11, feat12 = self.model.cluster_logits(embd2, embd3)
        cluloss = self.cluster_loss2(feat11, feat12)
        features = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
        
        losse = self.contrast_loss(features, labels=None, mask=adjacency)  # The original component part
        loss = a * (losse + cluloss)

        losses["loss"] = loss
       
        # Clustering loss
        if self.args.objective == "SCCL":
            output = self.model.get_cluster_prob(embd1)
            target = target_distribution(output).detach()

            cluster_loss = self.cluster_loss((output + 1e-08).log(), target) / output.shape[0]

            losses["cluster_loss"] = cluster_loss.item()

            loss += (1-a) * (cluster_loss)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses
        # return losses
        # return cluster_loss

    def get_neighbor_dataset(self, args, data, indices):
        """convert indices to dataset"""
        dataset = NeighborsDataset(data.dataset.train_semi_dataset, indices) #Pass in the original data and their indice
        self.train_dataloader = util_data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True) #data loading

        print("complete!")

    def get_neighbor_inds(self, args, data):
        """get indices of neighbors"""
        memory_bank = MemoryBank(len(data.dataset.train_semi_dataset), 768, len(data.dataset.all_label_list), 0.1)  #Store all the data
        fill_memory_bank(data.dataset.train_semi_dataloader, self.model, memory_bank)
        indices = memory_bank.mine_nearest_neighbors(args.topk, calculate_accuracy=False)

        return indices

    def get_adjacency(self, inds, neighbors, targets):
        """get adjacency matrix"""
        adj = torch.zeros(inds.shape[0], inds.shape[0])
        for b1, n in enumerate(neighbors):
            adj[b1][b1] = 1
            for b2, j in enumerate(inds):
                if j in n:
                    adj[b1][b2] = 1  # if in neighbors
                if (targets[b1] == targets[b2]) and (targets[b1] > 0) and (targets[b2] > 0):
                    adj[b1][b2] = 1  # if same labels
                    # this is useful only when both have labels
        return adj

    def train(self):
        print('\n={}/{}=Iterations/Batches'.format(self.args.max_iter, len(self.train_loader)))
        indices = self.get_neighbor_inds(self.args, self.train_loader)  # [Text length * Number of neighbors] The i-th line represents the text itself plus 50 neighbors
        self.get_neighbor_dataset(self.args, self.train_loader, indices)
        self.model.train()
        L = self.args.max_iter
        for i in np.arange(self.args.max_iter + 1):
        # for batch in tqdm(self.train_dataloader, desc="Iteration"):
            try:
                batch = next(train_loader_iter)
                # batch = next(self.train_dataloader)
            except:
                train_loader_iter = iter(self.train_dataloader)
                batch = next(train_loader_iter)

            #  New component section
            # 1. load data
            anchor = tuple(t.to('cuda') for t in batch["anchor"])  # anchor data
            neighbor = tuple(t.to('cuda') for t in batch["neighbor"])  # neighbor data
            pos_neighbors = batch["possible_neighbors"]  # all possible neighbor inds for anchor
            data_inds = batch["index"]  # neighbor data ind

            # 2. get adjacency matrix
            adjacency = self.get_adjacency(data_inds, pos_neighbors, batch["target"])  # (bz,bz)

            input_ids = anchor[0]
            attention_mask = anchor[1]
         
            #  The original component part
            # input_ids, attention_mask ,labels= self.prepare_transformer_input(batch)
            losses = self.train_step_virtual(input_ids, attention_mask) if self.args.augtype == "virtual" else self.train_step_explicit(input_ids, attention_mask, adjacency, i, L)

            # if (i == self.args.max_iter):
            if (self.args.print_freq > 0) and ((i % self.args.print_freq == 0) or (i == self.args.max_iter)):
                statistics_log(self.args.tensorboard, losses=losses, global_step=i)

                self.evaluate_embedding(i)
                # indices = self.get_neighbor_inds(self.args, self.train_loader)
                # self.get_neighbor_dataset(self, self.args, self.train_loader, indices)
                self.model.train()

        return None

    def evaluate_embedding(self, step):
       
        dataloader = unshuffle_loader(self.args)
        print('---- {} evaluation batches ----'.format(len(dataloader)))
     
        self.model.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                text, label = batch['text'], batch['label']
                feat = self.get_batch_token(text)
                embeddings = self.model(feat['input_ids'].cuda(), feat['attention_mask'].cuda(), task_type="evaluate")
                # res.extend(embeddings.to("cpu").numpy().tolist())
                # lab.extend(label.to("cpu").numpy().tolist())
                model_prob = self.model.get_cluster_prob(embeddings)
                if i == 0:
                    all_labels = label
                    all_embeddings = embeddings.detach()
                    all_prob = model_prob
                else:
                    all_labels = torch.cat((all_labels, label), dim=0)
                    all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                    all_prob = torch.cat((all_prob, model_prob), dim=0)
      
        # self.get_neighbor_dataset
        confusion, confusion_model = Confusion(self.args.num_classes), Confusion(self.args.num_classes)

        all_pred = all_prob.max(1)[1]
        confusion_model.add(all_pred, all_labels)
        confusion_model.optimal_assignment(self.args.num_classes)
        acc_model = confusion_model.acc()

        kmeans = cluster.KMeans(n_clusters=self.args.num_classes, random_state=self.args.seed)
        embeddings = all_embeddings.cpu().numpy()
        kmeans.fit(embeddings)
        pred_labels = torch.tensor(kmeans.labels_.astype(np.int))

        # clustering accuracy
        confusion.add(pred_labels, all_labels)
        confusion.optimal_assignment(self.args.num_classes)
        acc = confusion.acc()

        ressave = {"acc": acc, "acc_model": acc_model}
        ressave.update(confusion.clusterscores())
        for key, val in ressave.items():
            self.args.tensorboard.add_scalar('Test/{}'.format(key), val, step)

        np.save(self.args.resPath + 'acc_{}.npy'.format(step), ressave)
        np.save(self.args.resPath + 'scores_{}.npy'.format(step), confusion.clusterscores())
        np.save(self.args.resPath + 'mscores_{}.npy'.format(step), confusion_model.clusterscores())
        # np.save(self.args.resPath + 'mpredlabels_{}.npy'.format(step), all_pred.cpu().numpy())
        # np.save(self.args.resPath + 'predlabels_{}.npy'.format(step), pred_labels.cpu().numpy())
        # np.save(self.args.resPath + 'embeddings_{}.npy'.format(step), embeddings)
        # np.save(self.args.resPath + 'labels_{}.npy'.format(step), all_labels.cpu())

        print('[Representation] Clustering scores:', confusion.clusterscores())
        print('[Representation] ACC: {:.3f}'.format(acc))
        print('[Model] Clustering scores:', confusion_model.clusterscores())
        print('[Model] ACC: {:.3f}'.format(acc_model))
        return None