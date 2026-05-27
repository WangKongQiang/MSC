"""
Copyright https://www.ynu.edu.cn/ or its affiliated school. All Rights Reserved

Author: Kongqiang Wang (wangkongqiang60@gmail.com)
Date: 05/27/2026
"""

from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import TensorDataset
import os
import torch
import pandas as pd
import torch.utils.data as util_data
from torch.utils.data import Dataset
import re
import numpy as np

class VirtualAugSamples(Dataset):
    def __init__(self, train_x, train_y):
        assert len(train_x) == len(train_y)
        self.train_x = train_x
        self.train_y = train_y

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'label': self.train_y[idx]}

class ExplitAugSamples(Dataset):
    def __init__(self, train_x, train_x1, train_x2, train_y):
        assert len(train_y) == len(train_x) == len(train_x1) == len(train_x2)
        self.train_x = train_x
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        self.train_y = train_y

    def __len__(self):
        return len(self.train_y)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'augmentation_1': self.train_x1[idx], 'augmentation_2': self.train_x2[idx], 'label': self.train_y[idx]}

class ExplitAugSamples1(Dataset):#newly-built

    def __init__(self, input_id, attention_mask, label, args):
        assert len(label) == len(input_id) == len(attention_mask)
        self.input_id = input_id.cpu().numpy()
        self.attention_mask = attention_mask.cpu().numpy()
        self.label = label
        # self.args=args
        self.train_semi_dataset, self.train_semi_dataloader = self.get_semi_loader(self.input_id, self.attention_mask, self.label, args)
        self.all_label_list = get_labels(label)
        # self.input_id1 = input_id[:, 0, :].cpu().numpy()
        # self.input_id2 = input_id[:, 1, :].cpu().numpy()
        # self.input_id3 = input_id[:, 2, :].cpu().numpy()
        # self.attention_mask1 = attention_mask[:, 0, :].cpu().numpy()
        # self.attention_mask2 = attention_mask[:, 1, :].cpu().numpy()
        # self.attention_mask3 = attention_mask[:, 2, :].cpu().numpy()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # return {'input_id1': self.input_id1[idx], 'input_id2': self.input_id2[idx], 'input_id3': self.input_id3[idx], 'attention_mask1':self.attention_mask1[idx], 'attention_mask2':self.attention_mask2[idx], 'attention_mask3':self.attention_mask3[idx], 'label': self.label[idx]}
        return {'input_id': self.input_id[idx], 'attention_mask': self.attention_mask[idx], 'label': self.label[idx]}

    def get_semi_loader(self, semi_input_ids, semi_input_mask,  semi_label_ids, args):
        semi_input_ids = torch.from_numpy(semi_input_ids)
        semi_input_mask = torch.from_numpy(semi_input_mask)
        semi_label_ids = torch.from_numpy(semi_label_ids)
        semi_data = TensorDataset(semi_input_ids, semi_input_mask, semi_label_ids)
        semi_sampler = SequentialSampler(semi_data)
        semi_dataloader = util_data.DataLoader(semi_data, sampler=semi_sampler, batch_size=args.batch_size)
        return semi_data, semi_dataloader

def get_batch_token(text,tokenizer, max_length=32):
    text=text.tolist()
    token_feat = tokenizer.batch_encode_plus(
        text,
        max_length=max_length,
        return_tensors='pt',
        padding='max_length',
        truncation=True
    )
    return token_feat

def prepare_transformer_input(text,augmentation_1 ,augmentation_2,tokizer):#This step can be placed inside the dataloader

            text1, text2, text3 = text, augmentation_1, augmentation_2
            feat1 = get_batch_token(text1,tokizer)
            feat2 = get_batch_token(text2,tokizer)
            feat3 = get_batch_token(text3,tokizer)

            input_ids = torch.cat( [feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1), feat3['input_ids'].unsqueeze(1)],dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1),feat3['attention_mask'].unsqueeze(1)], dim=1)

            return input_ids.cuda(), attention_mask.cuda()

def explict_augmentation_loader(args,tokenizer):
    train_data = pd.read_csv(os.path.join(args.datapath, args.dataname+".csv"))
    # train_data = pd.read_csv("C:\\Users\\Janko\\Desktop\\sccl-main\\sccl-main\\data\\agnew\\train_charswap_20.csv")
    train_text = train_data[args.text].fillna('.').values
    train_text1 = train_data[args.augmentation_1].fillna('.').values
    train_text2 = train_data[args.augmentation_2].fillna('.').values
    train_label = train_data[args.label].astype(int).values

    input_id, attention_mask=prepare_transformer_input( train_text, train_text1,train_text2,tokenizer)
    all_label_list = get_labels(train_label)
    datasets = TensorDataset( input_id, attention_mask)
    semi_sampler = SequentialSampler(datasets)
    # semi_dataloader =util_data.DataLoader(datasets, sampler=semi_sampler, batch_size=args.train_batch_size)
    train_dataset1 = ExplitAugSamples1(input_id, attention_mask, train_label, args)  # The original dataset
    train_loader1 = util_data.DataLoader(train_dataset1, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # print(train_data[args.label])

    # temp=train_data[args.label].str.replace("[", '')
    # temp=temp.str.replace("]", "")
    # temp=temp.str.replace("\'", "")
    # train_label = temp.astype(int).values

    # print("explict_augmentation_loader:\n")
    # print(temp)

    # print(eval(train_text1))
    # temp = re.findall('\d+',train_text1)
    # train_label =
    # print(train_data[args.label])

    train_dataset = ExplitAugSamples(train_text, train_text1, train_text2, train_label)#The original dataset

    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader1

def virtual_augmentation_loader(args):
    train_data = pd.read_csv(os.path.join(args.datapath, args.dataname+".csv"))
    train_text = train_data[args.text].fillna('.').values
    train_label = train_data[args.label].astype(int).values

    train_dataset = VirtualAugSamples(train_text, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader

def get_labels(labels):
    labels = np.unique(np.array(labels))
    return labels

def unshuffle_loader(args):

    train_data = pd.read_csv(os.path.join(args.datapath, args.dataname+".csv"))
    # train_data = pd.read_csv("C:\\Users\\Janko\\Desktop\\sccl-main\\sccl-main\\data\\agnew\\train_charswap_20.csv")

    train_text = train_data[args.text].fillna('.').values
    # print(train_data[args.label])

    # temp=train_data[args.label].str.replace("[",'')
    # temp=temp.str.replace("]","")
    # temp=temp.str.replace("\'","")
    # train_label = temp.astype(int).values

    train_label = train_data[args.label].astype(int).values
    # print("unshuffle_loader:\n")

    train_dataset = VirtualAugSamples(train_text, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1,drop_last=True)
    return train_loader

class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        # transform = dataset.transform

        # if isinstance(transform, dict):
        #     self.anchor_transform = transform['standard']
        #     self.neighbor_transform = transform['augment']
        # else:
        #     self.anchor_transform = transform
        #     self.neighbor_transform = transform

        # dataset.transform = None
        # self.tok= AutoTokenizer.from_pretrained(tokenizer)
        self.dataset = dataset
        self.indices = indices  # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors + 1]
        assert (self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = list(self.dataset.__getitem__(index))

        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        # anchor['image'] = self.anchor_transform(anchor['image'])
        # neighbor['image'] = self.neighbor_transform(neighbor['image'])

        # output['anchor'] = anchor['image']
        # output['neighbor'] = neighbor['image']
        # anchor[0] = shuffle_tokens(anchor[0], self.tok)
        # neighbor[0] = shuffle_tokens(neighbor[0], self.tok)
        output['anchor'] = anchor[:3]  # source data
        output['neighbor'] = neighbor[:3]  # The nearest neighbor
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])  # All possible neighbors
        # Enhanced samples of the original text
        # Enhance the sample for possible neighbors
        # Enhance the sample for the nearest neighbor
        # output['target'] = anchor['target']
        output['target'] = anchor[-1]
        output['index'] = index
        # output[agu]=agu
        return output